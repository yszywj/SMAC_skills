"""Recover frozen local studies with new evidence directories and owned-process timeouts.

No controller source is changed. Original runs and plans are never overwritten;
failed/rejected/missing outcomes are never assigned a zero reward.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from decision_tree.local import study
from decision_tree.local.dataset import read_run, build_dataset
from decision_tree.local.runtime import digest, load_policy


class LaunchGate:
    """Shared finite launch budget and latched environmental circuit breaker."""
    def __init__(self, max_games=None):
        self.max_games, self.started, self.environment_streak = max_games, 0, 0
        self.stopped_reason = None
        self.lock = threading.Lock()

    def reserve(self):
        with self.lock:
            if self.stopped_reason or (self.max_games is not None and self.started >= self.max_games):
                return False
            self.started += 1
            return True

    def outcome(self, environment_failure=False):
        with self.lock:
            self.environment_streak = self.environment_streak + 1 if environment_failure else 0
            if self.environment_streak >= 3:
                self.stopped_reason = 'three_consecutive_environment_failures'

    def halt(self, reason):
        with self.lock:
            self.stopped_reason = reason

    def describe(self):
        with self.lock:
            return {'new_games_started': self.started, 'max_games': self.max_games,
                    'consecutive_environment_failures': self.environment_streak,
                    'circuit_open': self.stopped_reason is not None, 'stopped_reason': self.stopped_reason}


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    delays = (0.025, 0.05, 0.1, 0.2, 0.4, 0.8)
    for attempt in range(len(delays) + 1):
        try:
            tmp.replace(path)
            return
        except OSError as exc:
            # Windows readers may temporarily hold a handle without delete sharing.
            winerror = getattr(exc, 'winerror', None)
            retryable = winerror in (5, 32, 33) or (
                os.name == 'nt' and winerror is None and isinstance(exc, PermissionError))
            if not retryable or attempt == len(delays):
                raise
            time.sleep(delays[attempt])


def write_progress(path, value):
    """A display snapshot can be rebuilt from the retained run directories."""
    try:
        atomic_json(path, value)
    except OSError as exc:
        print(f'Could not publish progress snapshot {path}: {exc}', file=sys.stderr, flush=True)
        return False
    return True


def checked_child(path, parent):
    path, parent = Path(path).absolute(), Path(parent).resolve()
    if path.resolve().parent != parent or path.is_symlink():
        raise ValueError('Recovery path must be an ordinary direct child: ' + str(path))
    return path


def failure_kind(path, manual=False):
    """Only explicit environment failures or documented watchdog/manual hangs retry."""
    path = Path(path)
    if not path.exists():
        return 'unstarted'
    if manual:
        return 'manual_hang'
    timeout = path/'timeout.json'
    if timeout.is_file():
        try:
            record = study.read(timeout)
            command = study.read(path.parent/(path.name+'.command.json'))
            if record.get('owned_process') is True and record.get('command_sha256') == digest(command):
                return 'watchdog_timeout'
        except (OSError, ValueError, TypeError):
            pass
    try:
        summary = study.read(path/'summary.json')
    except (OSError, ValueError):
        return 'incomplete_unknown'
    if summary.get('status') == 'complete':
        return 'complete_needs_audit'
    error = str(summary.get('error', ''))
    # Do not retry model, action legality, feature or audit failures.
    lowered = error.lower()
    if any(x in lowered for x in ('illegal action', 'candidate', 'context mismatch', 'model', 'audit')):
        return 'logic_or_validation_error'
    if error.startswith(('ConnectError:', 'ConnectionError:', 'ConnectionRefusedError:',
                         'ConnectionResetError:', 'ConnectionAbortedError:', 'WebSocketConnectionClosedException:')):
        return 'environment_transport'
    if any(x in lowered for x in ('sc2 transport', 'sc2 execution failure', 'sc2 restart',
                                  'unable to restart sc2', 'failed to restart sc2',
                                  'websocket is already closed', 'connection to sc2 lost')):
        return 'environment_transport'
    return 'logic_or_unknown_failure'


def successful(path, manifest, policy):
    """Successful directories must also pass the full current frozen audit."""
    run = read_run(path)
    if run['manifest.json'] != manifest or run['run.json']['local_policy']['policy'] != policy:
        raise ValueError('Completed recovery run input mismatch')
    return run


def terminate_owned(process, command, record):
    """The live Popen handle prevents PID-reuse ambiguity; never discover/kill globally."""
    if process.args != command or process.pid != record['pid'] or digest(command) != record['command_sha256']:
        raise ValueError('Refusing to terminate a process not created for this exact command')
    if process.poll() is not None:
        return
    if os.name == 'nt':
        result = subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, timeout=20,
                                creationflags=subprocess.CREATE_NO_WINDOW)
        if result.returncode and process.poll() is None:
            raise RuntimeError('Owned process-tree termination failed: ' + result.stdout[-2000:])
    else:
        os.killpg(process.pid, signal.SIGKILL)
    process.wait(timeout=20)


def run_attempt(protocol, manifest, policy, destination, timeout_seconds=120):
    destination = Path(destination)
    study.check_sources(protocol['sources'])
    if destination.exists():
        raise ValueError('Refusing to overwrite an existing recovery attempt')
    command = study.command(protocol, manifest, policy, destination)
    command_path = destination.parent/(destination.name+'.command.json')
    study.save_once(command_path, command)
    console = destination.parent/(destination.name+'.console.log')
    process_record = destination.parent/(destination.name+'.process.json')
    kwargs = {'cwd': study.ROOT, 'env': {**os.environ, 'PYTHONUNBUFFERED': '1'},
              'stdin': subprocess.DEVNULL, 'stderr': subprocess.STDOUT}
    if os.name == 'nt':
        kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs['start_new_session'] = True
    with console.open('x', encoding='utf-8') as stream:
        process = subprocess.Popen(command, stdout=stream, **kwargs)
        record = {'pid': process.pid, 'command_sha256': digest(command),
                  'started_unix': time.time(), 'owned_process': True}
        study.save_once(process_record, record)
        try:
            returncode = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            # Write timeout evidence even if Windows refuses termination.
            destination.mkdir(exist_ok=True)
            timeout = {**record, 'timeout_seconds': timeout_seconds,
                       'reason': 'wall-clock watchdog; no reward assigned', 'termination': 'requested'}
            atomic_json(destination/'timeout.json', timeout)
            try:
                terminate_owned(process, command, record)
            except Exception as exc:
                timeout['termination'] = 'failed: ' + str(exc)
                atomic_json(destination/'timeout.json', timeout)
                raise
            timeout['termination'] = 'owned process tree terminated'
            atomic_json(destination/'timeout.json', timeout)
            return {'status': 'timeout', 'path': str(destination)}
    return {'status': 'exited', 'returncode': returncode, 'path': str(destination)}


def inputs_for_arm(arm, directory):
    original = checked_child(arm['path'], directory)
    if original.name != arm['name']:
        raise ValueError('Plan name/path mismatch')
    mp, pp = Path(arm['manifest']).resolve(), Path(arm['policy']).resolve()
    if mp.parent != directory/'inputs' or pp.parent != directory/'inputs':
        raise ValueError('Arm inputs escaped the frozen inputs directory')
    manifest, policy = study.read(mp), load_policy(pp)
    if digest(manifest) != arm['manifest_sha256'] or digest(policy) != arm['policy_sha256']:
        raise ValueError('Planned input changed')
    selection = policy['selection']
    if (selection['mode'] != 'forced' or selection['candidate_id'] != arm['candidate_id']
            or selection['anchor_step'] != arm['anchor_step']):
        raise ValueError('Planned intervention binding changed')
    return original, manifest, policy


def recover_arm(arm, directory, protocol, hung_runs=(), only_failed=False, timeout_seconds=120, gate=None):
    gate = gate or LaunchGate()
    original, manifest, policy = inputs_for_arm(arm, directory)
    paths = [original] + [checked_child(original.with_name(original.name+f'.recovery-{i:03d}'), directory)
                          for i in (1, 2)]
    attempts, chosen = [], original
    # Reuse any already-successful attempt; never touch its directory.
    for path in paths:
        if not path.exists():
            continue
        try:
            successful(path, manifest, policy)
            return {**arm, 'path': str(path), 'recovery_status': 'reused', 'attempts': attempts}
        except (OSError, ValueError, KeyError, TypeError) as exc:
            kind = failure_kind(path, manual=path == original and arm['name'] in hung_runs)
            attempts.append({'path': str(path), 'kind': kind, 'audit_error': str(exc)})
            chosen = path
            if kind not in ('manual_hang', 'watchdog_timeout', 'environment_transport'):
                return {**arm, 'path': str(path), 'recovery_status': 'rejected', 'attempts': attempts}
    if not original.exists() and only_failed:
        return {**arm, 'recovery_status': 'pending_unstarted', 'attempts': attempts}
    for path in paths[1:]:
        # Sidecars from an interrupted launch also reserve an attempt name.
        if path.exists() or any((path.parent/(path.name+s)).exists()
                                for s in ('.command.json', '.console.log', '.process.json')):
            continue
        if not gate.reserve():
            return {**arm, 'path': str(chosen), 'recovery_status': 'deferred', 'attempts': attempts,
                    'launch_gate': gate.describe()}
        try:
            launch = run_attempt(protocol, arm['manifest'], arm['policy'], path, timeout_seconds)
        except Exception:
            # In particular, never continue launching after a watchdog cannot
            # confirm that its own process tree has terminated.
            gate.halt('launch_or_process_cleanup_error')
            raise
        chosen = path
        try:
            successful(path, manifest, policy)
            gate.outcome()
            return {**arm, 'path': str(path), 'recovery_status': 'recovered',
                    'attempts': attempts + [launch]}
        except (OSError, ValueError, KeyError, TypeError) as exc:
            kind = failure_kind(path)
            attempts.append({**launch, 'kind': kind, 'audit_error': str(exc)})
            gate.outcome(environment_failure=kind in ('watchdog_timeout', 'environment_transport'))
            if kind not in ('watchdog_timeout', 'environment_transport'):
                break
    return {**arm, 'path': str(chosen), 'recovery_status': 'rejected', 'attempts': attempts}


def recover(directory, output, *, reason, hung_runs=(), only_failed=False, workers=1, timeout_seconds=120,
            max_games=None):
    directory, output = Path(directory).resolve(), Path(output).resolve()
    artifact_root = (study.ROOT/'decision_tree/artifacts').resolve()
    if not directory.is_relative_to(artifact_root) or not output.is_relative_to(artifact_root) or output == directory:
        raise ValueError('Study and separate recovery report directory must be inside repository artifacts')
    if type(workers) is not int or workers not in (1, 2) or not 1 <= timeout_seconds <= 3600 or not reason.strip():
        raise ValueError('Require workers 1/2, timeout 1..3600, and a recovery reason')
    if max_games is not None and (type(max_games) is not int or max_games < 0):
        raise ValueError('max_games must be a nonnegative integer')
    protocol = study.read(directory/'protocol.json')
    manifest, policy = study.read(directory/'manifest.json'), load_policy(directory/'base_policy.json')
    plan = study.read(directory/'plan.json')
    if (protocol.get('kind') != 'local_patch_collection' or digest(manifest) != protocol['manifest_sha256']
            or digest(policy) != protocol['policy_sha256'] or policy['selection']['mode'] != 'base'):
        raise ValueError('Frozen collection inputs changed')
    if protocol['sources'] != study.source_hashes(manifest):
        raise ValueError('Frozen source set or content changed')
    study.check_sources(protocol['sources'])
    base = successful(directory/'base', manifest, policy)
    if Path(plan['base_path']).resolve() != directory/'base' or plan['base_hashes'] != base['hashes']:
        raise ValueError('BASE differs from the frozen candidate plan')
    names = {a['name'] for a in plan['arms']}
    if len(names) != len(plan['arms']) or set(hung_runs) - names:
        raise ValueError('Duplicate arm or unknown manually identified hung run')
    for arm in plan['arms']:
        inputs_for_arm(arm, directory)
    recovery_protocol = {'schema_version': 1, 'kind': 'local_campaign_recovery', 'workers': workers,
                         'reason': reason, 'manual_hung_runs': sorted(set(hung_runs)),
                         'only_failed': only_failed, 'timeout_seconds': timeout_seconds, 'max_new_attempts_per_arm': 2,
                         'max_games': max_games, 'environment_circuit_breaker': 3,
                         'study_directory': str(directory), 'original_protocol_sha256': digest(protocol),
                         'original_plan_sha256': digest(plan), 'sources': protocol['sources'],
                         'recovery_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    output.mkdir(parents=True, exist_ok=True)
    study.save_once(output/'recovery_protocol.json', recovery_protocol)
    lock = output/'.recovery.lock'
    with lock.open('x', encoding='utf-8') as stream:
        stream.write(str(os.getpid()))
    results = {}
    gate = LaunchGate(max_games)
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(recover_arm, a, directory, protocol, hung_runs, only_failed, timeout_seconds, gate): a
                       for a in plan['arms']}
            for n, future in enumerate(as_completed(futures), 1):
                arm = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {**arm, 'recovery_status': 'rejected', 'error': type(exc).__name__ + ': ' + str(exc)}
                results[arm['name']] = result
                counts = {kind: sum(r['recovery_status'] == kind for r in results.values())
                          for kind in ('reused', 'recovered', 'rejected', 'pending_unstarted', 'deferred')}
                write_progress(output/'progress.json', {'processed': n, 'total': len(plan['arms']), 'counts': counts,
                                                        'launch_gate': gate.describe(),
                                                        'arms': [results[a['name']] for a in plan['arms'] if a['name'] in results]})
                print(f'Recovery {n}/{len(plan["arms"])} {result["recovery_status"]}: {arm["name"]}', flush=True)
        study.check_sources(protocol['sources'])
        effective = {**deepcopy(plan), 'arms': [results[a['name']] for a in plan['arms']],
                     'recovery_protocol_sha256': digest(recovery_protocol)}
        study.save_once(output/'effective_plan.json', effective)
        data = build_dataset(directory/'base', effective['arms'], effective)
        study.save_once(output/'paired.json', data)
        result = {'counts': counts if results else {}, 'accepted_rows': len(data['rows']),
                  'launch_gate': gate.describe(),
                  'rejected_pairs': len(data['rejected']), 'complete_anchor_rows': sum(r['anchor_complete'] for r in data['rows']),
                  'dataset': str(output/'paired.json'), 'original_plan_preserved': True, 'missing_rewards_imputed': False}
        study.save_once(output/'result.json', result)
        return result
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--reason', required=True)
    parser.add_argument('--hung-run', action='append', default=[])
    parser.add_argument('--only-failed', action='store_true')
    parser.add_argument('--workers', type=int, choices=(1, 2), default=1)
    parser.add_argument('--timeout-seconds', type=int, default=120)
    parser.add_argument('--max-games', type=int, help='limit newly launched games; reused evidence does not count')
    args = parser.parse_args()
    print(json.dumps(recover(args.study, args.output, reason=args.reason, hung_runs=args.hung_run,
                             only_failed=args.only_failed, workers=args.workers,
                             timeout_seconds=args.timeout_seconds, max_games=args.max_games), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
