"""P0-3: TeacherPairScorer — client of the resident teacher service.

Runs in the SMACv2 ADAPTER environment (no torch): talks to
tree_skills/teacher/service.py over a subprocess pipe (JSONL), keeps the
service alive across the whole run (weights loaded once), and:

- advances per-(episode, agent) observations exactly once per game_loop
  (begin_step sends ALL agents' raw obs in ONE request; duplicate loops
  are rejected client-side so history can never double-append);
- caches the FULL actor logits vector per healer per step, so multiple
  legal heal candidates of one healer cost ONE teacher call, never one
  per candidate;
- scores a (healer, target) candidate by reading the teacher's heal slot
  for that ally (action 6 + ally_index; heal and attack share the integer
  range — selection is by unit type, never by reusing attack mappings);
- never maps the keep extension (native n_actions) or padding slots.

Failure policy is decided by the caller (pair_strict): this scorer raises
on any service/transport/mapping error instead of silently returning
original-looking scores.
"""

import json
import os
import subprocess
import sys

from .action_mapping import ActionMappingError, SmacActionMap

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
SERVICE = os.path.join(HERE, "service.py")
TEACHER_PYTHON = os.path.join(REPO, "tree_skills", ".venv-teacher",
                              "Scripts", "python.exe")


class TeacherServiceError(RuntimeError):
    pass


class TeacherServiceClient:
    def __init__(self, checkpoint, last_token="off", python=TEACHER_PYTHON,
                 startup_timeout=180.0):
        if not os.path.isfile(python):
            raise TeacherServiceError(
                f"teacher python not found at {python}; create "
                "tree_skills/.venv-teacher with torch first")
        self.proc = subprocess.Popen(
            [python, SERVICE, "--checkpoint", checkpoint,
             "--last-token", last_token],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding="utf-8",
            cwd=REPO)
        hello = self._read(startup_timeout)
        if "model_sha256" not in hello:
            raise TeacherServiceError(f"bad service hello: {hello}")
        self.model_sha256 = hello["model_sha256"]
        self.model_args = hello["model_args"]
        self.history_len = hello["history_len"]

    def _read(self, timeout=None):
        # line-based read; a dead service surfaces as EOF
        line = self.proc.stdout.readline()
        if not line:
            err = self.proc.stderr.read()
            raise TeacherServiceError(f"teacher service died: {err[-2000:]}")
        return json.loads(line)

    def _request(self, payload):
        self.proc.stdin.write(json.dumps(payload) + "\n")
        self.proc.stdin.flush()
        reply = self._read()
        if "error" in reply:
            raise TeacherServiceError(
                f"{reply['error']['kind']}: {reply['error']['detail']}")
        return reply

    def init(self, dims):
        return self._request({"op": "init", "dims": dims})

    def reset_episode(self, episode):
        return self._request({"op": "reset", "episode": episode})

    def logits(self, episode, game_loop, agents):
        """agents: [{"agent": int, "obs": [...], "avail": [...]}] ->
        {agent: [logits...]} in ONE batched forward."""
        reply = self._request({"op": "logits", "id": game_loop,
                               "episode": episode, "game_loop": game_loop,
                               "agents": agents})
        return {row["agent"]: row["logits"] for row in reply["results"]}

    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.close()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.terminate()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class TeacherPairScorer:
    """Injected scorer for mode 'teacher-score'.

    The runner hook calls begin_step(episode, game_loop, agents_obs) once
    per real environment step (all agents, one service request). The
    scorer callable then only reads cached logits — one full logits vector
    per healer per step, split locally across that healer's candidates.
    """

    name = "teacher"

    def __init__(self, client, n_agents=10, n_enemies=10):
        self.client = client
        self.action_size = client.model_args["action_size"]
        self.mapping = SmacActionMap(n_agents=n_agents, n_enemies=n_enemies,
                                     teacher_action_size=self.action_size)
        self._logits = {}      # (episode, game_loop, agent) -> [logits]
        self._advanced = set()  # (episode, agent, game_loop) already sent
        self.calls = 0
        self._episode = None
        self._loop = None

    def begin_step(self, episode, game_loop, agents, reset=False):
        """Advance histories and fetch logits for one environment step.

        agents: [{"agent": int, "obs": [...], "avail": [...]}] — raw
        env.get_obs_agent() vectors and native availability masks. Passing
        the same (episode, game_loop) twice raises: history must never
        advance twice for one real step.
        """
        key = (episode, game_loop)
        if (self._episode, self._loop) == key and self._logits:
            raise TeacherServiceError(
                f"begin_step called twice for {key}; refusing to re-advance "
                "teacher history")
        if reset or episode != self._episode:
            self.client.reset_episode(episode)
        self._episode, self._loop = episode, game_loop
        fetched = self.client.logits(episode, game_loop, agents)
        self._logits = {(episode, game_loop, agent): vec
                        for agent, vec in fetched.items()}
        self.calls += 1

    def __call__(self, healer, target_id, context):
        episode, loop = self._episode, self._loop
        if episode is None or loop is None:
            raise TeacherServiceError("begin_step was never called")
        agent = int(healer[1:])
        logits = self._logits.get((episode, loop, agent))
        if logits is None:
            raise TeacherServiceError(
                f"no cached teacher logits for {healer} at ({episode}, {loop})")
        ally = int(target_id[1:])
        slot = self.mapping.heal_slot(ally)  # raises on out-of-range ally
        if slot >= len(logits):
            raise ActionMappingError(f"heal slot {slot} beyond teacher head")
        self.mapping.validate_teacher_slots([slot])  # keep/padding guard
        return float(logits[slot])
