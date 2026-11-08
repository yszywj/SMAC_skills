"""CLI bootstrap: only the bridge and sibling SMACv2 source enter sys.path."""
from bridge.runner import main

if __name__ == "__main__":
    import os
    import sys

    command = sys.argv[1] if len(sys.argv) > 1 else None
    if os.name == "nt" and command in ("run", "suite") and not {"-h", "--help"}.intersection(sys.argv):
        from display_guard.launcher import windows_startup_guard
        with windows_startup_guard():
            main()
    else:
        main()
