"""Optional existing dependency directories for a selected teacher Python.

This does not install anything or replace an environment. Extra packages
are appended so the selected Python's own torch/numpy remain preferred.
"""

import os
import sys


def add_teacher_dependencies():
    for entry in os.environ.get("TREE_SKILLS_TEACHER_DEPENDENCIES", "").split(os.pathsep):
        if not entry:
            continue
        path = os.path.abspath(entry)
        if not os.path.isdir(path):
            raise ValueError("teacher dependency directory does not exist: " + path)
        if path not in sys.path:
            sys.path.append(path)
