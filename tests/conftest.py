"""Ensures `from src...` imports resolve when pytest is invoked from any
working directory, by inserting the project root onto sys.path."""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
