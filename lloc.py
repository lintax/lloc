#!/usr/bin/env python3
"""Entry point wrapper: python lloc.py <command>  (equivalent to python -m lloc)."""
import sys

if sys.version_info < (3, 11):
    try:
        import tomli  # noqa: F401  (dev-environment fallback used by lloc.config)
    except ModuleNotFoundError:
        sys.exit("lloc requires Python 3.11+ (tomllib)")

from lloc.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
