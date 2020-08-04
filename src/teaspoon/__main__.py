"""
Entrypoint module, in case you use `python -mteaspoon`.


Why does this file exist, and why __main__? For more info, read:

- https://www.python.org/dev/peps/pep-0338/
- https://docs.python.org/2/using/cmdline.html#cmdoption-m
- https://docs.python.org/3/using/cmdline.html#cmdoption-m
"""
import sys

from teaspoon.cli import main

if __name__ == "__main__":
    sys.exit(main())
