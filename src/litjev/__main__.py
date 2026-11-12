"""`python -m litjev` is `litjev-serve`, for a tree that was not installed.

Jobs reach the cluster by rsync into a fresh directory (`scripts/sync.sh`), which
has no virtual environment in it, so `uv run` has nothing to run and the console
scripts do not exist. The other cluster scripts work around that with
`$HOME/htm-env/bin/python` and `PYTHONPATH=$PWD/src`; this gives them an entry
point to call that is the same one `litjev-serve` calls.
"""

from litjev.cli import serve

if __name__ == "__main__":
    serve()
