"""Stand-ins for the cluster, so the scripts that reach it can be run in a test.

`sync.sh` and `launch.sh` are the two pieces of this repository whose failures are
invisible until a run needs them, and both of them spend most of their lines on
`ssh` and `rsync`. Stubbing those two commands onto PATH is what makes the rest
executable: a test can assert on the commands the script built, which is the part
the script is responsible for, and know that nothing left the machine.

What the cluster then does with those commands is a different question and this
cannot answer it. That half is a submission.
"""


def network_stubs(where, ssh_cases=()):
    """`ssh` and `rsync` on a PATH of their own, appending every call to a log.

    `ssh_cases` are extra `case` branches, most specific first, for a test that
    needs the cluster to answer differently -- an absent directory, a failing
    check. Each is a `(pattern, body)` pair in `sh` syntax.
    """
    # Outside `where`, which is a git checkout in these tests: an untracked stub
    # in it changes the tree the script is being asked to identify.
    beside = where.parent / f"{where.name}-network"
    beside.mkdir(exist_ok=True)
    binaries = beside / "stubs"
    binaries.mkdir(exist_ok=True)
    log = beside / "reached"
    cases = "".join(f"  {pattern}) {body} ;;\n" for pattern, body in ssh_cases)
    for name in ("ssh", "rsync"):
        stub = binaries / name
        # The two substitutions the scripts read a value out of are answered, so
        # that a run which gets this far proceeds rather than failing elsewhere
        # and passing a test for the wrong reason.
        stub.write_text(
            f'#!/bin/sh\necho "{name} $*" >> "{log}"\n'
            'case "$*" in\n'
            f'{cases}'
            '  *--parsable*) echo 999999 ;;\n'
            '  *pwd*) echo /remote/hde ;;\n'
            'esac\nexit 0\n')
        stub.chmod(0o755)
    return binaries, log


def environment(binaries, home, **extra):
    """A PATH with the stubs first and nothing inherited from the test runner."""
    return {"PATH": f"{binaries}:/usr/bin:/bin", "HOME": str(home), **extra}
