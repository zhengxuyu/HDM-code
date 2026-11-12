"""Shared fixtures."""


def load_script(name):
    """Import a file from `scripts/` by path, independent of the working directory.

    `scripts/` is not a package, and `from scripts.x import y` only resolved
    because pytest puts its rootdir on `sys.path` when it is invoked from the
    repository root. Run the same file by absolute path from anywhere and it is
    `ModuleNotFoundError`. code-reviewer hit exactly that and read 1 failed where
    I had read 647 passed; CI runs from the root, so CI would never have caught
    it. The path here is relative to **this file**, which has no such dependency.
    """
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def script_source(name):
    """A script's text, by path rather than by working directory.

    Same reason as `load_script`: `Path("scripts/x.py").read_text()` passes from
    the repository root and raises `FileNotFoundError` anywhere else, and CI only
    ever runs from the root.
    """
    from pathlib import Path

    return (Path(__file__).resolve().parent.parent / "scripts"
            / f"{name}.py").read_text()


def repo_path(*parts):
    """A path inside the repository, resolved from this file rather than from cwd.

    The repository's tests invoked scripts and read sources by paths relative to
    the working directory. They pass when pytest is run from the root -- which is
    what CI does (`uv run --no-sync pytest -q`) -- and fail from anywhere else,
    so CI could never have reported them. code-reviewer asked for them fixed
    deliberately rather than found.
    """
    from pathlib import Path

    return Path(__file__).resolve().parent.parent.joinpath(*parts)
