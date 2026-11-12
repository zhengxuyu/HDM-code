"""What the launchers actually pass, rendered by running them.

Twice now something has been built that the cluster could not reach. The proposal
prompt existed and no test read it; `--fast-only` existed and `collect.sbatch`
built a fixed argument list with no way to pass it. Both times the code was
right, the path to it was not, and every test passed.

So these run the sbatch file with `uv` and `nvidia-smi` stubbed onto PATH, and
assert on the argument list that would have reached the CLI. Then they hand that
list to the CLI's own parser, which is what turns "the flag is spelled right"
into "the flag exists".
"""
import shlex
import subprocess

import pytest

from litjev.data.datasets import select_questions as select

COLLECT = "scripts/collect.sbatch"


def rendered(tmp_path, root, script=COLLECT, **environment):
    """The command line the sbatch would have run, as a list of arguments."""
    stubs = tmp_path / "stubs"
    stubs.mkdir(parents=True)
    recorded = tmp_path / "argv"
    for name in ("nvidia-smi",):
        (stubs / name).write_text("#!/bin/sh\nexit 0\n")
        (stubs / name).chmod(0o755)
    # The sbatch calls "${UV_BIN:-uv}", so the stub goes in by that name.
    runner = stubs / "uv-stub"
    runner.write_text(f'#!/bin/sh\nprintf "%s\\n" "$@" > "{recorded}"\nexit 0\n')
    runner.chmod(0o755)

    work = tmp_path / "submit"
    work.mkdir()
    result = subprocess.run(
        ["bash", str(root / script)], cwd=work,
        env={"PATH": f"{stubs}:/usr/bin:/bin", "HOME": str(tmp_path),
             "SLURM_SUBMIT_DIR": str(work), "UV_BIN": str(runner),
             "SLURM_JOB_ID": "999", **environment},
        capture_output=True, text=True)
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    return recorded.read_text().splitlines()


@pytest.fixture
def model():
    return {"LITJEV_MODEL": "/models/qwen"}


def test_the_model_is_required_rather_than_defaulted(tmp_path, pytestconfig):
    """A collection against the wrong checkpoint is worse than one that did not run.

    Stubbed like the others: without them this dies at `nvidia-smi` and the test
    passes for a reason that has nothing to do with the model.
    """
    with pytest.raises(AssertionError) as raised:
        rendered(tmp_path, pytestconfig.rootpath)
    assert "LITJEV_MODEL" in str(raised.value)


def test_fast_only_is_reachable_from_the_launcher(tmp_path, pytestconfig, model):
    """It was not, and the flag might as well not have existed on the cluster."""
    without = rendered(tmp_path / "a", pytestconfig.rootpath, **model)
    assert "--fast-only" not in without, "it must not be on unless asked"

    with_flag = rendered(tmp_path / "b", pytestconfig.rootpath, LITJEV_FAST_ONLY="1", **model)
    assert "--fast-only" in with_flag


def test_the_stride_is_reachable_and_defaults_to_every_question(tmp_path, pytestconfig, model):
    """A pass meant to speak to an existing record set has to take the same
    questions; records-4b-27063 is every eighth."""
    default = rendered(tmp_path / "a", pytestconfig.rootpath, **model)
    assert default[default.index("--stride") + 1] == "1"

    eighth = rendered(tmp_path / "b", pytestconfig.rootpath, LITJEV_STRIDE="8", **model)
    assert eighth[eighth.index("--stride") + 1] == "8"


def test_the_shard_offset_is_computed_per_array_task(tmp_path, pytestconfig, model):
    third = rendered(tmp_path / "a", pytestconfig.rootpath,
                     SLURM_ARRAY_TASK_ID="3", LITJEV_SHARD="500", **model)
    assert third[third.index("--offset") + 1] == "1500"
    assert third[third.index("--limit") + 1] == "500"


def test_the_unified_render_is_reachable_from_the_launcher(tmp_path, pytestconfig, model):
    """The flag the layer ablation needs, and the third of these to be caught late.

    It has to run on *this* launcher rather than `collect_sglang.sbatch`: the
    unified render is what a served head is read under, and the engine that path
    uses returns only the final hidden layer, so a sweep over `LITJEV_LAYERS`
    can happen nowhere else. Off unless asked, because it changes what the fast
    pass reads and `records-4b-27063` was collected without it.
    """
    import argparse
    from unittest.mock import patch

    without = rendered(tmp_path / "a", pytestconfig.rootpath, **model)
    assert "--unified-prompt" not in without, "it must not be on unless asked"

    passed = rendered(tmp_path / "b", pytestconfig.rootpath, LITJEV_UNIFIED_PROMPT="1",
                      LITJEV_LAYERS="-1,16,24", LITJEV_BUDGET="4096", **model)
    assert "--unified-prompt" in passed
    # The three the ablation needs arrive together, since any one of them missing
    # gives a collection that cannot answer the question it was run for.
    assert "--layers=-1,16,24" in passed
    assert passed[passed.index("--budget") + 1] == "4096"

    # Then through the CLI's own parser, which is what turns a spelling into a flag.
    arguments = passed[passed.index("litjev-collect") + 1:]
    from litjev import cli

    seen = {}
    original = argparse.ArgumentParser.parse_args

    def capture(self, *rest, **keywords):
        seen["arguments"] = original(self, *rest, **keywords)
        raise SystemExit(0)

    with patch.object(argparse.ArgumentParser, "parse_args", capture), \
            patch("sys.argv", ["litjev-collect", *arguments]), pytest.raises(SystemExit):
        cli.collect()
    assert seen["arguments"].unified_prompt is True
    assert seen["arguments"].layers == "-1,16,24"
    assert seen["arguments"].budget == 4096


def test_every_argument_it_passes_is_one_the_cli_accepts(tmp_path, pytestconfig, model):
    """The half a spelling check cannot do. A flag renamed in the CLI leaves the
    launcher passing something that parses as nothing until a job dies on it."""
    import argparse
    from unittest.mock import patch

    passed = rendered(tmp_path, pytestconfig.rootpath, LITJEV_FAST_ONLY="1",
                      LITJEV_STRIDE="8", **model)
    # Drop `run --locked --no-sync litjev-collect`, which is uv's and not the CLI's.
    arguments = passed[passed.index("litjev-collect") + 1:]
    assert arguments, shlex.join(passed)

    from litjev import cli

    seen = {}

    def capture(self, *rest, **keywords):
        seen["arguments"] = original(self, *rest, **keywords)
        raise SystemExit(0)

    original = argparse.ArgumentParser.parse_args
    with patch.object(argparse.ArgumentParser, "parse_args", capture), \
            patch("sys.argv", ["litjev-collect", *arguments]), pytest.raises(SystemExit):
        cli.collect()
    assert seen["arguments"].fast_only is True
    assert seen["arguments"].stride == 8
    assert seen["arguments"].model == "/models/qwen"


def test_array_shards_tile_the_subset_a_single_task_would_take():
    """They did not. Three tasks at SHARD=500 and stride 8 covered 625 of the 1500
    questions a single task takes, duplicated 375, and reported nothing wrong --
    a collection that looks complete and is a different, smaller sample."""
    questions = list(range(12000))
    stride, shard, tasks = 8, 500, 3
    single = select(questions, 0, stride, shard * tasks)
    shards = [select(questions, task * shard, stride, shard) for task in range(tasks)]

    assert sum(len(part) for part in shards) == len(single), "shards must not overlap"
    assert [q for part in shards for q in part] == single, "and must tile it in order"
    assert len(single) == shard * tasks


def test_a_stride_of_one_is_what_it_always_was():
    """Every record collected so far is stride 1 or offset 0, where the two
    readings agree, so no recorded number moves."""
    questions = list(range(100))
    assert select(questions, 30, 1, 10) == list(range(30, 40))
    assert select(questions, 0, 8, 5) == [0, 8, 16, 24, 32]


def test_every_launcher_names_a_partition(pytestconfig):
    """This cluster has no default, so one that names none is refused at submission
    with "No partition specified" -- after a sync, an ssh and a wait."""
    missing = [path.name for path in sorted((pytestconfig.rootpath / "scripts").glob("*.sbatch"))
               if "#SBATCH --partition" not in path.read_text()]
    assert not missing, missing


def test_a_gpu_job_does_not_ask_for_the_cpu_partition(pytestconfig):
    """It would queue behind CPU work and then fail for want of a device."""
    for path in sorted((pytestconfig.rootpath / "scripts").glob("*.sbatch")):
        text = path.read_text()
        if "#SBATCH --gres=gpu" in text:
            assert "#SBATCH --partition=cpu" not in text, path.name


def test_a_run_says_what_it_selected_including_when_it_is_short():
    """27862 passed --limit 300 against a 198-question set and the log never named
    either number, so the batch count had to be worked out backwards."""
    from litjev.data.datasets import describe_selection

    short = describe_selection(198, 198, 0, 1, 300)
    assert "198 of 198" in short and "limit 300" in short
    assert "fewer than asked for" in short, "a short selection must say so"
    # Not "the set is smaller": bench_routing filters by category after
    # selecting, so the count can be short for a reason the line cannot know.
    assert "the set is smaller" not in short

    full = describe_selection(12032, 1500, 0, 8, 1500)
    assert "1500 of 12032" in full and "stride 8" in full
    assert "fewer than asked for" not in full


def test_the_code_offset_is_reachable_from_the_launcher(tmp_path, pytestconfig, model):
    """The rotation run is two backbones of GPU time; the flag existing and the
    launcher not passing it is how the last three faults here worked."""
    without = rendered(tmp_path / "a", pytestconfig.rootpath, **model)
    assert "--code-offset" not in without

    rotated = rendered(tmp_path / "b", pytestconfig.rootpath, LITJEV_CODE_OFFSET="3", **model)
    assert rotated[rotated.index("--code-offset") + 1] == "3"


def test_a_result_says_which_backend_produced_it():
    """The two route the same question differently, which is why crossing them is
    forbidden here -- and Table 1's file records the device, the budget and the
    lambdas, and not this. Its fast column's backend is not recoverable."""
    import inspect

    from litjev import cli

    for command in (cli.bench_routing, cli.collect):
        source = inspect.getsource(command)
        assert '"backend"' in source, f"{command.__name__} records no backend"
