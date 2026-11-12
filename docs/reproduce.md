# Reproducing the reported numbers

Every experiment resolves its data through `src/litjev/datasets.py`, which pins
each source and records a content digest. Check the digest before comparing
numbers: a run whose manifest differs from the one below read different
questions, and its results are not comparable to ours.

## 1. Environment

```bash
uv sync --locked
uv run python -m pytest -q          # 63 tests, no GPU, no network
```

`uv sync` installs only what the lockfile pins, which means it removes three
things this repo needs. Restore all of them after any sync, or the tests fail to
collect and the backbone runs a slow fallback:

```bash
uv pip install -e ".[dev,games]"        # pytest, and gymnasium for two test modules
uv pip install flash-linear-attention   # the hybrid backbone's linear-attention layers
```

`flash-linear-attention` is not in the lockfile at all. Without it the backbone
still runs and still gives the same answers, but three quarters of its layers
fall back to a reference implementation, so any wall-clock measurement is
meaningless. The fused `causal_conv1d` kernel is separate again and has to be
built where `nvcc` exists; see `scripts/build_kernel.sbatch`.

## 2. Data

No GPU is needed to resolve the benchmarks and print their manifests:

```bash
uv run litjev-datasets --output datasets.json
```

| Benchmark | Source | Pinned at | Account needed |
| --- | --- | --- | --- |
| MMLU-Pro | `TIGER-Lab/MMLU-Pro` | `--dataset-revision` | no |
| AGIEval | `github.com/ruixiangcui/AGIEval`, `data/v1_1` | commit `84ab72d9` | no |
| GPQA-Diamond | `Idavidrein/gpqa`, config `gpqa_diamond` | `--dataset-revision` | **yes** |

Expected AGIEval manifest, English multiple-choice tasks only:

```
questions 2340   tasks aqua-rat, logiqa-en, lsat-ar, lsat-lr, lsat-rc, sat-en, sat-math
skipped   0      digest 23fb80787516a6551fa847f0f2e8f21a37261a163f728c95acbbb32c8cab9efe
```

GPQA is gated. Accept the terms at
<https://huggingface.co/datasets/Idavidrein/gpqa> and run `huggingface-cli
login`; without that the loader stops with an explanation rather than a stack
trace. AGIEval's two cloze tasks and its Chinese tasks are excluded: the fast
mode reads an answer off a candidate set, so cloze does not apply, and the
Chinese tasks ask a language-transfer question this paper does not.

Determinism notes. AGIEval question ids are the task name plus the line index
in the pinned file, so they are stable across machines. GPQA stores the correct
answer in its own column, so the four options are shuffled per question with a
seed recorded in the manifest; changing the seed changes the digest.

## 3. Main result, MMLU-Pro

Collect records, train the head, then benchmark the policies. The head is
trained once and every later evaluation reuses it.

```bash
uv run litjev-collect --model "$MODEL" --device-map cuda:0 \
  --dataset mmlu-pro --split test --stride 8 --limit 1500 \
  --layers=-1,16,24 --budget 1024 --output results/records.npz

uv run litjev-train-head --records results/records.npz \
  --objective cdpo --train-seed 0 --output results/head.safetensors

uv run litjev-bench-routing --model "$MODEL" --device-map cuda:0 \
  --decision-head results/head.safetensors --dataset mmlu-pro \
  --split test --stride 8 --limit 1500 \
  --categories chemistry,economics,psychology \
  --lambdas 0.1,0.05,0,-0.05,-0.1 --budget 1024 \
  --batch-size 10 --output results/bench-b10.json
```

`--categories` is applied after `--stride` and `--limit`, so the 346 questions
benchmarked are exactly the held-out categories of the training report. Repeat
with `--batch-size 1` for per-request latency; batched decoding runs until the
longest sequence in the batch finishes, so the two answer different questions.

## 4. Transfer, AGIEval and GPQA-Diamond

Same head, benchmarks it never saw, no retraining:

```bash
DECISION_HEAD=results/head.safetensors MODEL="$MODEL" \
  sbatch scripts/transfer.sbatch
```

The script writes the dataset manifests beside the results and runs both batch
sizes for each benchmark. Set `RUN_GPQA=0` to skip the gated dataset.

## 5. What each run records

`litjev-bench-routing` writes accuracy, generated tokens and wall-clock seconds
per policy, plus the dataset manifest, the selected question ids, the head
metadata and the device. Those fields are what a comparison needs: a report
that does not carry the same dataset digest is measuring something else.
