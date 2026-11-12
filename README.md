<div align="center">

# HDM &nbsp;·&nbsp; a hybrid decision model

**Official implementation of**
### The Value of Thinking: Learning When to Reason in Hybrid Decision Models

[![licence](https://img.shields.io/badge/licence-Apache%202.0-blue)](LICENSE)
[![python](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)
[![paper](https://img.shields.io/badge/paper-preprint-orange)](#citation)
[![blocks](https://img.shields.io/badge/weights-included-success)](weights/)

*One backbone, two decision modes, and a learned block that prices thinking
before either mode runs.*

</div>

---

<div align="center">
<img src="assets/architecture.png" alt="The backbone, the self-routing block, and the two modes" width="88%">
</div>

<div align="center">
<sub><b>Where the block sits.</b> The frozen backbone encodes a typed request with no token
generated. The self-routing block holds the only trained weights: it reads the readout state
and eight statistics of the fast answer, and returns the joint outcome of the two modes, from
which the confidence <code>c<sub>i</sub></code> and the value of thinking <code>v<sub>i</sub></code>
follow. Routing happens once, before either mode runs.</sub>
</div>

---

## What this is

The **fast mode** generates no token: one forward pass encodes the request and the
answer is read from the candidate distribution at the readout position. The **slow
mode** is the backbone's own reasoning, inherited unchanged. The **self-routing
block**, a few hundred kilobytes, estimates what thinking is worth on this question
and escalates only when that value clears a price the caller sets per request.

A question therefore costs either **no generated token** or the **T** a reasoning
pass takes. Nothing in between, and nothing is parsed out of generated text.

> The value of thinking is a log ratio, `log P(slow right) / P(fast right)`, not a
> difference. The reward discounts tokens rather than subtracting them, and a
> difference shrinks exactly where thinking is worth most.

## Routing on a live task

<div align="center">
<img src="assets/demo.gif" alt="One browser task under three policies" width="88%">
</div>

<div align="center">
<sub>One task, three policies, one clock: never thinking, always thinking, and the block
deciding per step. <code>demo/</code> has the harness.</sub>
</div>

## Repository

| | |
|---|---|
| `src/litjev/` | the two modes: typed requests, readouts, the block, the pipeline |
| `src/inference/` | SGLang backend, the decision service, and its standard-library client |
| `src/slime_htm/` | episodic rollouts, the environment protocol, CDPO |
| `weights/` | self-routing blocks fitted on MMLU-Pro, with their fit reports |
| `demo/` | one browser task under three policies, on the clock |
| `docs/` | the API, the schema, how the readouts work, how to reproduce |

## Serving both modes

The server carries the API and three pages that exercise it: `/` answers one typed
request with the fast mode, `/routing` runs the same request at several prices side
by side, and `/film` answers a visual environment frame by frame.
[`docs/deploy.md`](docs/deploy.md) covers installing, attaching a block, running it
on a GPU through SGLang, and reaching it safely from another machine.

```bash
litjev-serve --model Qwen/Qwen3.8-27B \
             --decision-head weights/head-27b-mmlu-pro.safetensors \
             --backend sglang --readout coded --preload
```

`POST /v1/systemone` answers with the fast mode alone. `POST /v1/systemtwo` takes a
routing block in the body and decides for itself:

```json
{"model": "litjev",
 "state": "All customers cannot sign in. No records are lost.",
 "questions": {"impact": {"type": "score", "instructions": "...", "criteria": ["..."]}},
 "routing": {"lambda": 0.0, "budget": 4096}}
```

`lambda` is the price of one escalation. It is **not** baked into the block: one fit
serves every price, so moving it moves the operating point along the cost-accuracy
curve without retraining.

## Routing inside an agent loop

For step-level decisions the engine, the tokenizer and the block sit together, and
only the decision crosses the network:

```bash
python -m inference.decision_service \
       --model <path> --head weights/head-27b-mmlu-pro.safetensors
```

| route | what it returns |
|---|---|
| `POST /fast` | the action distribution, with the value of thinking and the confidence beside it |
| `POST /slow` | reasons first, then reads the same candidates |
| `POST /solve` | lets the slow mode write an action where the environment can take one |
| `GET /health` | which block is loaded, down to a digest of its weights |

`inference.client.DecisionClient` is standard library only, so an environment with
its own dependencies can talk to the service without carrying torch.

## The blocks

| | `head-4b-mmlu-pro` | `head-27b-mmlu-pro` |
|---|---|---|
| backbone | Qwen3.5-4B | Qwen3.8-27B |
| readout / layer | `coded` / `-1`, raw | `coded` / `-1`, raw |
| fitted on | MMLU-Pro, 1,154 questions | MMLU-Pro, 1,154 questions |
| held out | 346, whole categories | 346, whole categories |
| AUROC, where thinking helps | 0.804 | 0.879 |

Fitted once and used unchanged everywhere else. [`weights/README.md`](weights/README.md)
records what they were fitted on, and the two places where a simpler rule scores higher.

## Citation

```bibtex
@article{yu2026valueofthinking,
  title  = {The Value of Thinking: Learning When to Reason in Hybrid Decision Models},
  author = {Yu, Zhengxu and Xu, Ke and Fang, Meng and Ammar, Haitham and
            Luo, Weilin and Wang, Jun},
  year   = {2026},
  note   = {Preprint}
}
```

## Licence

Apache 2.0, see [`LICENSE`](LICENSE). Third-party notices are in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
