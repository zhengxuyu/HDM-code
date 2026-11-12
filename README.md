# HDM: a hybrid decision model

Official implementation of **The Value of Thinking: Learning When to Reason in
Hybrid Decision Models**.

![The backbone, the self-routing block, and the two modes](assets/architecture.png)

The frozen backbone encodes a typed request with no token generated. The
self-routing block, which holds the only trained weights, reads the readout state
and eight statistics of the fast answer and returns the joint outcome of the two
modes, from which the confidence and the value of thinking follow. Routing happens
once: the model either returns the cached fast answer or decodes with the same
backbone.

One language model with two decision modes and a learned block that chooses
between them before either runs.

The **fast mode** generates no token: the model encodes the request in a single
forward pass and the answer is read from the candidate distribution at the
readout position. The **slow mode** is the backbone's own reasoning, inherited
unchanged. A **self-routing block** of a few hundred kilobytes reads the hidden
state that pass has already produced, estimates what thinking is worth on this
question, and escalates only when that value exceeds a price the caller sets per
request. A question therefore costs either no generated token or the T a
reasoning pass takes.

This repository holds three things:

- **the inference service**, which serves both modes from one backbone over an
  HTTP API, with the block applied beside the engine
- **the blocks we fitted on MMLU-Pro**, in `weights/`, used unchanged on every
  other benchmark
- **the training code**, both the supervised fit for single-turn questions and
  CDPO, the reinforcement-learning recipe for episodic tasks where no
  per-decision label exists

## Layout

```
src/litjev/      the two modes: typed requests, readouts, the block, the pipeline
src/inference/   SGLang backend, the decision service, and its client
src/slime_htm/   episodic rollouts, the environment protocol, CDPO
weights/         self-routing blocks fitted on MMLU-Pro, with their fit reports
docs/            the API, the schema, how the readouts work, how to reproduce
tests/
```

## Serving both modes

```bash
litjev serve --model Qwen/Qwen3.8-27B \
             --decision-head weights/head-27b-mmlu-pro.safetensors \
             --backend sglang --readout coded --preload
```

`POST /v1/systemone` answers with the fast mode alone. `POST /v1/systemtwo`
takes a routing block in the body and decides for itself:

```json
{"model": "litjev",
 "state": "All customers cannot sign in. No records are lost.",
 "questions": {"impact": {"type": "score", "instructions": "...", "criteria": [...]}},
 "routing": {"lambda": 0.0, "budget": 4096}}
```

`lambda` is the price of one escalation. It is not baked into the block: one fit
serves every price, so moving it moves the operating point along the
cost-accuracy curve without retraining.

## Routing inside an agent loop

For step-level decisions the engine, the tokenizer and the block sit together and
only the decision crosses the network:

```bash
python -m inference.decision_service --model <path> --head weights/head-27b-mmlu-pro.safetensors
```

`/fast` returns the action distribution with the value of thinking and the
confidence beside it; `/slow` reasons first and reads the same candidates;
`/solve` lets the slow mode write an action where the environment can take one.
`inference.client.DecisionClient` is standard library only, so an environment
with its own dependencies can talk to the service without carrying torch.

## What the block reads

The hidden state at the readout position, and eight statistics of the fast
answer. That is the whole input. Nothing is parsed out of generated text, and
the decision costs no forward pass of its own, which is what makes routing free.

The value of thinking is a log ratio, `log P(slow right) / P(fast right)`, not a
difference: the reward is a discount on tokens rather than a subtraction, and a
difference shrinks exactly where thinking is worth most.

## Reproducing

`docs/reproduce.md` covers the collections and the fits. The blocks in `weights/`
are the ones the paper reports; `weights/README.md` records what they were fitted
on and how they score against the baselines, including the two places where a
simpler rule scores higher.

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

Apache 2.0, see `LICENSE`. Third-party notices are in `THIRD_PARTY_NOTICES.md`.
