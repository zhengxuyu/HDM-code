# Deploying the playground

The server carries three pages and the API behind them. Nothing is built or
bundled: the pages are static files served by the same process that holds the
model, so deploying the playground is deploying the service.

| page | what it is for |
|---|---|
| `/` | one typed request, answered by the fast mode alone. The distribution and the confidence as the model reads them |
| `/routing` | the same request at several prices, side by side: where the block escalates and what it costs |
| `/health` | whether weights are loaded |

## Install

Python 3.11 or newer.

```bash
git clone https://github.com/zhengxuyu/HDM-code
cd HDM-code
uv sync                 # or: pip install -e .
```

The blocks are not in the repository; they are on the Hub. Fetch the one that
matches the backbone you mean to serve:

```bash
huggingface-cli download fredreick/HDM-blocks \
    head-4b-mmlu-pro.safetensors --local-dir weights/
```

`uv sync` resolves the pinned lockfile. The GPU wheels follow your CUDA; see
`[tool.uv.sources]` in `pyproject.toml` if your platform needs a different index.

## The smallest run

The transformers backend loads a checkpoint in process and needs no server. Weights
download on the first call, so the first request is slow and the rest are not.

```bash
hde-serve --model Qwen/Qwen3.5-4B --port 8000
```

Open <http://127.0.0.1:8000>. This serves the fast mode only: with no block
attached, `/v1/systemtwo` has nothing to route on and `/routing` will say so.

## With routing

Attach a block and the routing page starts deciding. The block must match the
backbone it was fitted on; the server refuses a pairing it was not trained for
rather than reading noise.

```bash
hde-serve --model Qwen/Qwen3.5-4B \
             --decision-head weights/head-4b-mmlu-pro.safetensors \
             --readout coded --port 8000
```

`--readout` has to be the one the block was fitted under. `coded` is the default and
is what both shipped blocks use.

## On a GPU, through SGLang

SGLang is far faster at the slow mode. The engine is built in process, which is why
`--preload` matters: it builds the engine on the main thread at startup rather than
inside the first request.

```bash
hde-serve --model /path/to/Qwen3.8-27B \
             --decision-head weights/head-27b-mmlu-pro.safetensors \
             --backend sglang --readout coded --preload \
             --host 0.0.0.0 --port 8000
```

Verify the engine returns what the block was trained on before serving anything that
matters:

```bash
python -m inference.verify --model /path/to/Qwen3.8-27B
```

That check exists because a wrong layer, a wrong position or a batched request
answered with another request's state fails silently: the block reads noise, the
page still renders, and the numbers mean nothing.

## Reaching it from somewhere else

**The playground has no authentication.** The server binds `127.0.0.1` for that
reason, and `--host 0.0.0.0` opens it to everything that can route to the machine.
On a shared or cluster node, forward the port instead of opening it:

```bash
ssh -N -L 8000:<node>:8000 <user>@<host>
```

Then open <http://127.0.0.1:8000> locally. If you do bind a public interface, put a
proxy with authentication in front of it.

## Behind an agent, not a browser

The pages are for looking. For step-level decisions inside a loop, run the decision
service instead: the engine, the tokenizer and the block sit together and only the
decision crosses the network.

```bash
python -m inference.decision_service \
       --model Qwen/Qwen3.5-4B \
       --head weights/head-4b-mmlu-pro.safetensors \
       --host 0.0.0.0 --port 30001
```

`GET /health` names the block it is serving, down to a digest of the weights. Record
that digest with any numbers you report: two blocks with identical structural fields
have served different decisions.

## Containers

`demo/Dockerfile` builds the browser half of the CUA demo, which drives a browser and
calls a decision service over HTTP. It deliberately carries no engine: shipping hidden
states over the wire costs 14.3 s a step against 0.7 s in process. Build the model
half wherever the GPUs are and point the container at it with `--decision-url`.
