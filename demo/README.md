# The CUA demo

One browser task, three policies, one clock.

| Policy | What it does | What it shows |
| --- | --- | --- |
| `fast` | never thinks, reads the action off the page's elements in one forward pass | the floor: fast, and often wrong |
| `slow` | thinks at every step | the ceiling, and what it costs |
| `routed` | thinks where the head says it is worth it | the claim |

All three run the same task from the same seed, and every step is timed, so the
difference is on the clock rather than in a table.

## The two halves

The browser has to run where someone can watch it and the backbone has to run
where the GPUs are. The engine and the head go in one job on the cluster, and
the demo calls it the way it would call any other model API:

    POST /fast  {state, actions}          -> probabilities, gain, confidence
    POST /slow  {state, actions, budget}  -> probabilities, generated tokens

The head reads a hidden state, so it has to be inside that job. It cannot sit
beside the browser: SGLang's HTTP server will return hidden states, but for
every position of the prompt, and on the 27B backbone a 900-token state is
48 MB of JSON. With the server and the caller on the same node that request
costs 14.3s against 0.75s without them, all of it spent encoding. In one
process the same forward pass costs about 0.7s, because the hidden states stay
tensors and only two numbers ever leave.

```bash
# on the cluster, holding a GPU
scripts/launch.sh browsergym scripts/serve_decision.sbatch decision-service-27b

# from wherever the browser is
ssh -p <port> -N -L 30001:<node>:30001 <user>@<cluster>
uv run python demo/cua_demo.py --decision-url http://localhost:30001 \
  --task browsergym/miniwob.click-checkboxes-large --output out/run1
```

`--model`/`--head` instead of `--decision-url` starts the engine in this
process, for a machine that has the GPU itself.

`--head` is a parameter. The head shipped with the paper is trained on MMLU-Pro;
swap in one trained on browser traces when there is one, and nothing else here
changes.

## Why lambda is calibrated rather than carried over

A head's predicted gain is calibrated to the domain it was trained on. Carried
into a different one it escalates almost everywhere or almost nowhere: on
ALFWorld the MMLU-Pro head escalated on 46 steps out of 50 at lambda = 0, which
is always-thinking wearing a router's clothes and shows nothing on a clock.

So the demo sets lambda the way the paper prescribes when a deployment names a
rate rather than a price: run the fast mode over a sample of this task's steps,
take the quantile of the gain that gives the escalation rate asked for. It needs
no labels, because the gain is produced by the fast mode at every step whether or
not the step escalates.

`--escalation-rate 0.3` asks for thinking on three steps in ten.

## What it writes

`trajectory.json` carries every step of every policy: the action, how many
candidates the page offered, whether the step escalated, the predicted gain, the
tokens generated, and four clocks: the forward pass, the thinking, the total so
far, and what the serving side measured for the same two calls. The last is
there so the record separates the model's time from the network's without
anyone having to take the total on trust. `frames/` holds a screenshot per step,
and the file is rewritten after each policy, so a run that is interrupted still
leaves what it had.

## The one constraint to watch

The fast mode reads its action off an enumerated candidate set, one single-token
code per candidate, and the tokenizer yields 52 of them. A page offering more
interactive elements than that is truncated to the first 52, which changes the
task. `--candidate-limit` makes the truncation explicit; a page that needs more
needs a different encoding, not a larger limit.
