"""Does the gathered readout return what the causal LM's logits did?

`_forward_readout` stopped materialising `batch x width x vocab` logits and
applies `lm_head` to the gathered rows instead. That is the same number only if
the head is the last thing the forward does and the gather picks the same
position, and **neither failure is visible downstream**: a wrong position returns
a pad token's logits, which still score something, and a dropped post-head
transform moves the candidate softmax -- every `distribution_stats` feature --
while leaving the answer argmax, and so every accuracy in the paper, untouched.

Usage, with the real collection's own arguments after the `--`:

    python scripts/verify_readout_equivalence.py report.json -- \
        --model .../qwen38-27b --layers=-1 --budget 4096 ... --output /dev/null

It patches the one seam and then runs `hde-collect` itself, rather than
reimplementing the collection loop: the loop decides batching, padding and which
position each row is read at, and a second copy of it here would be testing the
copy. Each patched call runs the real readout *and* the causal LM on the same
inputs in the same process, so node, decode and batch-composition differences --
each already mistaken for a real effect on this project -- are all held fixed and
only the change under test varies.

Run it at batch 1. The old path at batch 5 over a 5323-token sequence is the
12.3 GiB allocation this change exists to remove, so asking for both at once
would OOM in the comparison rather than in the thing compared.
"""
import json
import sys

import torch


def install(report):
    from hde.backends.backend import TransformersScorer

    original = TransformersScorer._forward_readout

    def wrapped(self, positions, **kwargs):
        base, readout = original(self, positions, **kwargs)
        # The path this replaces, on the same inputs. `output_hidden_states` is
        # passed exactly as the old call did, so the hidden tuple is comparable.
        whole = self.model(**kwargs, output_hidden_states=bool(self.feature_layers))
        rows = torch.arange(len(positions), device=readout.device)
        index = torch.as_tensor(list(positions), device=readout.device)
        reference = whole.logits[rows, index]

        entry = {
            "rows": len(positions),
            "width": int(kwargs["input_ids"].shape[1]),
            "max_abs_delta": (readout.float() - reference.float()).abs().max().item(),
            "same_argmax": bool(torch.equal(readout.argmax(-1), reference.argmax(-1))),
            "hidden_identical": None,
        }
        if base.hidden_states is not None and whole.hidden_states is not None:
            entry["hidden_identical"] = all(
                torch.equal(a, b) for a, b in
                zip(base.hidden_states, whole.hidden_states, strict=True))
        report.append(entry)
        del whole, reference
        return base, readout

    TransformersScorer._forward_readout = wrapped


def verdict(report):
    deltas = [r["max_abs_delta"] for r in report]
    argmax_failures = [r for r in report if not r["same_argmax"]]
    hidden_failures = [r for r in report if r["hidden_identical"] is False]
    return {
        "forwards_compared": len(report),
        # Reported rather than gated on: the gather changes the GEMM's shape, so
        # a non-zero difference at float16 is expected and bit-identity here
        # would reject a correct fix. The legs that are gated are below.
        "max_abs_delta": max(deltas) if deltas else None,
        "mean_abs_delta": sum(deltas) / len(deltas) if deltas else None,
        "argmax_identical": not argmax_failures,
        "argmax_failures": len(argmax_failures),
        # The hidden states are the head's actual input and pass through
        # untouched either way, so this leg *is* gated on bit-identity.
        "hidden_bit_identical": not hidden_failures,
        "hidden_forwards_checked": sum(1 for r in report if r["hidden_identical"] is True),
        "hidden_failures": len(hidden_failures),
        "widest_sequence": max((r["width"] for r in report), default=None),
    }


def main():
    if "--" not in sys.argv[2:]:
        print(__doc__, file=sys.stderr)
        return 2
    output = sys.argv[1]
    cut = sys.argv.index("--")
    forwarded = sys.argv[cut + 1:]

    report = []
    install(report)

    from hde.cli import collect

    sys.argv = ["hde-collect", *forwarded]
    status = 0
    try:
        collect()
    except SystemExit as stop:
        status = stop.code or 0

    summary = verdict(report)
    print(json.dumps(summary, indent=2))
    with open(output, "w") as handle:
        json.dump({"verdict": summary, "forwards": report}, handle, indent=2)

    if status:
        print(f"FAIL: the collection itself exited {status}", file=sys.stderr)
        return status
    if not report:
        # An empty comparison is a passing assertion over nothing, which is how
        # a reachability test on this project stayed green while the thing it
        # checked was gone.
        print("FAIL: nothing was compared, so this is not a passing check", file=sys.stderr)
        return 2
    if not summary["argmax_identical"] or not summary["hidden_bit_identical"]:
        print("FAIL: the readout does not reproduce the causal LM", file=sys.stderr)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
