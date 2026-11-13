"""Can SGLang load this backbone, and what shape are its hidden states?

Everything downstream rests on two answers, and both are cheaper to get by asking
the engine than by reading its source: whether it recognises the architecture at
all, and whether `return_hidden_states` gives one layer or all of them. The head
reads layer 48 of 64, so the difference decides whether this path serves the head
we have or only one trained on the last layer.
"""
import argparse
import json

import numpy as np

from hde.run.streaming import write_json_atomic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import sglang as sgl

    engine = sgl.Engine(model_path=args.model, enable_return_hidden_states=True,
                        mem_fraction_static=0.75)

    # Two prompts of deliberately different length. A middle dimension that tracks
    # the prompt length is sequence positions; one that stays fixed is layers. The
    # first version of this probe read ndim == 3 as "per-layer available" and was
    # wrong, so the shape is identified by what it varies with, not by its rank.
    prompts = {
        "short": "A two\nB three\nAnswer:",
        "long": ("Question: consider a prompt written to be clearly longer than the other "
                 "one, so that a dimension tracking sequence length cannot be mistaken for "
                 "one tracking depth.\nA yes\nB no\nC maybe\nAnswer:"),
    }
    seen = {}
    for name, prompt in prompts.items():
        output = engine.generate(prompt, sampling_params={"max_new_tokens": 1, "temperature": 0.0},
                                 return_hidden_states=True)
        meta = output["meta_info"]
        states = np.asarray(meta["hidden_states"], dtype=np.float32)
        seen[name] = {"shape": list(states.shape), "prompt_tokens": meta.get("prompt_tokens")}
        print(f"{name:6} prompt_tokens {meta.get('prompt_tokens'):4d}  "
              f"hidden_states {states.shape}", flush=True)

    short, long = seen["short"], seen["long"]
    varies = [i for i in range(min(len(short["shape"]), len(long["shape"])))
              if short["shape"][i] != long["shape"][i]]
    tracks_length = any(short["shape"][i] == short["prompt_tokens"] for i in varies)
    layer_dim = [i for i in range(len(short["shape"]))
                 if i not in varies and short["shape"][i] not in (1, 5120)]
    report = {
        "loaded": True,
        "shapes": seen,
        "dimensions_that_vary_with_prompt_length": varies,
        "a_dimension_tracks_sequence_length": tracks_length,
        "candidate_layer_dimension": layer_dim,
        "per_layer_available": bool(layer_dim),
        "meta_keys": sorted(meta.keys()),
    }
    print("\n" + json.dumps(report, indent=1), flush=True)
    if layer_dim:
        print(f"a fixed dimension of size {short['shape'][layer_dim[0]]} looks like depth; "
              "layer 48 may be reachable")
    else:
        print("no dimension of fixed non-trivial size: this engine returns one layer only.\n"
              "The layer-48 head cannot be served here; a head trained on layer -1 can.")
    write_json_atomic(args.output, report)


if __name__ == "__main__":
    main()
