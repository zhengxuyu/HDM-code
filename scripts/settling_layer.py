"""When does the fast answer stop changing, and does that say whether thinking helps?

The paper explains thinking by serial depth: one pass is $L$ layers deep whatever
the question, and writing the reasoning out buys more steps. Nothing in the paper
tests that. This measures a proxy for it.

Read the answer not only at the end but at every layer: apply the model's final
norm and its output projection to the readout state of layer $\\ell$, keep the
candidate codes, and take the argmax. The \\emph{settling layer} is the last layer
at which that argmax changes. A question whose answer is determined early settles
early; one the pass is still working on when it runs out of layers settles late.

The prediction is that thinking helps more on late-settling questions. That is a
statement about the recorded outcomes, not about our head, so it tests the
explanation rather than the model we trained. Whether the head has picked the
signal up is the second question, and the analysis script asks both.

The fast mode only: the slow outcomes are already recorded, and re-running the
slow mode would cost what the original collection cost.

    hde-collect --layers=... is the reference for the sampling; this repeats it.
"""
import argparse
import json

import numpy as np
import torch


def final_norm(model):
    """The norm applied before the output projection, whatever this model calls it.

    Found by scanning the module tree rather than by walking attribute names: the
    wrapper around the decoder differs between checkpoint families, and guessing
    the path is how the first run of this died on a model that had the module all
    along. Per-layer norms live under `layers.`, so they are excluded by path.
    """
    wanted = {"norm", "final_layernorm", "final_norm", "ln_f"}
    found = [(name, module) for name, module in model.named_modules()
             if name.split(".")[-1] in wanted and ".layers." not in f".{name}."]
    if not found:
        raise RuntimeError(
            "Cannot find the final norm; a logit lens needs it. Modules seen: "
            + ", ".join(name for name, _ in list(model.named_modules())[:40]))
    name, module = min(found, key=lambda pair: pair[0].count("."))
    print(f"  final norm: {name}", flush=True)
    return module


def main():
    from hde.backends.backend import ModelSettings, TransformersScorer
    from hde.data.datasets import LOADERS, load
    from hde.data.mmlu import ten_question_batches

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=sorted(LOADERS), default="mmlu-pro")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--split", default=None,
        help="what this dataset calls it, and left to the dataset when unset: test "
             "or validation for the multiple-choice sets, public or a tier for "
             "jevbench, and one fixed name each for agieval and gpqa-diamond, which "
             "refuse any other")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=1500)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--output", default="settling-4b.npz")
    arguments = parser.parse_args()

    loaded = load(arguments.dataset, split=arguments.split)
    questions, labels = loaded.questions, loaded.labels
    questions = questions[arguments.offset :: arguments.stride][: arguments.limit]
    if not questions:
        parser.error("No questions selected")

    # Every layer, including the embedding output, so the lens sees the whole stack.
    probe = TransformersScorer.load(ModelSettings(
        arguments.model, arguments.revision, arguments.device_map, arguments.dtype,
        feature_layers=(-1,)))
    depth = getattr(probe.model.config, "text_config", probe.model.config).num_hidden_layers
    del probe
    torch.cuda.empty_cache()

    layers = tuple(range(depth + 1))
    scorer = TransformersScorer.load(ModelSettings(
        arguments.model, arguments.revision, arguments.device_map, arguments.dtype,
        feature_layers=layers))
    norm = final_norm(scorer.model)
    projection = scorer.model.get_output_embeddings()
    device = next(projection.parameters()).device
    dtype = next(projection.parameters()).dtype

    identifiers, per_layer, counts = [], [], []
    for index, (request, valid_ids) in enumerate(ten_question_batches(questions)):
        schema = request.to_schema()
        valid = [name for name in valid_ids if name in labels]
        if not valid:
            continue
        rows = {row.name: row for row in scorer.score(request.state, schema)}
        for name in valid:
            row = rows[name]
            candidates = row.provenance["candidate_token_ids"]
            hidden = torch.as_tensor(np.asarray(row.hidden), device=device, dtype=dtype)
            with torch.no_grad():
                # The lens: every layer read through the same head the last layer uses.
                logits = projection(norm(hidden))[:, candidates].float().cpu().numpy()
            padded = np.full((len(layers), 16), -np.inf, dtype=np.float32)
            padded[:, : len(candidates)] = logits
            identifiers.append(name)
            per_layer.append(padded)
            counts.append(len(candidates))
        if index % 10 == 0:
            print(f"  batch {index}, {len(identifiers)} questions", flush=True)

    np.savez_compressed(
        arguments.output,
        question_id=np.array(identifiers),
        layer_logits=np.stack(per_layer),
        candidate_count=np.array(counts),
        layers=np.array(layers),
        metadata=np.array(json.dumps({
            "model_id": arguments.model, "revision": arguments.revision,
            "dataset": arguments.dataset, "split": arguments.split,
            "offset": arguments.offset, "stride": arguments.stride,
            "limit": arguments.limit, "depth": depth,
            "note": "layer_logits is [question, layer, candidate], padded with -inf",
        })))
    print(f"wrote {arguments.output}: {len(identifiers)} questions, {len(layers)} layers")


if __name__ == "__main__":
    main()
