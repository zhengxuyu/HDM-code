"""Vary how many dependent steps a question needs, and hold everything else still.

The settling-layer probe asked whether the value of thinking tracks a proxy for
running out of depth, and it did not. The proxy may have been wrong; the way to
find out without a proxy is to build the variable instead of measuring it.

Each request carries a table of arrows over $n$ symbols, drawn as a permutation
so every walk stays inside the table and no symbol is a dead end. A question asks
where a walk of $k$ arrows ends. Answering needs $k$ lookups, and the $j$-th
cannot start before the $(j-1)$-th has finished, so $k$ is the number of steps
that must happen one after another. Everything else is held: the table is the
same for every question in a request, so the prompt length does not move with
$k$; the symbols are the same; and nothing has to be known that is not written
down, so no question is harder because the model has read less about it.

What the depth story predicts, if it is right:

    the fast mode's accuracy falls as $k$ grows,
    the slow mode's holds up longer, since each token it writes buys a pass,
    so the questions thinking rescues concentrate at larger $k$,
    and a head that reads a shortfall in depth gives them a larger value.

The last line is the one that tests the head. It is applied here out of the
domain it was trained on, so its gains are not calibrated to this task; what is
read is whether they rise with $k$, not what they equal.

    uv run python scripts/depth_probe.py --head head.safetensors --out depth.npz
"""
import argparse
import random

SYMBOLS = ["Ada", "Ben", "Cleo", "Dov", "Eli", "Fay", "Gus", "Hana", "Ivo", "Jun",
           "Kit", "Lev", "Mira", "Nils", "Ora", "Pax", "Quinn", "Rhea", "Sol", "Tove"]


def make_table(symbols, generator):
    """One cycle through every symbol, so a walk of $k$ steps really takes $k$ steps.

    A permutation drawn at random breaks into cycles, and a walk on a cycle of
    length $c$ returns to where it started every $c$ steps: asking for $k$ steps
    then asks for $k \bmod c$, and the depth the question is supposed to fix is
    not the depth it has. Threading every symbol onto a single cycle makes the
    landing point distinct for every $k$ below the number of symbols, which is
    why the depths must stay under it.
    """
    order = symbols[:]
    generator.shuffle(order)
    return {order[i]: order[(i + 1) % len(order)] for i in range(len(order))}


def walk(table, start, steps):
    place = start
    for _ in range(steps):
        place = table[place]
    return place


def build_request(depths, symbols, generator, index):
    """One table, one question per depth, so a batch holds the whole depth axis."""
    from litjev.data.schema import Choice, SystemOneRequest

    table = make_table(symbols, generator)
    lines = "\n".join(f"{source} -> {target}" for source, target in table.items())
    state = ("Each line gives one arrow. Follow arrows exactly as written.\n\n" + lines)
    questions, answers, meta = {}, {}, {}
    for depth in depths:
        start = generator.choice(symbols)
        name = f"t{index}d{depth}"
        questions[name] = Choice(
            instructions=(f"Start at {start} and follow {depth} "
                          f"{'arrow' if depth == 1 else 'arrows'}. Where do you end?"),
            criteria={symbol: symbol for symbol in symbols})
        answers[name] = walk(table, start, depth)
        meta[name] = depth
    return SystemOneRequest(model="litjev", state=state, questions=questions), answers, meta


def main():
    from litjev.backends.backend import ModelSettings, TransformersScorer
    from litjev.data.collect import collect_batch, save_records

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--layers", default="-1,16,24")
    parser.add_argument("--depths", default="1,2,3,4,6,8,10,12,14")
    parser.add_argument("--tables", type=int, default=100, help="one request each")
    parser.add_argument("--symbols", type=int, default=16)
    parser.add_argument("--budget", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="depth-4b.npz")
    arguments = parser.parse_args()

    from litjev._legacy.heads import parse_feature_layers

    depths = [int(value) for value in arguments.depths.split(",")]
    symbols = SYMBOLS[: arguments.symbols]
    if len(symbols) < 3:
        parser.error("need at least three symbols")
    if max(depths) >= len(symbols):
        parser.error(f"a walk of {max(depths)} steps wraps on {len(symbols)} symbols, "
                     "so its depth would not be what it says")
    generator = random.Random(arguments.seed)

    layers = parse_feature_layers(arguments.layers)
    scorer = TransformersScorer.load(ModelSettings(
        arguments.model, arguments.revision, arguments.device_map, arguments.dtype,
        feature_layers=layers))

    records, categories = [], {}
    for index in range(arguments.tables):
        request, answers, meta = build_request(depths, symbols, generator, index)
        # The depth rides in the category field, which the record already carries,
        # so the existing analysis groups by it with nothing added.
        categories.update({name: f"depth_{depth}" for name, depth in meta.items()})
        records.extend(collect_batch(scorer, request, list(answers), answers,
                                     arguments.budget, categories))
        print(f"  table {index + 1}/{arguments.tables}, {len(records)} records", flush=True)

    save_records(arguments.output, records, {
        "model_id": arguments.model, "revision": arguments.revision,
        "hidden_size": scorer.hidden_size, "feature_layers": list(layers),
        "budget": arguments.budget, "depths": depths, "symbols": len(symbols),
        "tables": arguments.tables, "seed": arguments.seed,
        "task": "iterated_map_walk",
    })
    print(f"wrote {arguments.output}: {len(records)} records")


if __name__ == "__main__":
    main()
