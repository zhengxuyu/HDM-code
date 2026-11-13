"""Print what SGLang actually returns for token_ids_logprob.

The fast mode reads its answer from the logprobs of a chosen set of token ids at
one position. My first attempt indexed that structure from the shape I assumed it
had and crashed on a None. This prints it instead.
"""
import argparse

from litjev.run.streaming import write_json_atomic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import sglang as sgl
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    codes = [tokenizer.encode(f" {c}", add_special_tokens=False)[0] for c in "ABC"]
    print(f"candidate token ids for ' A', ' B', ' C': {codes}", flush=True)

    engine = sgl.Engine(model_path=args.model, mem_fraction_static=0.75)
    output = engine.generate(
        "Question: which is largest?\n A one\n B two\n C three\nAnswer:",
        sampling_params={"max_new_tokens": 1, "temperature": 0.0},
        return_logprob=True, token_ids_logprob=codes)
    meta = output["meta_info"]

    print("\nmeta_info keys:", sorted(meta))
    for key in sorted(meta):
        if "logprob" not in key:
            continue
        value = meta[key]
        print(f"\n--- {key} ---")
        print(f"type {type(value).__name__}", end="")
        if isinstance(value, list):
            print(f", length {len(value)}")
            nones = sum(item is None for item in value)
            print(f"  {nones} of {len(value)} entries are None")
            for index, item in enumerate(value):
                if item is None:
                    continue
                print(f"  [{index}] {type(item).__name__} "
                      f"{len(item) if hasattr(item, '__len__') else ''}: {str(item)[:160]}")
        else:
            print(f": {str(value)[:200]}")

    write_json_atomic(args.output,
                      {key: str(meta[key])[:2000] for key in meta if "logprob" in key})


if __name__ == "__main__":
    main()
