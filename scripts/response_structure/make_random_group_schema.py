#!/usr/bin/env python
"""Create a preregistered random-dimension ActionSchema control."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from acwm.action_latent.response import ActionGroup, ActionSchema


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = json.loads(args.schema.read_text())
    sizes = [len(group["indices"]) for group in source["groups"]]
    dims = list(range(source["action_dim"]))
    rng = random.Random(args.seed)
    rng.shuffle(dims)

    groups = []
    stats = {}
    source_stats = source.get("group_statistics", {})
    offset = 0
    for index, size in enumerate(sizes):
        indices = tuple(sorted(dims[offset:offset + size]))
        offset += size
        name = f"random_seed_{args.seed}_group_{index}"
        groups.append(
            ActionGroup(
                name=name,
                indices=indices,
                semantics="random_dimension_control",
                provenance=f"Random dimension grouping control generated from {args.schema} with seed={args.seed}.",
            )
        )
        stats[name] = {
            key: [source_stats[f"dim_{dim}"][key][0] for dim in indices]
            for key in ("mean", "std", "q05", "q95")
        }

    schema = ActionSchema(
        action_dim=source["action_dim"],
        groups=groups,
        representation=source.get("representation", "unknown"),
        dimension_names=source.get("dimension_names"),
        provenance=f"Random grouping control derived from {args.schema}; seed={args.seed}.",
        group_statistics=stats,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(schema.to_dict(), ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "seed": args.seed, "group_count": len(groups)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
