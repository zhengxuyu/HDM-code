"""Fail a job in seconds if its environment is missing something it needs later.

The ALFWorld environment is separate from the project venv, and an import that
only happens on the three hundredth episode costs five hours to discover. This
has been wrong twice by being too narrow, so it names everything rather than a
representative.
"""
import importlib
import sys

REQUIRED = (
    "causal_conv1d",              # the fused kernel; without it timings are meaningless
    "alfworld.agents.environment",
    "litjev._legacy.heads",
    "inference",                  # the SGLang backend
    "slime_htm.envs.alfworld",    # the environment adapter
    "torch",
    "transformers",
    "yaml",
)

missing = []
for name in REQUIRED:
    try:
        importlib.import_module(name)
    except Exception as error:  # noqa: BLE001  reporting any failure is the point
        missing.append(f"{name}: {type(error).__name__}: {error}")

if missing:
    print("the ALFWorld environment is incomplete:")
    for line in missing:
        print(f"  {line}")
    print("see experiments/alfworld/README.md for what it needs")
    sys.exit(1)
print("environment: " + ", ".join(REQUIRED) + " all present")
