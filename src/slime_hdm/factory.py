"""Build the pieces a rollout needs from `args`, without knowing the task.

An agentic step is already a typed request: one state, one `choice` question, and
the admissible actions as its candidates. So the step scorer is a thin adapter
over the same `score` and `think` a benchmark uses, and the two backends, SGLang
and transformers, arrive here unchanged.

The environment is loaded by import path. Naming it in the config rather than
importing it here is what keeps this package from becoming one benchmark's.
"""

import importlib

import numpy as np

from hde.data.schema import Choice, DecisionSchema

ACTION = "action"
INSTRUCTIONS = "Choose the action that makes most progress on the task."


class StepScorer:
    """`fast` and `slow` for one agentic step, over any LogitProvider.

    The candidate set changes at every step, so the schema is rebuilt each time;
    the codes are assigned by the provider, which is where the single-token
    requirement is checked.
    """

    def __init__(self, provider, instructions=INSTRUCTIONS):
        self.provider = provider
        self.instructions = instructions

    def _schema(self, actions):
        if not actions:
            raise ValueError("a step needs at least one admissible action")
        criteria = {str(index): action for index, action in enumerate(actions)}
        return DecisionSchema({ACTION: Choice(instructions=self.instructions, criteria=criteria)})

    @staticmethod
    def _distribution(field):
        logits = np.asarray(field.logits, dtype=np.float64)
        shifted = logits - logits.max()
        weights = np.exp(shifted)
        return weights / weights.sum()

    def fast(self, context, actions):
        """One forward pass: the action distribution, and the hidden state beside it."""
        (field,) = self.provider.score(context, self._schema(actions))
        hidden = None if field.hidden is None else np.asarray(field.hidden).reshape(-1)
        return self._distribution(field), hidden

    def slow(self, context, actions, budget):
        """Reason first, then read the same candidates.

        Returns the distribution, what the reasoning cost, and the reasoning
        itself, which a caller may show and no caller may parse: the answer is
        read from the candidate codes either way.
        """
        (field,) = self.provider.think(context, self._schema(actions), [ACTION], budget)
        return self._distribution(field), int(field.generated_tokens), field.reasoning

    def solve(self, context, instructions, budget):
        """Reason, then **write** the action instead of picking one.

        `slow` reads its answer off the same enumerated codes the fast mode used,
        so escalating bought a longer thought and no larger action space: a step
        whose right action is text was unwinnable at every price, and a gate run
        that way reports a restriction of ours as a property of the venue.

        This is the other shape, and it is the one `hde.construct` already
        relies on for questions whose answers were never enumerated -- the
        backends have had `solve` all along. The caller parses what comes back,
        strictly, and records a refusal rather than substituting a click.

        Returns `(text, generated_tokens, truncated)`.
        """
        return self.provider.solve(context, instructions, budget)


class ClientScorer:
    """`StepScorer`'s contract, served over HTTP instead of in this process.

    This exists because the engine **cannot** be in the browser's process.
    Playwright's sync API runs an event loop on the calling thread and SGLang's
    in-process `Engine` drives its own with `run_until_complete`, so the first
    scoring call of the first episode dies with "Cannot run the event loop while
    another loop is running". exps hit that in 28840 at 2m24s. The gate never met
    it because the gate has always reached the model over HTTP with `urllib`,
    which needs no loop -- the project had the right architecture for a different
    stated reason (the cost of shipping hidden states) and this conflict was
    written down nowhere.

    `fast` asks for the hidden state because CDPO trains on it. That is not the
    48 MB the decision service exists to avoid: that figure is sglang's own server
    encoding a state for every prompt position, while this ships one vector of
    about 27 KB per step.
    """

    def __init__(self, client, instructions=None):
        self.client = client
        self.instructions = instructions

    def fast(self, context, actions):
        answer = self.client.fast(context, list(actions), want_hidden=True)
        hidden = answer.get("hidden_state")
        if hidden is None:
            raise RuntimeError(
                "the decision service returned no hidden state. CDPO trains on it, "
                "so a service that cannot supply it cannot collect for training -- "
                "check that it is running a build with /fast want_hidden")
        return (np.asarray(answer["probabilities"], dtype=np.float64),
                np.asarray(hidden, dtype=np.float32))

    def slow(self, context, actions, budget):
        answer = self.client.slow(context, list(actions), budget)
        return (np.asarray(answer["probabilities"], dtype=np.float64),
                int(answer.get("generated_tokens", 0)),
                answer.get("reasoning"))

    def solve(self, context, instructions, budget):
        answer = self.client.solve(context, instructions, budget)
        return (answer.get("text", ""), int(answer.get("generated_tokens", 0)),
                bool(answer.get("truncated", False)))


def load_environment(args, evaluation=False):
    """Import the environment named on `args` and construct it.

    `hdm_environment_path` is `module:factory`, and the factory takes the args and
    a flag saying whether this is evaluation, so a benchmark can hand back its
    held-out split without a second setting.
    """
    path = getattr(args, "hdm_environment_path", None)
    if not path:
        raise ValueError("set --hdm-environment-path to module:factory")
    module_name, _, factory_name = path.partition(":")
    if not factory_name:
        raise ValueError(f"--hdm-environment-path must be module:factory, got {path!r}")
    factory = getattr(importlib.import_module(module_name), factory_name)
    return factory(args, evaluation=evaluation)


def build_scorer(args):
    """The backend the two modes run on. SGLang unless transformers is asked for."""
    head_layers = _head_layers(args)
    backend = getattr(args, "hdm_backend", "sglang")
    if backend == "sglang":
        from inference import SGLangScorer, SGLangSettings

        provider = SGLangScorer.load(SGLangSettings(
            model_id=args.hf_checkpoint,
            feature_layers=head_layers,
            batch_questions=False,
            mem_fraction_static=getattr(args, "hdm_mem_fraction_static", 0.8),
        ))
    elif backend == "transformers":
        from hde.backends.backend import ModelSettings, TransformersScorer

        provider = TransformersScorer.load(ModelSettings(
            args.hf_checkpoint, "main", getattr(args, "hdm_device_map", "cuda:0"),
            "bfloat16", feature_layers=head_layers))
    else:
        raise ValueError(f"unknown --hdm-backend {backend!r}")
    return StepScorer(provider, instructions=getattr(args, "hdm_instructions", INSTRUCTIONS))


def _head_layers(args):
    from hde.core.block import DecisionHead

    path = getattr(args, "hdm_decision_head", None)
    if not path:
        return ()
    return DecisionHead.load(path).metadata.feature_layers


def build_head(args):
    """The head that decides, or None to explore at random instead.

    A head that has not been trained cannot explore, so the first rollout of a run
    has none and `hdm_escalate_prob` decides. Later rollouts load the head the
    previous one produced.
    """
    from hde.core.block import DecisionHead

    path = getattr(args, "hdm_decision_head", None)
    return DecisionHead.load(path) if path else None
