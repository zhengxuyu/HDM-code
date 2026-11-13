"""One way to obtain a proposer, whichever backend is serving.

The two backends propose candidates differently and cannot propose them the same
way: SGLang has no beam search, and transformers has no `n`-sample call that
returns what the server returns. The difference does not reach the decision head,
which reads the readout's distribution over the constructed codes either way, so
what a caller needs is a proposer and not a backend.

SGLang is the default because everything else in the paper is served that way and
the two backends do not agree closely enough to share a table: their readout
states point the same way but differ by a few percent in magnitude, which moves a
predicted value of thinking by more than a tenth of the spacing of the price grid
we sweep. A run on transformers therefore has to be reported as its own group.
"""


def load_proposer(model, backend="sglang", revision="main", dtype="bfloat16",
                  device_map="auto", feature_layers=()):
    """A scorer that can both propose candidates and read one out.

    Returns the scorer itself: both backends expose `propose`, `score` and
    `think`, so nothing downstream has to know which one it received.
    """
    if backend == "sglang":
        from inference import SGLangScorer, SGLangSettings

        # dtype and device_map belong to the transformers loader; SGLang decides
        # both from the checkpoint and the node. Say so rather than dropping a
        # setting the caller thought they had made -- and name only the setting
        # actually at fault. 28281 passed `device_map='cuda:0'` with an allowed
        # dtype and was told that neither was taken, so the message read as two
        # things to fix and one of them was not wrong.
        refused = []
        if dtype not in (None, "bfloat16"):
            refused.append(f"dtype={dtype!r}")
        if device_map not in (None, "auto"):
            refused.append(f"device_map={device_map!r}")
        if refused:
            raise ValueError(
                f"the sglang backend does not take "
                f"{' or '.join(refused)}; it chooses "
                f"{'them' if len(refused) > 1 else 'that'} from the checkpoint and the "
                f"node. Drop the setting, or pass backend='transformers'")
        return SGLangScorer.load(SGLangSettings(
            model_id=model, revision=revision, feature_layers=tuple(feature_layers),
            batch_questions=False))
    if backend == "transformers":
        from hde.backends.backend import ModelSettings, TransformersScorer

        return TransformersScorer.load(ModelSettings(
            model, revision, device_map, dtype, feature_layers=tuple(feature_layers)))
    raise ValueError(f"unknown backend {backend!r}; use 'sglang' or 'transformers'")


def propose(scorer, state, instructions, width, max_new_tokens, temperature, seed=None,
            verified=True):
    """Propose candidates, whichever backend is behind the scorer.

    Both `propose` methods take the same arguments. The transformers one refuses
    a temperature, a seed and the constraint rather than ignoring them: a beam
    does not sample, and it decodes without a grammar, so accepting any of the
    three would suggest otherwise.
    """
    return scorer.propose(state, instructions, width=width,
                          max_new_tokens=max_new_tokens, temperature=temperature,
                          seed=seed, verified=verified)


def add_proposal_arguments(parser):
    """The options every proposal run takes, in one place rather than two.

    They were copied between the scripts and drifted the moment one of them
    gained a setting, which is the whole of the argument for putting them here.
    """
    parser.add_argument("--backend", choices=["sglang", "transformers"], default="sglang",
                        help="sglang is the standard here; transformers exists only for the "
                             "beam search sglang has no equivalent of, and its readout "
                             "differs enough that its numbers form their own group")
    parser.add_argument("--rank", choices=["likelihood", "frequency", "both"],
                        default="likelihood",
                        help="how proposals are ordered, which fixes which one the readout "
                             "is compared against. 'both' scores the two orderings from one "
                             "proposal pass, so the difference between them is not partly "
                             "the difference between two draws")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="sampling only; report it with whatever coverage it produced")
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--no-verified-proposals", dest="verified", action="store_false",
                        help="let the proposal pass write free text instead of holding it "
                             "to a constrained shape. The shape is what removes the parse "
                             "that four rounds of extraction faults lived in; this exists "
                             "to measure what it changes, and the record says which was used")
    from hde.candidates.construction_settings import DEFAULT_SHUFFLE

    parser.add_argument("--shuffle-candidates", dest="shuffle", default=DEFAULT_SHUFFLE,
                        help="the seed for the order constructed candidates are listed in, "
                             "combined with the question id. On by default: in rank order "
                             "the slot a biased readout favours is also one particular "
                             "rank, and the readout took the runner-up on all but one of "
                             "478 questions with no number able to say whether that was "
                             "about the answers or about the slot")
    parser.add_argument("--no-shuffle-candidates", dest="shuffle", action="store_const",
                        const=None,
                        help="list constructed candidates in rank order, for a comparison "
                             "that wants the confound on purpose")
    parser.add_argument("--seed", type=int, default=None,
                        help="sampling only: the engine's per-request sampling seed, "
                             "recorded with the run. Two runs of one configuration came "
                             "out at 45%% and 35%% coverage without it. A seed, not a "
                             "guarantee: SGLang's batching makes the arithmetic depend on "
                             "what else is in the batch, so an engine not started for "
                             "deterministic inference can still diverge")
    return parser


def check_proposal_arguments(parser, arguments):
    """Refuse combinations that would run and report something other than they say."""
    if arguments.backend == "transformers":
        if arguments.rank in ("frequency", "both"):
            parser.error("a beam returns each answer once, so ranking by frequency there "
                         "silently ranks by likelihood; use --backend sglang to sample")
        if arguments.seed is not None:
            parser.error("a beam does not sample, so there is no draw for --seed to fix; "
                         "it is already the same every time")
        if arguments.verified:
            parser.error("this backend decodes without a grammar, so it cannot be held to "
                         "the proposal shape; use --backend sglang, or pass "
                         "--no-verified-proposals and accept a parsed answer")
        arguments.temperature = None
    elif arguments.temperature is not None and arguments.temperature <= 0:
        parser.error("sampling at temperature zero returns one answer width times, which "
                     "is not a choice; the collapse is worth measuring just above it")



def looks_like_codes(proposals):
    """Whether these proposals are option letters rather than answers.

    `propose` once rendered the prompt that says "Return only the option code",
    so a beam over a maths question returned B, C and A. Every candidate set was
    two or three letters, coverage came out at 1.4% over 500 questions, and
    nothing in the run said anything was wrong until someone read the proposals.
    """
    texts = [text.strip() for text, _ in proposals if text.strip()]
    if not texts:
        return False
    return sum(len(text) <= 2 and text.isalpha() for text in texts) > len(texts) / 2


# The temperature the sampler is asked about. High enough that a sampler which is
# working cannot return one answer ten times over, and it is a question about the
# engine rather than about the run, so it does not have to be the run's own.
PROBE_TEMPERATURE = 1.0


def probe_sampler(scorer, state, prompts, width, max_new_tokens, seed, tries=10):
    """Ask the engine, once, whether a seed fixes its draw or replaces it.

    `propose` sends one request carrying `width` samples and one seed. If this
    SGLang expands `n` into copies that share the seed, a seeded run returns
    `width` identical proposals on every question. In a sweep that is not a
    failure: coverage lands near zero and the gate records an artefact of the
    seed as a finding about the model. The running watch catches it above
    temperature 0.5, where a total collapse cannot be the temperature's doing,
    but the sweep goes down to 0.1 -- and there the artefact and the real
    collapse the sweep is measuring are the same picture.

    So this asks the question directly, at a temperature of its own, before the
    run starts. Each question is proposed twice, without the seed and with it.
    One that varies unseeded and does not seeded is conclusive and nothing else
    is: a question can legitimately return one answer `width` times, which is
    why an unseeded collapse moves on to the next question rather than deciding.

    Running out of questions without an answer is itself an abort. The point of
    asking is that an unverified seeded run is silently wrong below temperature
    0.5, and letting one through because the probe could not decide would put it
    back in exactly that blind spot with a line in the record saying the sampler
    had been asked.

    Returns `(abort, facts)`. `facts` belongs in the run's record either way: a
    seeded number should say that the sampler was asked.
    """
    seeded, unseeded = [], []
    for prompt in list(prompts)[:tries]:
        loose = propose(scorer, state, prompt, width, max_new_tokens, PROBE_TEMPERATURE)
        fixed = propose(scorer, state, prompt, width, max_new_tokens, PROBE_TEMPERATURE,
                        seed=seed)
        unseeded.append(len({text.strip() for text, _ in loose}))
        seeded.append(len({text.strip() for text, _ in fixed}))
        if unseeded[-1] > 1 and seeded[-1] == 1:
            return (
                f"at temperature {PROBE_TEMPERATURE} this engine returns {unseeded[-1]} "
                f"distinct proposals without a seed and 1 with seed {seed}, so the seed is "
                f"replacing the draw rather than fixing it: one request carrying n samples "
                f"and one seed is being expanded into copies that share it. Every question "
                f"would be degenerate and the coverage reported would be an artefact of the "
                f"seed. Run without --seed, or send one request per sample with seeds "
                f"seed+i"), {"probe_seeded": seeded, "probe_unseeded": unseeded}
        if unseeded[-1] > 1:
            return None, {"probe_seeded": seeded, "probe_unseeded": unseeded,
                          "probe": "the seed does not collapse the draw"}
    # Every question probed came back degenerate unseeded too, so none of them say
    # anything about the seed -- and an unanswered question here is not a pass.
    return (
        f"none of the {len(unseeded)} questions probed varied at temperature "
        f"{PROBE_TEMPERATURE} even without a seed, so whether seed {seed} fixes this "
        f"engine's draw or replaces it is still unknown. A seeded run below temperature "
        f"0.5 is silently wrong if it replaces it, which is the whole reason for asking. "
        f"Run without --seed, or probe a dataset whose questions have more than one "
        f"plausible answer"), {"probe_seeded": seeded, "probe_unseeded": unseeded,
                               "probe": "undecided"}


# Above this, a total collapse cannot be the temperature's doing, so with a seed
# set it is the seed's. The number is a judgement and not a measurement: the
# sweep runs down to 0.1, where every question returning one answer is exactly
# what is being measured, and this has to sit clear of that.
SAMPLING_FLOOR = 0.5


def watch_proposals(seen, distinct, coded, minimum_distinct=0, seeded=False, temperature=None):
    """Stop a run whose proposals are not answers. Returns (abort, warning).

    Two symptoms abort. Proposals that look like option letters on a benchmark
    that has none mean the prompt is asking for something else, and no number the
    run produces is worth having. And every question returning a single answer at
    a temperature that samples, with a seed set, means the seed has replaced the
    draw rather than fixed it: one request carrying `n` samples and one seed can
    be expanded into copies that share it, in which case every question is
    degenerate and the run would record that artefact as a finding about the
    model. Neither is something a sweep should survive.

    A candidate set that collapses to one or two answers is otherwise not a
    fault. It is what low temperature does, and measuring where it starts is part
    of what the sweep is for; aborting on it would kill the run that was
    measuring it. It is reported instead, and `minimum_distinct` above zero turns
    it back into an abort for a caller that knows its run should not collapse.
    """
    if seen < 20:
        return None, None
    median = sorted(distinct)[len(distinct) // 2]
    if coded > seen / 2:
        return (f"{coded} of {seen} questions proposed what look like option letters "
                f"rather than answers. The proposal prompt is probably asking for a code"), None
    if (seeded and temperature is not None and temperature >= SAMPLING_FLOOR
            and max(distinct) == 1):
        return (f"every one of {seen} questions returned a single distinct proposal at "
                f"temperature {temperature} with a seed set. Sampling at that temperature "
                f"does not do that, so the seed is not fixing the draw but replacing it: "
                f"one request carrying n samples and one seed can be expanded into copies "
                f"that share it. Every question is degenerate and the coverage this would "
                f"report is an artefact of the seed. Rerun without --seed, or send one "
                f"request per sample with seeds seed+i"), None
    if minimum_distinct and median <= minimum_distinct:
        return (f"the median question yields {median} distinct proposals over {seen} "
                f"questions, below the {minimum_distinct} this run asked for"), None
    if median <= 2:
        return None, (f"the median question yields {median} distinct proposals over {seen} "
                      f"questions. At a low temperature that is the collapse, not a fault; "
                      f"at a high one it is worth reading the proposals above")
    return None, None



