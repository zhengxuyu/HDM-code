"""What a run needs in order to build a candidate set for a question without one.

Kept apart from the engine because these are a run's parameters rather than the
engine's state: two runs of one engine at different temperatures are two
different measurements, and the record has to be able to say which it was.

Every field is recorded with whatever the run produces. The temperature and the
width decide how many distinct candidates a question yields, which decides
whether it has a choice to read out at all; the seed decides whether the run can
be repeated; and the shuffle seed decides the order the options are listed in,
which is a channel the readout has been measured to use.
"""

from dataclasses import asdict, dataclass

from litjev.candidates.construct import LIKELIHOOD

# A fixed seed rather than a random one: the listing order has to be the same on
# a resume as it was before the run died, and the same on a rerun as it was the
# first time. It is combined with the question id, so questions differ from each
# other while a question does not differ from itself.
DEFAULT_SHUFFLE = "litjev"


@dataclass(frozen=True)
class ConstructionSettings:
    """How to turn a question with no candidate set into one with candidates.

    `shuffle` is the run's seed for the listing order, combined with the question
    id, and it is on by default. Rank order was the default for every run before
    this and is the reason the readout's slot preference was also a rank
    preference: it took the runner-up on all but one of 478 questions, and no
    number could say whether that was a judgement about the answers or about the
    slot. Rotation then found the preference positional, and the content readout
    keeps a listing order, so the confound outlives the labels.

    `None` turns it off, for a comparison that wants rank order on purpose. It
    is recorded either way, since two runs that list the same candidates in
    different orders are two different measurements.
    """

    # The proposal is a generation step and the readout is not, so they cannot
    # share a state. Once the readout's is unified to "Answer each question using
    # its listed options", handing it to the proposer tells the model to use
    # options that do not exist yet -- which is the whole reason it is proposing.
    state: str = "Answer the question."
    width: int = 10
    max_new_tokens: int = 32
    temperature: float = 0.7
    rank: str = LIKELIHOOD
    seed: int | None = None
    shuffle: str | None = DEFAULT_SHUFFLE
    verified: bool = True

    def __post_init__(self):
        if self.width < 2:
            raise ValueError("fewer than two proposals is not a choice")
        if self.temperature is not None and self.temperature <= 0:
            raise ValueError(
                "sampling at temperature zero returns one answer width times, which is "
                "not a candidate set; the collapse is worth measuring just above it")

    def as_record(self):
        """For the run's record, since two runs at different settings are two
        different measurements and the file has to say which."""
        return asdict(self)
