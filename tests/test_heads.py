
from pathlib import Path

import pytest


def test_the_supervised_objective_is_unweighted_by_default():
    """Class weighting is the handicap the loss study found; it is opt-in now."""
    import inspect

    from litjev._legacy.heads import train_head

    assert inspect.signature(train_head).parameters["weighted_supervised"].default is False


def test_the_cdpo_anchor_is_unweighted_by_default():
    from litjev._legacy.heads import CDPOConfig

    assert CDPOConfig().weighted_anchor is False


def test_a_saved_head_records_the_objective_it_was_trained_under(tmp_path):
    """A checkpoint that does not say how it was trained invites benchmarking the wrong one."""
    import numpy as np

    from litjev._legacy.heads import DecisionHead, HeadMetadata

    metadata = HeadMetadata(
        model_id="m", revision="r", hidden_size=8, feature_layers=(1,),
        hidden_width=8, pca_dim=0,
        training={"objective": "supervised", "class_weighted": False},
    )
    path = tmp_path / "head.safetensors"
    DecisionHead(metadata).save(path)
    loaded = DecisionHead.load(path)
    assert loaded.metadata.training["objective"] == "supervised"
    assert loaded.metadata.training["class_weighted"] is False
    del np


def test_the_objectives_former_name_still_works_and_normalises():
    """This objective was named CDPO, renamed PMPO while the paper carried that
    name, and named CDPO again when the paper went back to it. Heads exist under
    both names, so rejecting either would make those records unreadable -- while
    keeping both as distinct values would give one objective two names comparable
    only by accident. Both are accepted and normalised, and what a run records is
    the name the paper uses now."""
    import numpy as np

    from litjev._legacy.heads import (
        CDPOConfig,
        HeadMetadata,
        PMPOConfig,
        cdpo_loss,
        normalise_objective,
        pmpo_loss,
        train_head,
    )

    assert normalise_objective("pmpo") == "cdpo"
    assert normalise_objective("cdpo") == "cdpo"
    assert normalise_objective("supervised") == "supervised"
    with pytest.raises(ValueError, match="former name"):
        normalise_objective("dpo")

    # The import names that existed before the rename still resolve, to the same
    # objects, so a script written against them is not silently running something
    # else.
    assert PMPOConfig is CDPOConfig
    assert pmpo_loss is cdpo_loss

    # And `objective="pmpo"` trains the CDPO objective rather than raising.
    rng = np.random.default_rng(0)
    meta = HeadMetadata(model_id="m", revision="r", hidden_size=4, feature_layers=(-1,))
    features = rng.normal(size=(64, meta.input_dim)).astype("float32")
    labels = rng.integers(0, 4, size=64)
    head, history = train_head(meta, features, labels, epochs=2, batch_size=16,
                               objective="pmpo", seed=0)
    assert history, "training ran under the former name"


def test_the_report_is_keyed_by_the_canonical_name_not_the_one_passed_in():
    """`run_training(objective="pmpo")` returns a report keyed `cdpo`, so a reader
    that indexes `by_objective` with the string it passed in raises KeyError. That
    is not hypothetical: `scripts/table3.py`, which produces the paper's objective
    table, did exactly that from the first rename until 2026-10-09 and would have
    crashed on any rerun. The three report readers normalise at the lookup; this
    fails if one of them stops."""
    import re

    from litjev._legacy.heads import OBJECTIVES, normalise_objective

    assert normalise_objective("pmpo") not in ("pmpo",), "the former name is not a key"
    assert normalise_objective("pmpo") in OBJECTIVES

    for name in ("ablate.py", "table3.py", "repair.py"):
        source = (Path(__file__).resolve().parents[1] / "scripts" / name).read_text()
        for hit in re.findall(r'\[.by_objective.\]\[([^\]]+)\]', source):
            assert "normalise_objective" in hit, f"{name} indexes by_objective with {hit}"


def test_both_names_of_the_config_argument_at_once_is_refused_everywhere():
    """`train_head` took `cdpo=` silently while `run_training` raised. Two rules for
    one mistake means the answer depends on which entry point you came through."""
    import inspect

    import numpy as np

    from litjev._legacy.heads import CDPOConfig, HeadMetadata, train_head
    from litjev.train.training import run_training

    meta = HeadMetadata(model_id="m", revision="r", hidden_size=4, feature_layers=(-1,))
    features = np.zeros((16, meta.input_dim), dtype="float32")
    labels = np.zeros(16, dtype="int64")
    with pytest.raises(ValueError, match="not both"):
        train_head(meta, features, labels, epochs=1, objective="cdpo",
                   cdpo=CDPOConfig(), pmpo=CDPOConfig())

    # And the two signatures agree about which name is current.
    for function in (train_head, run_training):
        names = inspect.signature(function).parameters
        assert "cdpo" in names and "pmpo" in names, function.__name__


def test_the_served_gain_is_the_one_the_loss_thresholds():
    """Train against one rule and serve under another and nothing raises; the
    head is simply wrong at every price. These are the same quantity.

    Driven through the functions each consumer actually calls, not by re-deriving
    the formula here. The first version of this test re-derived it, which is why it
    stayed green while `slime_hdm.head_trainer` was left on the old difference:
    a test that recomputes a definition checks arithmetic, not agreement.
    """
    import math

    import numpy as np
    import torch

    from litjev.core.block import escalation_gain, escalation_gain_batch

    rng = np.random.default_rng(0)
    outcomes = rng.dirichlet(np.ones(4), size=50)
    batched = escalation_gain_batch(torch.tensor(outcomes, dtype=torch.float64)).numpy()
    for outcome, trained in zip(outcomes, batched, strict=True):
        served = escalation_gain(outcome)
        assert math.isclose(served, trained, abs_tol=1e-6), (outcome, served, trained)


def test_the_off_policy_update_scores_the_rule_the_rollout_followed():
    """The importance ratio is `pi_theta / mu`, and `mu` was recorded by a rollout
    that escalated on `escalation_gain`. If the loss scores its policy by any other
    function of the head, the ratio divides two different things and the update
    chases a rule nothing executes. That is what happened for one commit, and the
    test that should have caught it re-derived the formula instead of calling it.
    """
    import math

    import numpy as np
    import torch

    from litjev.core.block import escalation_gain
    from slime_hdm.head_trainer import HeadRLConfig, _escalation_logprob

    config = HeadRLConfig()
    rng = np.random.default_rng(3)
    logits = torch.tensor(rng.normal(size=(400, 4)) * 0.5, dtype=torch.float64)
    prices = torch.tensor(rng.uniform(-0.2, 0.4, size=400), dtype=torch.float64)

    trained = _escalation_logprob(
        logits, torch.ones(400, dtype=torch.bool), prices, config).exp().numpy()
    checked = 0
    for row, price, got in zip(logits, prices, trained, strict=True):
        outcome = torch.softmax(row, dim=-1).numpy()
        want = 1.0 / (1.0 + math.exp(
            -(escalation_gain(outcome) - float(price)) / config.temperature))
        # The policy clamps to [1e-6, 1-1e-6], and a temperature of 0.05 saturates
        # most rows past it. Those rows are not a function of the gain any more, so
        # they pin nothing and are skipped rather than loosened into agreement.
        if not 1e-5 < want < 1 - 1e-5:
            continue
        checked += 1
        assert math.isclose(want, got, rel_tol=1e-6), (outcome, float(price), want, got)
    assert checked >= 20, f"only {checked} rows were off the clamp; the test is vacuous"


def test_no_module_re_derives_the_gain_as_a_difference():
    """One definition, in `litjev.core.block`. There were three copies of the old
    difference when the reward became a discount and one was missed, so the shape
    of that mistake is what this forbids."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "src"
    difference = re.compile(r"outcome\[[^]]*\b2\s*\]\s*-\s*outcome\[[^]]*\b1\s*\]"
                            r"|\[\.\.\.,\s*2\]\s*-\s*[a-z_]*\[\.\.\.,\s*1\]")
    guilty = []
    for path in root.rglob("*.py"):
        if "_legacy" in path.parts:
            continue   # kept as it shipped; nothing live imports it
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if difference.search(line):
                guilty.append(f"{path.relative_to(root)}:{number}: {line.strip()}")
    assert not guilty, ("these re-derive the gain instead of calling "
                        "escalation_gain/escalation_gain_batch:\n" + "\n".join(guilty))


def test_the_gain_is_a_log_ratio_and_not_a_difference():
    """The reward discounts rather than subtracts, so what is thresholded is the
    ratio the two modes succeed at. A difference is bounded by how much the fast
    mode already gets wrong, so it shrinks where thinking is worth most."""
    import math

    from litjev.core.block import escalation_gain

    # Same difference, very different ratio: the second is where thinking matters.
    strong = [0.85, 0.05, 0.10, 0.00]   # P(fast)=0.90  P(slow)=0.95
    weak = [0.02, 0.00, 0.07, 0.91]     # P(fast)=0.02  P(slow)=0.09
    assert math.isclose((strong[0] + strong[2]) - (strong[0] + strong[1]), 0.05, abs_tol=1e-9)
    assert math.isclose((weak[0] + weak[2]) - (weak[0] + weak[1]), 0.07, abs_tol=1e-9)
    assert escalation_gain(weak) > 6 * escalation_gain(strong), (
        "a 4.5x improvement on a question the fast mode cannot answer must rank "
        "above a 1.06x improvement on one it nearly always answers"
    )


def test_a_saturated_head_does_not_return_an_infinity():
    from litjev.core.block import escalation_gain

    assert escalation_gain([0.0, 1.0, 0.0, 0.0]) < 0
    assert escalation_gain([0.0, 0.0, 1.0, 0.0]) > 0
    for outcome in ([0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]):
        assert abs(escalation_gain(outcome)) < 100


def test_the_pca_projection_does_not_depend_on_the_svds_sign_convention(monkeypatch):
    """`U S V^T == (-U) S (-V)^T`, so LAPACK may return either, and ours disagree.

    On a byte-identical 1154x2560 standardized matrix -- same sha1, singular
    values agreeing to seven figures -- this repository's x86 and arm64 torch
    builds returned **every one of the first ten** right singular vectors with
    opposite signs. A flipped component negates that coordinate of `projection`
    while the MLP's initialisation does not follow it, so the fit lands elsewhere:
    `layer_-1_pca64` read 0.843574 on the cluster and 0.838395 on a laptop from
    the same records, the same seed and the same code, while every raw candidate
    was bit-identical on both. Seeding cannot fix that, since nothing is random.

    This simulates the other platform by negating what `svd` returns, which is the
    only way to test a platform difference on one machine.
    """
    import numpy as np
    import torch

    from litjev.core.block import DecisionHead, HeadMetadata

    meta = HeadMetadata(model_id="m", revision="r", hidden_size=16,
                        feature_layers=(-1,), hidden_width=8, pca_dim=4)
    features = np.random.default_rng(0).normal(size=(64, meta.input_dim)).astype("float32")

    def fit():
        torch.manual_seed(0)
        head = DecisionHead(meta)
        head.fit_normalizer(features)
        return head.projection.clone(), head.predict(features)

    straight_projection, straight_prediction = fit()

    real_svd = torch.linalg.svd

    def flipped(matrix, **kwargs):
        left, singular, right = real_svd(matrix, **kwargs)
        # A correct decomposition with the other sign convention.
        return -left, singular, -right

    monkeypatch.setattr(torch.linalg, "svd", flipped)
    flipped_projection, flipped_prediction = fit()

    assert torch.allclose(straight_projection, flipped_projection), \
        "the projection follows the SVD's sign convention, so PCA heads are not portable"
    assert np.allclose(straight_prediction, flipped_prediction)
    # And the fix is not vacuous: the sign convention really did change what came
    # back, so an unpinned projection would have differed here.
    assert not torch.allclose(real_svd(torch.as_tensor(features[:, :meta.hidden_dim]),
                                       full_matrices=False)[2],
                              flipped(torch.as_tensor(features[:, :meta.hidden_dim]),
                                      full_matrices=False)[2])


def test_a_feature_constant_in_training_does_not_saturate_the_head():
    """`std.clamp_min(1e-6)` was an amplifier, not division-by-zero protection.

    The question-type one-hots are the last three features and every MMLU-Pro row
    is `choice`, so in training they are constant and the old code gave all three
    `std = 1e-6`. A `noul` question then presented plus and minus **one million**
    where every trained feature spans about plus or minus three, which saturated
    the head rather than biasing it: every gain collapsed to exactly zero, so
    nothing escalated at a non-negative price and all such items sat at one value.

    That is the mechanism behind this project's reported JevBench type refusal, so
    the test is written on the quantity that was reported: the spread of the gain
    over items that differ only in a feature the training set never varied.
    """
    import numpy as np
    import torch

    from litjev.core.block import DecisionHead, HeadMetadata
    from litjev.core.block_utils import build_features
    from litjev._legacy.routing import escalation_gain

    rng = np.random.default_rng(0)
    meta = HeadMetadata(model_id="m", revision="r", hidden_size=16, feature_layers=(-1,))
    # A training set in which the last feature never varies, as `choice` does not.
    train = rng.normal(size=(128, meta.input_dim)).astype("float32")
    train[:, -1] = 1.0
    torch.manual_seed(0)
    head = DecisionHead(meta)
    head.fit_normalizer(train)

    assert head.std[-1].item() == 1.0, "a constant feature must get unit scale"
    # And the amplification is gone: the old code gave 1e6 here.
    standardized = (0.0 - head.mean[-1].item()) / head.std[-1].item()
    assert abs(standardized) <= 2.0, f"a flipped one-hot standardises to {standardized}"

    # The head still discriminates among items whose only difference from the
    # training distribution is that feature.
    serve = rng.normal(size=(64, meta.input_dim)).astype("float32")
    serve[:, -1] = 0.0
    gains = np.array([escalation_gain(row) for row in head.predict(serve)])
    assert len(set(np.round(gains, 6))) > 1, "the head returns one value for every item"
    assert gains.std() > 1e-3, f"gain spread collapsed to {gains.std()}"


def test_an_old_head_refuses_to_saturate_instead_of_returning_one_answer():
    """The guard that would have caught the type refusal at its first serving.

    A head fitted before unit scale has `std` at the clamp floor on every feature
    its training set never varied. Serving a deviating value there standardises to
    about a million and the head returns the same outcome for every row. It must
    say so rather than answer.

    And it must *not* refuse the case that is fine: on a benchmark where those
    features hold their training constants, every one standardises to exactly zero
    whatever the divisor is, which is why this is checked per call and not at load.
    """
    import numpy as np
    import pytest
    import torch

    from litjev.core.block import DecisionHead, HeadMetadata

    meta = HeadMetadata(model_id="m", revision="r", hidden_size=16, feature_layers=(-1,))
    rng = np.random.default_rng(0)
    train = rng.normal(size=(128, meta.input_dim)).astype("float32")
    train[:, -1] = 1.0
    torch.manual_seed(0)
    head = DecisionHead(meta)
    head.fit_normalizer(train)
    # Put the old scaling back on that feature, as a head on disk carries it.
    head.std[-1] = 1e-6

    # The case that is fine: the feature holds its training constant.
    ok = rng.normal(size=(8, meta.input_dim)).astype("float32")
    ok[:, -1] = 1.0
    head.predict(ok)

    # The case that saturated and was reported as a finding.
    bad = ok.copy()
    bad[:, -1] = 0.0
    with pytest.raises(ValueError, match="saturate"):
        head.predict(bad)
    # And the message points at the cause rather than at the symptom.
    with pytest.raises(ValueError, match="clamp floor"):
        head.predict(bad)
