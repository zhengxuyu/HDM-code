"""The main table's fast column, checked against a collection made another way.

27948 reported 0.6153 for the 27B fast mode where Table 1 says 0.665, on the
same benchmark, the same split and the same 1500-question stride-8 subset, which
read as a table-integrity problem: a published column produced under a setting
nobody could reproduce.

It is not. Table 1 reports the held-out categories -- chemistry, economics and
psychology -- and 27948 covered every category. Restricted to the three, the
independently collected rows give 0.6590 against the table's 0.6647 and 0.5376
against its 0.538, on the same 346 questions, from a different job, through a
different code path (`hde-collect --fast-only` rather than `bench_routing`),
and for both backbones.

So this is not a regression test on a constant. It pins Table 1's fast column to
data collected separately from the run that produced it, and it fires if the
readout path stops agreeing with itself.
"""
import csv
from pathlib import Path

import pytest

ROWS = Path("experiments/mmlu-pro/results/position-rows-27b-27948.csv")
# Section 4: the head is trained on the remaining categories and reported on these.
HELD_OUT = {"chemistry", "economics", "psychology"}
# Table 1, the "Fast only (System One)" rows.
PUBLISHED = {"fast_correct_27b": 0.665, "fast_correct_4b": 0.538}
# 27930 against 27063 disagreed on 21 of 1500, every one a near-tie, so a few
# questions of slack is the readout's own noise rather than a difference in kind.
SLACK = 3 / 346


@pytest.fixture(scope="module")
def rows(pytestconfig):
    path = pytestconfig.rootpath / ROWS
    if not path.exists():
        pytest.skip(f"{ROWS} is not in this checkout")
    with open(path) as handle:
        return list(csv.DictReader(handle))


def test_the_subset_is_the_one_the_table_reports(rows):
    """346 questions in three categories, which is the table's own n."""
    held = [row for row in rows if row["category"] in HELD_OUT]
    assert len(rows) == 1500
    assert len(held) == 346, "a different n would make the comparison a different one"


@pytest.mark.parametrize("column", sorted(PUBLISHED))
def test_the_fast_column_reproduces_on_the_held_out_categories(rows, column):
    held = [row for row in rows if row["category"] in HELD_OUT]
    measured = sum(int(row[column]) for row in held) / len(held)
    assert measured == pytest.approx(PUBLISHED[column], abs=SLACK), (
        f"{column}: {measured:.4f} against the table's {PUBLISHED[column]}, a gap of "
        f"{abs(measured - PUBLISHED[column]) * len(held):.0f} questions")


def test_the_whole_subset_is_lower_which_is_the_categories_and_not_a_fault(rows):
    """The number that looked like a problem, and why it is not one: the held-out
    three are not a random sample of MMLU-Pro, and over every category the same
    readout scores lower."""
    everything = sum(int(row["fast_correct_27b"]) for row in rows) / len(rows)
    held = [row for row in rows if row["category"] in HELD_OUT]
    inside = sum(int(row["fast_correct_27b"]) for row in held) / len(held)
    assert everything < inside
    assert everything == pytest.approx(0.6153, abs=0.002), "27948's reported figure"
