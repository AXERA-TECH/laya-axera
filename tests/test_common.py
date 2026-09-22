"""Hardware-free tests for prompt construction and calibration."""

import numpy as np
import pytest

from laya_axera.common import (
    build_sequence,
    confidence_from_probs,
    render_options,
    temp_bucket,
    to_internal,
)


class StubTokenizer:
    """One id per whitespace token, with fixed special ids like a real checkpoint."""

    mask_token = "<mask>"
    mask_token_id = 4
    cls_token_id = 1
    sep_token_id = 2
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return {"input_ids": [10 + i for i, _ in enumerate(text.split())]}


TOK = StubTokenizer()


def test_to_internal_choice_list_and_dict():
    q = to_internal({"type": "choice", "instructions": "pick", "criteria": ["a", "b"]})
    assert list(q["crit"]) == ["a", "b"]
    q = to_internal({"type": "choice", "instructions": "pick", "criteria": {"a": "first"}})
    assert render_options(q) == ["a: first"]


@pytest.mark.parametrize(
    "qdef",
    [
        {"type": "rank", "instructions": "x"},
        {"type": "choice", "instructions": "x", "criteria": []},
        {"type": "choice", "instructions": "x", "criteria": ["a", "a"]},
        {"type": "score", "instructions": "x", "criteria": {}},
        {"type": "noul"},
    ],
)
def test_to_internal_rejects(qdef):
    with pytest.raises(ValueError):
        to_internal(qdef)


def test_noul_renders_false_then_true():
    q = to_internal({"type": "noul", "instructions": "is it?"})
    options = render_options(q)
    assert options[0].startswith("false: ") and options[1].startswith("true: ")


def test_build_sequence_shape_and_markers():
    q = to_internal(
        {"type": "choice", "instructions": "route this", "criteria": {"a": "one", "b": "two"}}
    )
    ids, markers = build_sequence(TOK, "some state text here", q, 256, 128)
    assert ids[0] == TOK.cls_token_id and ids[-1] == TOK.sep_token_id
    assert len(ids) <= 256
    assert len(markers) == 2
    assert all(ids[m] == TOK.mask_token_id for m in markers)


def test_build_sequence_truncates_long_state():
    q = to_internal({"type": "noul", "instructions": "long?"})
    state = "word " * 5000
    ids, markers = build_sequence(TOK, state, q, 256, 128)
    assert len(ids) == 256
    assert len(markers) == 2


def test_confidence_bounds():
    assert confidence_from_probs(np.array([1.0, 0.0]), 2) == pytest.approx(1.0)
    assert confidence_from_probs(np.array([0.5, 0.5]), 2) == pytest.approx(0.0)
    assert confidence_from_probs(np.array([1.0]), 1) == 1.0


def test_temp_bucket():
    assert temp_bucket(0, 2) == "choice:2"
    assert temp_bucket(1, 4) == "score:3-5"
    assert temp_bucket(2, 2) == "noul:2"
