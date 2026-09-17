# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Unit tests for the optional Naturalness Metric."""

import json
import typing

import pytest

from cxas_scrapi.evals.naturalness import (
    NaturalnessConfig,
    NaturalnessFactor,
    NaturalnessLabel,
    NaturalnessOutput,
    TurnNaturalness,
    evaluate_naturalness,
    extract_agent_turns,
    parse_naturalness_config,
)

TRACE = [
    "User: event: welcome",
    "Agent Text: Thank you for calling. How may I assist you today?",
    "User: I was charged twice and I am furious",
    (
        "Tool Call (Output): lookup_account with args {'id': 1}\n"
        "Agent Text: [warm] Oh no... I'm really sorry about that.\n"
        "Agent Text: [short pause] Let me pull it up, one sec."
    ),
    "User: fine. account is one two three",
    "Agent Text: Got it. Give me just a moment...",
]


class _FakeClient:
    """Minimal stand-in for GeminiGenerate that returns a canned output."""

    def __init__(self, output: typing.Any) -> None:
        self.output = output
        self.last_prompt: typing.Any = None
        self.call_count = 0

    def generate(
        self, prompt: typing.Any = None, **_: typing.Any
    ) -> typing.Any:
        self.last_prompt = prompt
        self.call_count += 1
        return self.output


def _turn(idx: int, scores: list[int]) -> TurnNaturalness:
    qualities = ["emotion", "pacing", "grammarStyle"]
    return TurnNaturalness(
        turn_index=idx,
        agent_utterance=f"utterance {idx}",
        # A deliberately over-generous holistic score: the factor mean is
        # what the metric must actually use.
        label=NaturalnessLabel.HUMAN_LIKE,
        score=5.0,
        justification="j",
        factors=[
            NaturalnessFactor(quality=q, score=s, reason="r")
            for q, s in zip(qualities, scores, strict=True)
        ],
    )


def _output() -> NaturalnessOutput:
    return NaturalnessOutput(
        turns=[_turn(0, [1, 1, 1]), _turn(1, [4, 4, 4]), _turn(2, [5, 5, 5])],
        conversation_factors=[
            NaturalnessFactor(quality="personaConsistency", score=3, reason="r")
        ],
        summary="summary text",
    )


def test_extract_agent_turns_pairs_user_and_agent() -> None:
    turns = extract_agent_turns(TRACE)

    assert len(turns) == 3
    assert turns[0].turn_index == 0
    assert turns[0].user == "event: welcome"
    assert turns[1].user == "I was charged twice and I am furious"
    assert turns[2].agent == "Got it. Give me just a moment..."


def test_extract_agent_turns_preserves_emotive_tags_and_tool_calls() -> None:
    turns = extract_agent_turns(TRACE)

    assert "[warm]" in turns[1].agent
    assert "[short pause]" in turns[1].agent
    assert turns[1].tool_calls == [
        "Tool Call (Output): lookup_account with args {'id': 1}"
    ]


def test_extract_agent_turns_ignores_agent_only_tool_noise() -> None:
    assert extract_agent_turns([]) == []
    assert extract_agent_turns(["User: hello"]) == []


@pytest.mark.parametrize(
    "test_case",
    [
        None,
        {"name": "x"},
        {"naturalness_metric": False},
        {"naturalness_metric": {"enabled": False}},
    ],
)
def test_parse_naturalness_config_disabled(
    test_case: dict[str, typing.Any] | None,
) -> None:
    """A test case without (or opting out of) the metric leaves it off."""
    assert parse_naturalness_config(test_case) is None


@pytest.mark.parametrize(
    "key", ["naturalness_metric", "naturalness", "naturalness_config"]
)
def test_parse_naturalness_config_shorthand_and_aliases(key: str) -> None:
    config = parse_naturalness_config({key: True})

    assert config is not None
    assert config.enabled is True
    assert config.turn_weight == 0.75
    assert config.pass_threshold is None


def test_parse_naturalness_config_tolerates_unknown_keys() -> None:
    """Forward compatible: a newer YAML must still load on an older lib."""
    config = parse_naturalness_config(
        {"naturalness_metric": {"some_future_key": 42, "pass_threshold": 3.0}}
    )

    assert config is not None
    assert config.pass_threshold == 3.0


@pytest.mark.parametrize(
    "raw", [{"turn_weight": "not-a-number"}, ["bad"], "bad"]
)
def test_parse_naturalness_config_malformed_is_off_not_fatal(
    raw: typing.Any,
) -> None:
    assert parse_naturalness_config({"naturalness_metric": raw}) is None


def test_parse_naturalness_config_clamps_turn_weight() -> None:
    config = parse_naturalness_config(
        {"naturalness_metric": {"turn_weight": 9}}
    )

    assert config is not None
    assert config.turn_weight == 1.0


def test_parse_naturalness_config_run_level_overrides() -> None:
    # True enables the metric for a test case that declared nothing.
    assert parse_naturalness_config({"name": "x"}, True) is not None
    # False force-disables it everywhere.
    assert parse_naturalness_config({"naturalness_metric": True}, False) is None
    # A dict is merged OVER whatever the test case declared.
    merged = parse_naturalness_config(
        {"naturalness_metric": {"pass_threshold": 3.0}},
        {"model": "gemini-3.1-pro-preview"},
    )
    assert merged is not None
    assert merged.pass_threshold == 3.0
    assert merged.model == "gemini-3.1-pro-preview"


def test_evaluate_naturalness_scores_turns_from_factor_means() -> None:
    client = _FakeClient(_output())

    result = evaluate_naturalness(
        client, "fake-model", TRACE, NaturalnessConfig()
    )

    assert result is not None
    # The grader claimed a holistic 5.0 on every turn; the factors it
    # actually cited must win.
    assert result.turns[0].score == 1.0
    assert result.turns[0].label is NaturalnessLabel.BOT_LIKE
    assert result.turns[1].score == 4.0
    assert result.turns[1].label is NaturalnessLabel.HUMAN_LIKE
    assert result.turns[2].score == 5.0
    assert result.turns[2].label is NaturalnessLabel.HUMAN_LIKE


def test_evaluate_naturalness_blends_turn_and_conversation_scores() -> None:
    result = evaluate_naturalness(
        _FakeClient(_output()), "fake-model", TRACE, NaturalnessConfig()
    )

    assert result is not None
    # turn mean = 3.3333, conversation mean = 3.0
    # -> 0.75 * 3.3333 + 0.25 * 3.0 = 3.25
    assert result.overall_score == 3.25
    assert result.overall_label is NaturalnessLabel.TRANSITIONAL
    assert result.turn_count == 3
    assert result.summary == "summary text"
    assert result.model == "fake-model"
    assert result.factor_averages == {
        "emotion": 3.33,
        "grammarStyle": 3.33,
        "pacing": 3.33,
    }
    assert result.label_counts == {
        "Bot-like": 1,
        "Transitional": 0,
        "Human-like": 2,
    }


def test_evaluate_naturalness_uses_turn_mean_without_conversation_factors() -> (
    None
):
    output = _output()
    output.conversation_factors = []

    result = evaluate_naturalness(
        _FakeClient(output), "fake-model", TRACE, NaturalnessConfig()
    )

    assert result is not None
    assert result.overall_score == 3.33


@pytest.mark.parametrize(
    ("threshold", "expected"),
    [(3.0, True), (4.5, False), (None, None)],
)
def test_evaluate_naturalness_pass_threshold(
    threshold: float | None, expected: bool | None
) -> None:
    result = evaluate_naturalness(
        _FakeClient(_output()),
        "fake-model",
        TRACE,
        NaturalnessConfig(pass_threshold=threshold),
    )

    assert result is not None
    assert result.passed is expected


def test_evaluate_naturalness_prompt_includes_turns_and_rubric() -> None:
    client = _FakeClient(_output())

    evaluate_naturalness(client, "fake-model", TRACE, NaturalnessConfig())

    assert "[agent turn 1]" in client.last_prompt
    assert "Bot-like" in client.last_prompt
    assert "grammarStyle" in client.last_prompt
    # include_tool_calls defaults to True.
    assert "lookup_account" in client.last_prompt


def test_evaluate_naturalness_pins_configured_model() -> None:
    config = NaturalnessConfig(model="gemini-3.1-pro-preview")

    result = evaluate_naturalness(
        _FakeClient(_output()), "fallback-model", TRACE, config
    )

    assert result is not None
    assert result.model == "gemini-3.1-pro-preview"


def test_evaluate_naturalness_result_is_json_serialisable() -> None:
    result = evaluate_naturalness(
        _FakeClient(_output()), "fake-model", TRACE, NaturalnessConfig()
    )

    assert result is not None
    # Must survive the round trip into sim_results.json.
    dumped = json.loads(json.dumps(result.model_dump()))
    assert dumped["overall_label"] == "Transitional"


def test_evaluate_naturalness_empty_trace_returns_none() -> None:
    client = _FakeClient(_output())

    assert evaluate_naturalness(client, "m", [], NaturalnessConfig()) is None
    assert client.call_count == 0


def test_evaluate_naturalness_empty_model_output_returns_none() -> None:
    assert (
        evaluate_naturalness(_FakeClient(None), "m", TRACE, NaturalnessConfig())
        is None
    )
    assert (
        evaluate_naturalness(
            _FakeClient(NaturalnessOutput()), "m", TRACE, NaturalnessConfig()
        )
        is None
    )


def test_evaluate_naturalness_grading_error_returns_none() -> None:
    """A quota error or similar must never fail the simulation."""

    class _Boom:
        def generate(self, **_: typing.Any) -> typing.Any:
            raise RuntimeError("quota exceeded")

    assert (
        evaluate_naturalness(_Boom(), "m", TRACE, NaturalnessConfig()) is None
    )
