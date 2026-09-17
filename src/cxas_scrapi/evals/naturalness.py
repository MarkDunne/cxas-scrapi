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


"""Optional Naturalness Metric for SimulationEvals.

Grades how human an agent sounds, per turn and in aggregate, using Gemini.

The metric is entirely opt-in. A simulation test case that does not declare a
``naturalness_metric`` block behaves exactly as it did before this module
existed: no extra Gemini calls are made and no extra keys appear in the
results. This keeps existing simulation YAML schemas both backward and
forward compatible.
"""

import enum
import logging
import os
import statistics
import typing
from typing import Any

import pydantic
from google import genai

from cxas_scrapi.prompts import naturalness_prompts

logger = logging.getLogger(__name__)

# Keys a test case may use to declare the metric. The plural/suffixed forms
# are accepted so hand-written YAML does not fail on a near-miss.
_CONFIG_KEYS = (
    "naturalness_metric",
    "naturalness",
    "naturalness_config",
)

_MAX_UTTERANCE_CHARS = 400


class NaturalnessLabel(str, enum.Enum):
    """Graded label assigned to an agent turn and to the whole call."""

    BOT_LIKE = "Bot-like"
    TRANSITIONAL = "Transitional"
    HUMAN_LIKE = "Human-like"


class NaturalnessFactor(pydantic.BaseModel):
    """A single scored conversational quality, e.g. pacing or grammarStyle."""

    quality: str = ""
    score: int = 3
    value: str = ""
    reason: str = ""


class TurnNaturalness(pydantic.BaseModel):
    """Naturalness grading for one agent turn."""

    turn_index: int = 0
    agent_utterance: str = ""
    label: NaturalnessLabel = NaturalnessLabel.TRANSITIONAL
    score: float = 0.0
    justification: str = ""
    factors: list[NaturalnessFactor] = []


class NaturalnessOutput(pydantic.BaseModel):
    """Raw structured response returned by the grading model."""

    turns: list[TurnNaturalness] = []
    conversation_factors: list[NaturalnessFactor] = []
    summary: str = ""


class NaturalnessConfig(pydantic.BaseModel):
    """Per-test-case configuration for the Naturalness Metric.

    Unknown keys are preserved rather than rejected so that a YAML file
    written against a newer version of the library still loads here.
    """

    model_config = pydantic.ConfigDict(extra="allow")

    enabled: bool = True
    # Falls back to the simulation's eval_model when unset.
    model: str | None = None
    # Override the graded dimensions. Empty means "use the defaults".
    turn_qualities: list[str] = []
    conversation_qualities: list[str] = []
    # Free-text rubric addendum, e.g. brand voice or locale specifics.
    extra_guidance: str = ""
    # Blend between the per-turn mean and the conversation-level factors.
    turn_weight: float = 0.75
    # Score bands used to derive labels from numeric scores.
    bot_like_below: float = 2.5
    human_like_at_or_above: float = 3.75
    # When set, the simulation's pass/fail also requires this overall score.
    pass_threshold: float | None = None
    # Interleave captured agent audio so speech is judged acoustically.
    # Requires the simulation to run with capture_agent_audio=True.
    use_audio: bool = False
    # Show tool calls in the transcript so "thinking out loud while looking
    # something up" can be judged in context.
    include_tool_calls: bool = True

    @pydantic.field_validator("turn_weight")
    @classmethod
    def _clamp_turn_weight(cls, v: float) -> float:
        return min(max(v, 0.0), 1.0)


class NaturalnessResult(pydantic.BaseModel):
    """Aggregated Naturalness Metric result for a whole simulation."""

    overall_score: float = 0.0
    overall_label: NaturalnessLabel = NaturalnessLabel.TRANSITIONAL
    turn_count: int = 0
    turns: list[TurnNaturalness] = []
    conversation_factors: list[NaturalnessFactor] = []
    # Mean score per quality across all graded turns.
    factor_averages: dict[str, float] = {}
    # How many turns landed in each label band.
    label_counts: dict[str, int] = {}
    summary: str = ""
    model: str = ""
    pass_threshold: float | None = None
    passed: bool | None = None

    def model_dump(self, **kwargs: typing.Any) -> typing.Any:
        kwargs.setdefault("mode", "json")
        return super().model_dump(**kwargs)


class AgentTurn(pydantic.BaseModel):
    """One (user, agent) exchange extracted from a simulation trace."""

    turn_index: int
    user: str = ""
    agent: str = ""
    tool_calls: list[str] = []


def parse_naturalness_config(
    test_case: dict[str, Any] | None,
    override: Any = None,
) -> NaturalnessConfig | None:
    """Resolves the naturalness configuration for a test case.

    Args:
        test_case: The simulation test case dict. May omit the metric
            entirely, which is the common case.
        override: A run-level override. ``True`` enables the metric with
            defaults for every test case, ``False`` force-disables it, and a
            dict is merged over whatever the test case declared.

    Returns:
        A :class:`NaturalnessConfig` when the metric should run, otherwise
        ``None``. Malformed configuration is logged and treated as "off" so
        that a bad block can never fail an otherwise healthy simulation.
    """
    if override is False:
        return None

    raw: Any = None
    if test_case:
        for key in _CONFIG_KEYS:
            if key in test_case:
                raw = test_case[key]
                break

    if isinstance(override, dict):
        base = dict(raw) if isinstance(raw, dict) else {}
        base.update(override)
        raw = base
    elif override is True and raw is None:
        raw = True

    if raw is None:
        return None
    # `naturalness_metric: true` / `false` shorthand.
    if isinstance(raw, bool):
        return NaturalnessConfig() if raw else None
    if not isinstance(raw, dict):
        logger.warning(
            "Ignoring naturalness_metric: expected a mapping or bool, got %s.",
            type(raw).__name__,
        )
        return None

    try:
        config = NaturalnessConfig(**raw)
    except pydantic.ValidationError as exc:
        logger.warning("Ignoring malformed naturalness_metric config: %s", exc)
        return None

    return config if config.enabled else None


def _clean_agent_line(line: str) -> str:
    """Strips the trace prefix from an agent text line."""
    for prefix in ("Agent Text: ", "Agent: "):
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    return ""


def extract_agent_turns(trace: list[str]) -> list[AgentTurn]:
    """Pairs user utterances with the agent response that followed.

    ``trace`` is the ``detailed_trace`` built by
    :meth:`SimulationEvals.simulate_conversation`: alternating ``"User: ..."``
    entries and multi-line agent blocks whose lines are prefixed with
    ``"Agent Text: "``, ``"Tool Call ..."`` and similar.
    """
    turns: list[AgentTurn] = []
    pending_user = ""
    index = 0

    for entry in trace or []:
        if not isinstance(entry, str):
            continue
        if entry.startswith("User: "):
            pending_user = entry[len("User: ") :].strip()
            continue

        agent_parts: list[str] = []
        tool_calls: list[str] = []
        for raw_line in entry.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            text = _clean_agent_line(line)
            if text:
                agent_parts.append(text)
            elif line.startswith("Tool Call"):
                tool_calls.append(line)

        if not agent_parts and not tool_calls:
            continue

        turns.append(
            AgentTurn(
                turn_index=index,
                user=pending_user,
                agent=" ".join(agent_parts),
                tool_calls=tool_calls,
            )
        )
        pending_user = ""
        index += 1

    return [t for t in turns if t.agent]


def _render_transcript(turns: list[AgentTurn], include_tool_calls: bool) -> str:
    """Formats turns for the grader, numbering each agent turn."""
    lines: list[str] = []
    for turn in turns:
        if turn.user:
            lines.append(f"[user] {turn.user}")
        if include_tool_calls and turn.tool_calls:
            for tool_call in turn.tool_calls:
                lines.append(f"    ({tool_call})")
        lines.append(f"[agent turn {turn.turn_index}] {turn.agent}")
        lines.append("")
    return "\n".join(lines).strip()


def _qualities_block(overrides: list[str], default_block: str) -> str:
    """Renders the quality list, honouring a caller override."""
    if not overrides:
        return default_block
    return "\n".join(f"-   `{q}`" for q in overrides)


def _build_prompt(turns: list[AgentTurn], config: NaturalnessConfig) -> str:
    """Assembles the grading prompt from the rubric and the transcript."""
    extra = config.extra_guidance.strip()
    if extra:
        extra = f"## Additional guidance for this agent\n\n{extra}"

    return (
        naturalness_prompts.NATURALNESS_METRIC_PROMPT.replace(
            "{rubric}", naturalness_prompts.NATURALNESS_RUBRIC
        )
        .replace(
            "{turn_qualities}",
            _qualities_block(
                config.turn_qualities,
                naturalness_prompts.DEFAULT_TURN_QUALITIES,
            ),
        )
        .replace(
            "{conversation_qualities}",
            _qualities_block(
                config.conversation_qualities,
                naturalness_prompts.DEFAULT_CONVERSATION_QUALITIES,
            ),
        )
        .replace("{extra_guidance}", extra)
        .replace(
            "{transcript}",
            _render_transcript(turns, config.include_tool_calls),
        )
    )


def _build_audio_contents(
    prompt: str,
    turns: list[AgentTurn],
    audio_paths: dict[int, str],
) -> list[Any]:
    """Interleaves captured agent WAVs after each agent turn."""
    contents: list[Any] = [prompt, "\n\nRAW AGENT AUDIO BY TURN:\n"]
    for turn in turns:
        path = audio_paths.get(turn.turn_index)
        if not path or not os.path.exists(path):
            continue
        contents.append(f"\n[audio for agent turn {turn.turn_index}]\n")
        try:
            with open(path, "rb") as handle:
                contents.append(
                    genai.types.Part.from_bytes(
                        data=handle.read(), mime_type="audio/wav"
                    )
                )
        except OSError as exc:
            logger.warning("Could not attach audio %s: %s", path, exc)
    return contents


def _label_for_score(
    score: float, config: NaturalnessConfig
) -> NaturalnessLabel:
    """Maps a 1-5 score onto a graded label using the configured bands."""
    if score < config.bot_like_below:
        return NaturalnessLabel.BOT_LIKE
    if score >= config.human_like_at_or_above:
        return NaturalnessLabel.HUMAN_LIKE
    return NaturalnessLabel.TRANSITIONAL


def _clamp_score(value: float) -> float:
    return min(max(float(value), 1.0), 5.0)


def _normalise_turns(
    graded: list[TurnNaturalness],
    source: list[AgentTurn],
    config: NaturalnessConfig,
) -> list[TurnNaturalness]:
    """Repairs model output: clamps scores and re-derives labels.

    The turn score is recomputed as the mean of the factor scores rather than
    taken from the model's separate holistic number. The factors are the
    evidence the model actually cited, so deriving from them keeps
    ``overall_score``, ``factor_averages`` and the labels mutually consistent
    and makes the metric reproducible.
    """
    by_index = {t.turn_index: t for t in source}
    normalised: list[TurnNaturalness] = []

    for turn in graded:
        for factor in turn.factors:
            factor.score = int(min(max(factor.score, 1), 5))

        score = (
            statistics.fmean(f.score for f in turn.factors)
            if turn.factors
            else turn.score
        )

        turn.score = round(_clamp_score(score), 2)
        turn.label = _label_for_score(turn.score, config)

        if not turn.agent_utterance:
            original = by_index.get(turn.turn_index)
            if original:
                turn.agent_utterance = original.agent[:_MAX_UTTERANCE_CHARS]

        normalised.append(turn)

    normalised.sort(key=lambda t: t.turn_index)
    return normalised


def _factor_averages(turns: list[TurnNaturalness]) -> dict[str, float]:
    """Means each quality across all graded turns."""
    buckets: dict[str, list[int]] = {}
    for turn in turns:
        for factor in turn.factors:
            if factor.quality:
                buckets.setdefault(factor.quality, []).append(factor.score)
    return {
        quality: round(statistics.fmean(scores), 2)
        for quality, scores in sorted(buckets.items())
    }


def _aggregate(
    turns: list[TurnNaturalness],
    conversation_factors: list[NaturalnessFactor],
    config: NaturalnessConfig,
) -> float:
    """Blends the per-turn mean with the conversation-level factor mean."""
    if not turns:
        return 0.0

    turn_mean = statistics.fmean(t.score for t in turns)
    conv_scores = [
        int(min(max(f.score, 1), 5)) for f in conversation_factors if f.quality
    ]
    if not conv_scores:
        return round(_clamp_score(turn_mean), 2)

    conv_mean = statistics.fmean(conv_scores)
    blended = (
        config.turn_weight * turn_mean + (1.0 - config.turn_weight) * conv_mean
    )
    return round(_clamp_score(blended), 2)


def evaluate_naturalness(
    gemini_client: Any,
    model_name: str,
    trace: list[str],
    config: NaturalnessConfig,
    audio_paths: dict[int, str] | None = None,
) -> NaturalnessResult | None:
    """Grades the naturalness of every agent turn in a simulation trace.

    Args:
        gemini_client: A ``GeminiGenerate`` instance.
        model_name: Fallback model, used when the config does not pin one.
        trace: The simulation's ``detailed_trace``.
        config: Resolved metric configuration.
        audio_paths: Optional map of simulation turn number to agent WAV
            path, used only when ``config.use_audio`` is set.

    Returns:
        A :class:`NaturalnessResult`, or ``None`` when there was nothing to
        grade or the grading call failed. Failures are logged rather than
        raised so the metric can never break a simulation run.
    """
    turns = extract_agent_turns(trace)
    if not turns:
        logger.info("Naturalness metric skipped: no agent turns in trace.")
        return None

    target_model = config.model or model_name
    prompt: Any = _build_prompt(turns, config)

    if config.use_audio and audio_paths:
        prompt = _build_audio_contents(prompt, turns, audio_paths)

    try:
        output: NaturalnessOutput | None = gemini_client.generate(
            prompt=prompt,
            model_name=target_model,
            response_mime_type="application/json",
            response_schema=NaturalnessOutput,
        )
    except Exception as exc:  # noqa: BLE001 - metric must never break a run
        logger.error("Naturalness grading failed: %s", exc)
        return None

    if not output or not output.turns:
        logger.warning("Naturalness grading returned no turn gradings.")
        return None

    graded_turns = _normalise_turns(output.turns, turns, config)
    for factor in output.conversation_factors:
        factor.score = int(min(max(factor.score, 1), 5))

    overall = _aggregate(graded_turns, output.conversation_factors, config)
    label_counts: dict[str, int] = {
        label.value: 0 for label in NaturalnessLabel
    }
    for turn in graded_turns:
        label_counts[turn.label.value] += 1

    passed = None
    if config.pass_threshold is not None:
        passed = overall >= config.pass_threshold

    return NaturalnessResult(
        overall_score=overall,
        overall_label=_label_for_score(overall, config),
        turn_count=len(graded_turns),
        turns=graded_turns,
        conversation_factors=output.conversation_factors,
        factor_averages=_factor_averages(graded_turns),
        label_counts=label_counts,
        summary=output.summary,
        model=target_model,
        pass_threshold=config.pass_threshold,
        passed=passed,
    )
