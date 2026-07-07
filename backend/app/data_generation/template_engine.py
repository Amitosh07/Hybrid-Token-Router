"""Template engine for metadata-rich prompt synthesis."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from app.data_generation.constraint_library import render_constraint
from app.data_generation.domain_library import DomainProfile


@dataclass(frozen=True)
class PromptSpec:
    """Complete prompt-generation specification."""

    prompt_id: str
    category: str
    difficulty: str
    expected_reasoning: str
    domain: str
    output_format: str
    constraint_names: tuple[str, ...]
    constraint_count: int
    estimated_complexity: float
    prompt_length: str
    variant: int
    generation_template: str
    prompt: str

    def to_dict(self) -> dict[str, object]:
        """Return a serializable dictionary."""
        return asdict(self)


class TemplateEngine:
    """Render diverse prompts from structured specifications."""

    def render(
        self,
        *,
        prompt_id: str,
        category: str,
        difficulty: str,
        expected_reasoning: str,
        domain: DomainProfile,
        output_format: str,
        constraints: list[object],
        prompt_length: str,
        variant: int,
    ) -> PromptSpec:
        """Render one prompt specification."""
        actor = domain.actors[variant % len(domain.actors)]
        artifact = domain.artifacts[(variant // 2) % len(domain.artifacts)]
        risk = domain.risks[(variant // 3) % len(domain.risks)]
        metric = domain.metrics[(variant // 5) % len(domain.metrics)]
        entity = domain.entities[(variant // 7) % len(domain.entities)]
        constraint_text = " ".join(render_constraint(c, variant) for c in constraints)
        task = self._task_sentence(category, difficulty, actor, artifact, risk, metric, entity, variant)
        format_sentence = f"Format the response as {output_format}."
        length_sentence = self._length_sentence(prompt_length, difficulty)
        prompt = " ".join(part for part in (task, format_sentence, length_sentence, constraint_text) if part).strip()
        complexity = self._estimated_complexity(difficulty, expected_reasoning, len(constraints), prompt_length)
        template_name = f"{category}_{variant % 2}"
        return PromptSpec(
            prompt_id=prompt_id,
            category=category,
            difficulty=difficulty,
            expected_reasoning=expected_reasoning,
            domain=domain.name,
            output_format=output_format,
            constraint_names=tuple(c.name for c in constraints),
            constraint_count=len(constraints),
            estimated_complexity=complexity,
            prompt_length=prompt_length,
            variant=variant,
            generation_template=template_name,
            prompt=prompt,
        )

    def _task_sentence(
        self,
        category: str,
        difficulty: str,
        actor: str,
        artifact: str,
        risk: str,
        metric: str,
        entity: str,
        variant: int,
    ) -> str:
        verbs = {
            "coding": (
                f"Write production-ready code for a {actor} that processes a {artifact} involving {entity} and handles {risk}.",
                f"Implement a service component that tracks {metric} while validating a {artifact} for {entity}.",
            ),
            "planning": (
                f"Create an implementation plan for a {actor} to improve {metric} while reducing {risk}.",
                f"Design a phased rollout for {artifact} adoption across teams that depend on {entity}.",
            ),
            "reasoning": (
                f"Analyze whether a {actor} should prioritize {artifact} improvements or direct mitigation of {risk}.",
                f"Reason through the trade-offs between optimizing {metric} and protecting {entity} from {risk}.",
            ),
            "translation": (
                f"Translate a {artifact} for a {actor} into Spanish while preserving references to {entity}, {metric}, and {risk}.",
                f"Translate this operational notice about {risk} into French for stakeholders using {artifact}.",
            ),
            "summarization": (
                f"Summarize a detailed {artifact} for a {actor}, highlighting {metric}, {risk}, and impact on {entity}.",
                f"Condense a long review of {artifact} into the decisions, open questions, and risks around {entity}.",
            ),
            "creative_writing": (
                f"Write a creative scene about a {actor} confronting {risk} while studying a mysterious {artifact}.",
                f"Compose a reflective story in which {entity} becomes a symbol for changing {metric}.",
            ),
            "general": (
                f"Explain how a {artifact} affects {metric}, {risk}, and day-to-day decisions for a {actor}.",
                f"Describe the practical implications of {risk} for {entity} in a way a non-specialist can understand.",
            ),
            "mathematics": (
                f"Solve a quantitative problem where {metric} changes across three periods for {entity} and explain the calculation.",
                f"Compute the break-even threshold for a {artifact} when {risk} adds a variable cost to each {entity}.",
            ),
        }
        base = verbs[category][variant % 2]
        if difficulty == "hard":
            return base + " Include interacting constraints, uncertainty, and second-order effects."
        if difficulty == "medium":
            return base + " Include a realistic constraint and one exception case."
        return base

    def _length_sentence(self, prompt_length: str, difficulty: str) -> str:
        if prompt_length == "long":
            return "Use enough detail to support a robust answer, including context, assumptions, and acceptance criteria."
        if prompt_length == "medium":
            return "Provide enough context for a precise answer without requiring external documents."
        if difficulty == "hard":
            return "Keep the request concise but technically dense."
        return ""

    def _estimated_complexity(
        self,
        difficulty: str,
        expected_reasoning: str,
        constraint_count: int,
        prompt_length: str,
    ) -> float:
        difficulty_score = {"easy": 0.25, "medium": 0.55, "hard": 0.82}[difficulty]
        reasoning_score = {"low": 0.05, "medium": 0.12, "high": 0.20}[expected_reasoning]
        length_score = {"short": 0.0, "medium": 0.05, "long": 0.10}[prompt_length]
        constraint_score = min(0.18, constraint_count * 0.045)
        return round(min(1.0, difficulty_score + reasoning_score + length_score + constraint_score), 4)

