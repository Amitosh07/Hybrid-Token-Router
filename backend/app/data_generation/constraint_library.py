"""Constraint, output-format, reasoning-depth, and length libraries."""

from __future__ import annotations

from dataclasses import dataclass


from app.data_generation.output_format_library import OUTPUT_FORMATS
from app.data_generation.reasoning_library import REASONING_DEPTHS

PROMPT_LENGTHS: tuple[str, ...] = ("short", "medium", "long")


@dataclass(frozen=True)
class Constraint:
    """A reusable prompt constraint."""

    name: str
    text: str
    applies_to: tuple[str, ...] = ()


CONSTRAINTS: tuple[Constraint, ...] = (
    Constraint("word_limit", "Keep the answer under {limit} words."),
    Constraint("include_examples", "Include at least {count} concrete examples."),
    Constraint("edge_cases", "Discuss edge cases and failure modes."),
    Constraint("no_jargon", "Avoid unexplained jargon."),
    Constraint("auditability", "Make assumptions explicit for auditability."),
    Constraint("risk_tradeoffs", "Compare risks, trade-offs, and mitigations."),
    Constraint("schema", "Return the core result using the requested schema."),
    Constraint("citations_placeholder", "Mark any facts that would need external verification."),
    Constraint("accessibility", "Use accessible language for a mixed-expertise audience."),
    Constraint("security", "Address security and privacy implications."),
    Constraint("latency", "Optimize for low-latency operational use."),
    Constraint("testing", "Include a minimal validation or testing approach.", ("coding",)),
    Constraint("complexity", "State time and space complexity where applicable.", ("coding", "mathematics")),
    Constraint("step_by_step", "Show the reasoning steps without skipping assumptions.", ("reasoning", "mathematics")),
    Constraint("tone", "Maintain a professional, neutral tone."),
    Constraint("locale", "Preserve local conventions, units, and formatting.", ("translation",)),
    Constraint("faithfulness", "Do not add facts that are not present in the source material.", ("translation", "summarization")),
    Constraint("rhyme_meter", "Use a consistent rhythm or structural pattern.", ("creative_writing",)),
    Constraint("stakeholders", "Address impacts on at least three stakeholder groups.", ("planning", "reasoning")),
    Constraint("rollback", "Include rollback or contingency steps.", ("planning", "cloud", "cybersecurity")),
)


def constraints_for(category: str, start: int, count: int) -> list[Constraint]:
    """Select deterministic constraints for a category."""
    eligible = [c for c in CONSTRAINTS if not c.applies_to or category in c.applies_to]
    selected: list[Constraint] = []
    for offset in range(count):
        selected.append(eligible[(start + offset) % len(eligible)])
    return selected


def render_constraint(constraint: Constraint, variant: int) -> str:
    """Render a constraint with deterministic parameter variation."""
    return constraint.text.format(
        limit=(80, 120, 180, 250)[variant % 4],
        count=(1, 2, 3, 4)[variant % 4],
    )

