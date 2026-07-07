"""Output format library defining structural formats for generated prompts."""

from __future__ import annotations

from dataclasses import dataclass

OUTPUT_FORMATS: tuple[str, ...] = (
    "plain paragraphs",
    "numbered steps",
    "bullet list",
    "markdown table",
    "JSON object",
    "YAML block",
    "checklist",
    "short executive brief",
    "annotated code block",
    "comparison matrix",
)


@dataclass(frozen=True)
class FormatProfile:
    """Detailed profile for a specific output structure."""

    name: str
    description: str
    mime_type: str
    structured: bool


FORMAT_PROFILES: dict[str, FormatProfile] = {
    "plain paragraphs": FormatProfile(
        name="plain paragraphs",
        description="Standard narrative text formatted as descriptive paragraphs.",
        mime_type="text/plain",
        structured=False,
    ),
    "numbered steps": FormatProfile(
        name="numbered steps",
        description="A list of sequential steps numbered starting from 1.",
        mime_type="text/plain",
        structured=True,
    ),
    "bullet list": FormatProfile(
        name="bullet list",
        description="A bulleted list of items.",
        mime_type="text/plain",
        structured=True,
    ),
    "markdown table": FormatProfile(
        name="markdown table",
        description="A GitHub-Flavored Markdown table representation.",
        mime_type="text/markdown",
        structured=True,
    ),
    "JSON object": FormatProfile(
        name="JSON object",
        description="A parsable, valid JSON schema block.",
        mime_type="application/json",
        structured=True,
    ),
    "YAML block": FormatProfile(
        name="YAML block",
        description="A valid YAML document.",
        mime_type="application/yaml",
        structured=True,
    ),
    "checklist": FormatProfile(
        name="checklist",
        description="A checklist formatted with [ ] or [x] checkboxes.",
        mime_type="text/markdown",
        structured=True,
    ),
    "short executive brief": FormatProfile(
        name="short executive brief",
        description="A concise executive briefing covering key points.",
        mime_type="text/plain",
        structured=False,
    ),
    "annotated code block": FormatProfile(
        name="annotated code block",
        description="A code block with comments or annotations.",
        mime_type="text/plain",
        structured=True,
    ),
    "comparison matrix": FormatProfile(
        name="comparison matrix",
        description="A comparative matrix grid detailing trade-offs.",
        mime_type="text/markdown",
        structured=True,
    ),
}


def get_format_profile(format_name: str) -> FormatProfile:
    """Retrieve format details by name."""
    if format_name not in FORMAT_PROFILES:
        raise ValueError(f"Unsupported format profile: {format_name}")
    return FORMAT_PROFILES[format_name]
