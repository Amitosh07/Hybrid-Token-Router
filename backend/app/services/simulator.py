"""High-Fidelity LLM Simulator for Development and Offline Testing.

Simulates response text, token counts, and execution latency matching the
complexity and category of the user prompt.
"""

from __future__ import annotations

import asyncio
import random
from typing import Final

from app.services.feature_extractor import extract_features

# Sample sentences for generating text length variations
_SAMPLES: Final[dict[str, list[str]]] = {
    "coding": [
        "Defines the class structure and manages interface state.",
        "Implements a binary search algorithm to achieve logarithmic time complexity O(log N).",
        "Checks for key existence in the hash map before proceeding with database commits.",
        "Establishes a secure connection pool to handle concurrency safely under heavy load.",
        "Refactors the database access object to utilize transactions and snapshot isolation."
    ],
    "mathematics": [
        "Integrates the function over the real bounds to evaluate convergence.",
        "Calculates the expected value using Bayes' theorem with the given prior distribution.",
        "Solves the system of equations by performing row reduction and verifying the determinant.",
        "Proves the base case and applies mathematical induction to show the theorem holds for all integers.",
        "Applies the Fourier transform to extract spectral coefficients from the noise signal."
    ],
    "planning": [
        "Establishes milestones and maps week-by-week progress for the engineering team.",
        "Outlines risk mitigation plans and disaster recovery policies for infrastructure.",
        "Structures the rollout phase with clear KPIs and stakeholder accountability metrics.",
        "Defines the hiring plan and onboarding schedule for technical lead roles.",
        "Coordinates the go-to-market strategy, aligning marketing campaigns with feature release dates."
    ],
    "translation": [
        "Traduit le document technique en conservant la terminologie exacte.",
        "Translates the business agreement ensuring compliance with local jurisdiction regulations.",
        "Übersetzt die Fehlermeldungen und passt die Formatierung an die Richtlinien an.",
        "Traduccion del contrato de licencia con atencion especial a las clausulas de responsabilidad.",
        "Translates the user manual while maintaining the original tone and layout constraints."
    ],
    "summarization": [
        "Summarizes the meeting transcript highlight key decisions and follow-up assignments.",
        "Provides a TL;DR highlighting the main findings of the clinical trial data.",
        "Condenses the 40-page contract down to three core liability and compliance clauses.",
        "Recaps the academic abstract, presenting the methodology and statistical significance.",
        "Extracts the key points from the regulatory filing regarding financial governance."
    ],
    "creative_writing": [
        "Crafts a compelling opening scene for the screenplay set in a dystopian control room.",
        "Composes a sonnet exploring the tension between technological advancement and human connection.",
        "Invokes an unreliable narrator to tell the story of a lost cache of ancient data.",
        "Drafts dialogue demonstrating the conflict between the protagonist and the security AI.",
        "Generates a descriptive story detailing the atmospheric conditions on a foreign planet."
    ],
    "general": [
        "Provides a comprehensive overview of the historical context of industrial regulations.",
        "Explains the core principles of utilitarianism compared to deontological ethics.",
        "Describes the policy changes regarding data privacy standards across regions.",
        "Drafts a professional email responding to the client's questions about migration costs.",
        "Analyzes the economic consequences of market fluctuations on consumer behavior."
    ]
}


async def generate_simulated(prompt: str, provider_name: str) -> str:
    """Generate a realistic mock LLM response with simulated processing latency.

    Args:
        prompt:        Raw input prompt.
        provider_name: "local" (Qwen 3B) or "remote" (Groq Llama 8B).

    Returns:
        A simulated text response.
    """
    # 1. Inspect prompt features on the fly
    features = extract_features(prompt)
    complexity = features["complexity"]
    task_type = features["task_type"]

    # Normalize task type to key in samples
    cat_key = task_type if task_type in _SAMPLES else "general"
    samples = _SAMPLES[cat_key]

    # 2. Simulate quality, response length, and latencies based on difficulty
    # Remote (Llama 8B) performs better on hard tasks than Local (Qwen 3B)
    if complexity == "easy":
        # High quality for both, fast response
        local_words_cnt = random.randint(35, 75)
        remote_words_cnt = random.randint(45, 95)
        latency_ms = random.randint(120, 320) if provider_name == "local" else random.randint(220, 500)
    elif complexity == "medium":
        # Moderate difference
        local_words_cnt = random.randint(65, 130)
        remote_words_cnt = random.randint(110, 220)
        latency_ms = random.randint(220, 520) if provider_name == "local" else random.randint(450, 950)
    else:  # hard
        # Large difference: local struggles, remote excels
        local_words_cnt = random.randint(50, 100)      # shorter/incomplete response
        remote_words_cnt = random.randint(220, 420)     # detailed comprehensive response
        latency_ms = random.randint(320, 720) if provider_name == "local" else random.randint(750, 1650)

    # 3. Simulate processing time
    await asyncio.sleep(latency_ms / 1000.0)

    # 4. Generate structured content
    # Extract terms from the prompt to ensure realistic coverage metric score
    prompt_words = [w for w in prompt.split() if len(w) > 4][:12]
    matched_terms_str = ", ".join(prompt_words) if prompt_words else "context, information, requirements"

    target_words = local_words_cnt if provider_name == "local" else remote_words_cnt
    
    # Build base template
    if task_type == "coding":
        code_block = (
            "```python\n"
            "# Simulated implementation\n"
            "def handle_request(data):\n"
            "    # Code body\n"
            "    return True\n"
            "```"
        )
        response = f"Here is the code solution addressing: {prompt[:40]}...\n\n{code_block}\n\nKey parameters: {matched_terms_str}."
    elif task_type == "mathematics":
        response = f"Let's solve the math question step by step. We evaluate the input: '{prompt[:45]}'.\n\nResult = 42. Verified terms: {matched_terms_str}."
    elif task_type == "planning":
        response = (
            f"Here is the action roadmap for '{prompt[:45]}':\n"
            "  1. Assessment and scope planning\n"
            "  2. Implementation of target goals\n"
            "  3. Verification and rollout testing\n"
            f"Matched terms: {matched_terms_str}."
        )
    else:
        # Default text
        response = f"Detailed response addressing the prompt topic: '{prompt[:45]}'. Terms: {matched_terms_str}."

    # Pad response to hit the word count target
    current_words = response.split()
    if len(current_words) < target_words:
        pad_sentences = []
        while len(current_words) + len(pad_sentences) * 8 < target_words:
            pad_sentences.append(random.choice(samples))
        response += " " + " ".join(pad_sentences)

    return response
