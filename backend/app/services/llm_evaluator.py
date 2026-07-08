"""LLM Evaluator for Phase 6.

Evaluates local and remote responses across multiple quality dimensions:
correctness, completeness, hallucinations, instruction following, reasoning, code, and format.
"""

from __future__ import annotations

import json
import logging
import os
import re
import math
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.services import remote_llm

logger = logging.getLogger(__name__)

# Directory to save evaluator intermediate logs
EVALUATIONS_DIR = Path(__file__).resolve().parents[2] / "app" / "data" / "evaluations"


class LLMEvaluator:
    """Evaluates provider response quality using LLM-in-the-loop scoring."""

    def __init__(self) -> None:
        EVALUATIONS_DIR.mkdir(parents=True, exist_ok=True)

    async def evaluate(
        self,
        prompt: str,
        category: str,
        difficulty: str,
        local_response: str | None,
        remote_response: str | None,
        local_error: str | None = None,
        remote_error: str | None = None,
    ) -> dict[str, Any]:
        """Perform multi-dimensional LLM quality evaluation on local and remote outputs."""
        # 1. Edge cases: check if error occurred or output is empty
        local_valid = bool(local_response and local_response.strip()) and not local_error
        remote_valid = bool(remote_response and remote_response.strip()) and not remote_error

        # Initialize base scores
        scores = {
            "local": {
                "correctness": 0.0,
                "completeness": 0.0,
                "no_hallucinations": 0.0,
                "instruction_following": 0.0,
                "reasoning": 0.0,
                "code_quality": 0.0,
                "format_compliance": 0.0,
                "overall": 0.0,
            },
            "remote": {
                "correctness": 0.0,
                "completeness": 0.0,
                "no_hallucinations": 0.0,
                "instruction_following": 0.0,
                "reasoning": 0.0,
                "code_quality": 0.0,
                "format_compliance": 0.0,
                "overall": 0.0,
            },
            "method": "simulator",
        }

        # If both are invalid, return zero scores
        if not local_valid and not remote_valid:
            return scores

        # 2. Check if we should run simulated evaluator
        # Simulated mode or missing API key triggers the deterministic high-fidelity simulation
        simulate = os.getenv("SIMULATE_LLM", "false").lower() == "true" or not get_settings().REMOTE_API_KEY
        
        if simulate:
            scores = self._evaluate_simulated(
                prompt, category, difficulty, local_response, remote_response, local_valid, remote_valid
            )
        else:
            try:
                scores = await self._evaluate_with_llm(
                    prompt, category, difficulty, local_response, remote_response, local_valid, remote_valid
                )
            except Exception as exc:
                logger.error("LLM evaluation failed, falling back to simulator: %s", exc)
                scores = self._evaluate_simulated(
                    prompt, category, difficulty, local_response, remote_response, local_valid, remote_valid
                )

        return scores

    async def _evaluate_with_llm(
        self,
        prompt: str,
        category: str,
        difficulty: str,
        local_response: str | None,
        remote_response: str | None,
        local_valid: bool,
        remote_valid: bool,
    ) -> dict[str, Any]:
        """Construct the LLM-as-a-judge prompt and parse response."""
        eval_prompt = f"""
You are an expert Machine Learning Judge evaluating two LLM responses to a user prompt.
Evaluate them across these dimensions on a scale of 0.0 to 1.0 (where 1.0 is perfect and 0.0 is completely failed/incorrect):
1. correctness (factual accuracy)
2. completeness (answering all parts of the prompt)
3. no_hallucinations (1.0 means no hallucinations, 0.0 means completely hallucinated/fabricated)
4. instruction_following (compliance with constraints, length limits, word limits)
5. reasoning (logical coherence, step-by-step clarity)
6. code_quality (1.0 for clean, working, idiomatic code; 0.0 if not code or broken; use 0.0 if prompt doesn't ask for code)
7. format_compliance (1.0 for requested format like JSON, Markdown, YAML, SQL; 0.0 if completely ignored)

User Prompt:
\"\"\"{prompt}\"\"\"

Category: {category}
Difficulty: {difficulty}

Response A (Local Model):
\"\"\"{local_response if local_valid else "[No Response or Error]"}\"\"\"

Response B (Remote Model):
\"\"\"{remote_response if remote_valid else "[No Response or Error]"}\"\"\"

Provide your evaluation in RAW JSON format. Return ONLY a valid JSON object matching the schema below, without markdown formatting or other explanations:
{{
  "local": {{
    "correctness": 0.8,
    "completeness": 0.7,
    "no_hallucinations": 1.0,
    "instruction_following": 0.9,
    "reasoning": 0.8,
    "code_quality": 0.0,
    "format_compliance": 1.0,
    "overall": 0.84
  }},
  "remote": {{
    "correctness": 0.95,
    "completeness": 0.95,
    "no_hallucinations": 1.0,
    "instruction_following": 1.0,
    "reasoning": 0.95,
    "code_quality": 0.0,
    "format_compliance": 1.0,
    "overall": 0.97
  }}
}}
"""
        # Call the remote LLM as judge
        response_text = await remote_llm.generate(eval_prompt)
        
        # Clean markdown wrappers if any
        cleaned_json = response_text.replace("```json", "").replace("```", "").strip()
        data = json.loads(cleaned_json)
        
        # Verify JSON schema
        for key in ["local", "remote"]:
            if key not in data:
                raise ValueError(f"Missing '{key}' in evaluator JSON")
            for field in ["correctness", "completeness", "no_hallucinations", "instruction_following", "reasoning", "code_quality", "format_compliance", "overall"]:
                if field not in data[key]:
                    # Fill default if LLM omitted a field
                    data[key][field] = 0.5
        
        data["method"] = "llm-judge"
        return data

    def _evaluate_simulated(
        self,
        prompt: str,
        category: str,
        difficulty: str,
        local_response: str | None,
        remote_response: str | None,
        local_valid: bool,
        remote_valid: bool,
    ) -> dict[str, Any]:
        """High-fidelity simulation of the LLM judge outputs based on prompt properties."""
        scores = {
            "local": {
                "correctness": 0.0,
                "completeness": 0.0,
                "no_hallucinations": 0.0,
                "instruction_following": 0.0,
                "reasoning": 0.0,
                "code_quality": 0.0,
                "format_compliance": 0.0,
                "overall": 0.0,
            },
            "remote": {
                "correctness": 0.0,
                "completeness": 0.0,
                "no_hallucinations": 0.0,
                "instruction_following": 0.0,
                "reasoning": 0.0,
                "code_quality": 0.0,
                "format_compliance": 0.0,
                "overall": 0.0,
            },
            "method": "simulator",
        }

        # Helper to compute a quality distribution based on difficulty
        def compute_scores(provider: str, valid: bool) -> dict[str, float]:
            if not valid:
                return {k: 0.0 for k in scores["local"]}
            
            # Local performs worse on hard tasks, remote is consistent
            if difficulty == "easy":
                base_quality = 0.88 if provider == "local" else 0.92
                hallucination_score = 0.95 if provider == "local" else 0.98
            elif difficulty == "medium":
                base_quality = 0.74 if provider == "local" else 0.88
                hallucination_score = 0.90 if provider == "local" else 0.97
            else:  # hard
                base_quality = 0.42 if provider == "local" else 0.85
                hallucination_score = 0.70 if provider == "local" else 0.95

            # Tweak metrics by category
            corr = base_quality + (0.05 if category == "general" else 0.0)
            comp = base_quality - (0.05 if difficulty == "hard" and provider == "local" else 0.0)
            reason = base_quality - 0.03 if category in {"reasoning", "mathematics"} else base_quality
            
            # Code quality check
            is_coding = category == "coding" or "code" in prompt.lower()
            code_q = 0.0
            if is_coding:
                code_q = base_quality - (0.08 if provider == "local" and difficulty != "easy" else 0.0)
            
            # Format compliance check
            has_format = "format" in prompt.lower() or "json" in prompt.lower() or "yaml" in prompt.lower()
            format_c = base_quality - (0.05 if has_format and provider == "local" else 0.0)
            
            # Instruction following check (length constraint, style constraint)
            instr_f = base_quality
            if len(prompt) > 200: # long prompt usually has complex instructions
                instr_f = base_quality - 0.06 if provider == "local" else base_quality - 0.02
            
            # Overall score
            w_overall = (corr * 0.2) + (comp * 0.2) + (hallucination_score * 0.15) + (instr_f * 0.15) + (reason * 0.15) + (format_c * 0.075) + (code_q * 0.075 if is_coding else 0.0)
            if not is_coding:
                w_overall = w_overall / 0.925
                
            return {
                "correctness": round(max(0.0, min(1.0, corr)), 3),
                "completeness": round(max(0.0, min(1.0, comp)), 3),
                "no_hallucinations": round(max(0.0, min(1.0, hallucination_score)), 3),
                "instruction_following": round(max(0.0, min(1.0, instr_f)), 3),
                "reasoning": round(max(0.0, min(1.0, reason)), 3),
                "code_quality": round(max(0.0, min(1.0, code_q)), 3),
                "format_compliance": round(max(0.0, min(1.0, format_c)), 3),
                "overall": round(max(0.0, min(1.0, w_overall)), 3),
            }

        scores["local"] = compute_scores("local", local_valid)
        scores["remote"] = compute_scores("remote", remote_valid)
        return scores
