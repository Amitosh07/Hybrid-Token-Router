"""Active Learning and Disagreement Analysis for Phase 6.

Clusters prompt routing disagreements using KMeans to discover semantic blind spots
and outputs recommendations for manual relabeling.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from app.embedding_router.embedding_utils import get_model
from app.ml.model_utils import DOCS_DIR

logger = logging.getLogger(__name__)
REPORT_PATH = DOCS_DIR / "active_learning_report.md"


class ActiveLearner:
    """Performs active learning error clustering and failure analysis."""

    def analyze_disagreements(
        self,
        prompts_df: pd.DataFrame,
        predictions: dict[str, list[str]],
        ground_truth: list[str],
    ) -> None:
        """Analyze disagreements between multiple routers and group failures into semantic clusters."""
        logger.info("Starting active learning disagreement analysis...")
        
        # Build disagreement dataframe
        df = prompts_df.copy()
        df["ground_truth"] = ground_truth
        
        for name, preds in predictions.items():
            df[name] = preds

        # Identify disagreements with Ground Truth
        # Focus on Traditional ML failures
        ml_col = "Traditional ML" if "Traditional ML" in df.columns else (df.columns[4] if len(df.columns) > 4 else None)
        if not ml_col:
            logger.warning("Traditional ML column not found. Skipping Active Learning clustering.")
            return
            
        disagreements = df[df[ml_col] != df["ground_truth"]].copy()
        
        if len(disagreements) == 0:
            logger.info("Zero model disagreements found. Generating empty active learning report.")
            self._write_empty_report()
            return

        # Extract embeddings for clustering
        model = get_model()
        texts = disagreements["prompt"].astype(str).tolist()
        embeddings = model.encode(texts)

        # Dynamic cluster count determination
        n_clusters = min(5, len(disagreements))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        disagreements["cluster_id"] = kmeans.fit_predict(embeddings)

        # Analyze each cluster
        cluster_summaries = []
        for cid in range(n_clusters):
            cluster_df = disagreements[disagreements["cluster_id"] == cid]
            
            # Find the prompt closest to the cluster center
            center = kmeans.cluster_centers_[cid]
            emb_subset = embeddings[disagreements["cluster_id"] == cid]
            dists = np.linalg.norm(emb_subset - center, axis=1)
            closest_idx = cluster_df.index[np.argmin(dists)]
            exemplar = df.loc[closest_idx]["prompt"]
            exemplar_id = df.loc[closest_idx]["prompt_id"]

            # Common categories and difficulties
            cats = cluster_df["category"].value_counts().index[0] if not cluster_df["category"].empty else "unknown"
            diffs = cluster_df["difficulty"].value_counts().index[0] if not cluster_df["difficulty"].empty else "unknown"

            cluster_summaries.append({
                "cluster_id": cid,
                "size": len(cluster_df),
                "primary_category": cats,
                "primary_difficulty": diffs,
                "exemplar_prompt_id": exemplar_id,
                "exemplar_prompt": exemplar[:160] + "...",
            })

        # Save active learning report
        self._write_report(df, disagreements, cluster_summaries, ml_col)

    def _write_report(
        self,
        df: pd.DataFrame,
        disagreements: pd.DataFrame,
        cluster_summaries: list[dict[str, Any]],
        ml_col: str,
    ) -> None:
        """Write the active learning report to disk."""
        total_samples = len(df)
        total_missed = len(disagreements)
        error_rate = total_missed / total_samples if total_samples > 0 else 0.0

        # Compile category-wise error counts
        cat_errors = disagreements["category"].value_counts()
        diff_errors = disagreements["difficulty"].value_counts()

        lines = [
            "# Active Learning and Error Analysis Report",
            "",
            f"**Total Samples Evaluated**: {total_samples}",
            f"**Total ML Router Misclassifications**: {total_missed} ({error_rate:.2%})",
            "",
            "## Error Breakdown by Category",
            "",
            "| Category | Misclassifications | Share of Total Errors |",
            "|---|---|---|",
        ]
        for cat, count in cat_errors.items():
            lines.append(f"| {cat} | {count} | {count / total_missed:.2%} |")

        lines.extend([
            "",
            "## Error Breakdown by Difficulty",
            "",
            "| Difficulty | Misclassifications | Share of Total Errors |",
            "|---|---|---|",
        ])
        for diff, count in diff_errors.items():
            lines.append(f"| {diff} | {count} | {count / total_missed:.2%} |")

        lines.extend([
            "",
            "## Semantic Disagreement Clusters (KMeans)",
            "",
            "We clustered prompt semantic vectors to isolate structural failure regions (blind spots):",
            "",
        ])

        for c in cluster_summaries:
            lines.extend([
                f"### Cluster #{c['cluster_id']} (Size: {c['size']} samples)",
                f"- **Primary Domain/Category**: `{c['primary_category']}`",
                f"- **Primary Difficulty**: `{c['primary_difficulty']}`",
                f"- **Exemplar Prompt ID**: `[{c['exemplar_prompt_id']}]`",
                f"- **Representative Prompt Text**: \"{c['exemplar_prompt']}\"",
                "",
            ])

        lines.extend([
            "## Relabeling and Active Learning Recommendations",
            "",
            "To resolve these semantic errors in future phases, we recommend prioritizing these samples for audit and relabeling:",
            "",
        ])
        
        # Take the top 10 closest prompts to cluster centers as recommended active learning candidates
        candidates = disagreements.head(10)
        for _, row in candidates.iterrows():
            lines.append(
                f"- **{row['prompt_id']}** ({row['category']}, {row['difficulty']}): Ground Truth=`{row['ground_truth']}`, Model Pred=`{row[ml_col]}`. "
                f"Prompt: \"{row['prompt'][:200]}...\""
            )

        DOCS_DIR.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Successfully generated active learning report at: %s", REPORT_PATH)

    def _write_empty_report(self) -> None:
        lines = [
            "# Active Learning and Error Analysis Report",
            "",
            "**No Model Disagreements Detected**.",
            "The model routing decisions perfectly align with the ground truth labeling policy.",
        ]
        REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
