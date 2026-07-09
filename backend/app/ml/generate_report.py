import json
import pandas as pd
import numpy as np

# Load JSON results
with open("scratch_audit_results.json", "r") as f:
    res = json.load(f)

# Sort categories by total samples descending, then by name
cat_stats = sorted(res['cat_stats'], key=lambda x: (-x['Total'], x['Category']))
parent_stats = sorted(res['parent_stats'], key=lambda x: (-x['Total'], x['Parent Category']))

# Select parent categories for section 2
mostly_local_parents = [p for p in parent_stats if p['REMOTE %'] < 20.0]
mostly_remote_parents = [p for p in parent_stats if p['REMOTE %'] > 80.0]

# Build Category Distribution Section
cat_dist_md = ""
for c in cat_stats:
    cat_dist_md += f"""### {c['Category'].replace('_', ' ').title()}
Total: {c['Total']:,}
LOCAL: {c['LOCAL']:,}
REMOTE: {c['REMOTE']:,}
REMOTE Rate: {c['REMOTE %']:.2f}%
Average Length: {c['Avg Length']:.2f}
Average Complexity: {c['Avg Complexity']:.4f}
Average Reasoning Score: {c['Avg Reasoning']:.2f}

"""

# Build Feature Importance Table
imp_table = "| Rank | Feature | Random Forest Importance | Correlation with REMOTE | Description |\n"
imp_table += "|---|---|---|---|---|\n"
feature_descriptions = {
    "complexity_score": "Overall complexity score in [0, 1] mapped from semantic features.",
    "reasoning_score": "Integer reasoning score in [0, 10] (dominant routing signal).",
    "technical_complexity": "Heuristic technical complexity in [0, 1] based on domain keywords.",
    "api_keywords": "Count/presence of API-related engineering terms.",
    "task_complexity": "Cognitive task complexity score based on verb groupings.",
    "contains_code": "Boolean flag indicating code syntax or keywords.",
    "code_indicators": "Counts of specific coding keyword matches.",
    "constraint_density": "Ratio of constraints to prompt length.",
    "reasoning_depth": "Depth score of logical/analytical reasoning patterns.",
    "system_design_keywords": "Count of system architecture and design terms.",
    "algorithmic_complexity": "Presence of algorithm/complexity terms (e.g. Big-O).",
    "math_indicators": "Count of math keywords or LaTeX symbols.",
    "contains_math": "Boolean flag indicating mathematical symbols/keywords.",
    "context_complexity": "Lexical context load and concept density score.",
    "constraint_complexity": "Aggregate score for formatting/style constraints.",
    "word_count": "Number of words in the prompt.",
    "estimated_input_tokens": "Estimated token length based on char count.",
    "prompt_length": "Character count of the prompt.",
    "contains_numbers": "Presence of numeric digits.",
    "contains_question": "Presence of interrogative keywords or question marks.",
    "presence_of_tables": "Presence of Markdown or CSV table patterns.",
    "contains_json": "Boolean flag indicating JSON brackets or keys.",
    "contains_markdown": "Presence of Markdown styling tags.",
    "sql_indicators": "Presence of SQL query statements or keywords.",
    "multi_turn_context": "Flag for conversational turn history.",
    "dependency_between_subtasks": "Score for logical subtask dependencies."
}

# Create a mapping for quick lookup of correlation
corr_map = {c['Feature']: c['Correlation'] for c in res['correlations']}

for rank, imp in enumerate(res['importances'], 1):
    f_name = imp['Feature']
    corr_val = corr_map.get(f_name, 0.0)
    desc = feature_descriptions.get(f_name, "Metadata / Lexical feature.")
    imp_table += f"| {rank} | `{f_name}` | {imp['Importance']:.4f} | {corr_val:+.4f} | {desc} |\n"

# Mostly Local Parents Table (top 15)
local_parents_table = "| Parent Category | Total Samples | LOCAL | REMOTE | REMOTE % | Avg Reasoning |\n|---|---:|---:|---:|---:|---:|\n"
for p in mostly_local_parents[:20]:
    local_parents_table += f"| {p['Parent Category']} | {p['Total']} | {p['LOCAL']} | {p['REMOTE']} | {p['REMOTE %']:.1f}% | {p['Avg Reasoning']:.2f} |\n"

# Mostly Remote Parents Table (top 15)
remote_parents_table = "| Parent Category | Total Samples | LOCAL | REMOTE | REMOTE % | Avg Reasoning |\n|---|---:|---:|---:|---:|---:|\n"
for p in mostly_remote_parents[:20]:
    remote_parents_table += f"| {p['Parent Category']} | {p['Total']} | {p['LOCAL']} | {p['REMOTE']} | {p['REMOTE %']:.1f}% | {p['Avg Reasoning']:.2f} |\n"

# Format report template
report_md = f"""# Merged Dataset Audit Report
**Role**: Senior ML Engineer  
**Date**: July 9, 2026  
**Conversation ID**: 18afc6e4-41f8-422c-a989-13476094e741

---

## Executive Summary

This report presents a comprehensive dataset audit of the merged routing dataset containing **{res['total_samples']:,} prompts** (LOCAL: **{5850:,}** (39.0%), REMOTE: **{9150:,}** (61.0%)).

This audit analyzes the category distribution, parent category representation, prompt length buckets, predictors of routing, category imbalance, dataset quality, and training readiness.

> [!WARNING]
> **Audit Conclusion**: The dataset is **NOT suitable for production training** in its current state. The 61% REMOTE distribution is heavily distorted by artificial template suffixes in the imported dataset, domain keyword bias in the project data, a narrow prompt length distribution (all prompts are between 24 and 80 words), and rule-based false positives in the feature extractor.

---

## 1. Category Distribution

The dataset contains a mix of existing project data and an imported dataset. Below is the detailed category-by-category breakdown of the {res['total_samples']:,} prompts.

{cat_dist_md}

---

## 2. Parent Category Analysis

The imported dataset (`10k_chatbot_prompts`) contains `parent_category` and `subcategory` metadata. We analyzed these 9,970 samples separately to isolate their routing patterns.

### Overrepresentation Analysis
The imported dataset represents **66.47%** of the entire merged dataset. In terms of category frequency, there is a highly artificial flat distribution:
- Out of **998 unique parent categories**, 996 have exactly **10 samples** each, and 2 have **20 samples** each.
- Niche, obscure academic or hobby categories (e.g., *Dollmaking*, *Origami*, *Braising*, *Archaeozoology*) are collectively heavily overrepresented compared to real-world production distributions, which are typically dominated by coding, general conversation, planning, and basic Q&A.

### Mostly LOCAL Parent Categories
The following parent categories have the lowest REMOTE rates (mostly routed to LOCAL):

{local_parents_table}

*Interpretation*: Mostly LOCAL categories represent basic arts, crafts, simple language learning, culinary techniques, and daily life hobbies. These prompts have low technical complexity and average reasoning scores around 3.0, placing their routing scores below the threshold of 23.

### Mostly REMOTE Parent Categories
The following parent categories have the highest REMOTE rates (mostly routed to REMOTE):

{remote_parents_table}

*Interpretation*: Mostly REMOTE categories represent advanced computer science, cloud computing, mathematics, and software architecture. These prompts naturally contain dense technical terminology (e.g., *Kubernetes*, *API*, *Transformer*) and complex reasoning instructions, easily pushing them past the routing threshold of 23.

---

## 3. Prompt Length Analysis

Prompts were partitioned into the following length buckets:
- **0–100 words**
- **100–250 words**
- **250–500 words**
- **500+ words**

### Length Bucket Distribution
| Bucket | Total Samples | LOCAL Samples | REMOTE Samples | REMOTE % | Average Routing Score |
|---|---:|---:|---:|---:|---:|
| **0–100 words** | {res['total_samples']:,} | {5850:,} | {9150:,} | 61.0% | {res['bucket_stats'][0]['Avg Routing Score']:.2f} |
| **100–250 words** | 0 | 0 | 0 | 0.0% | N/A |
| **250–500 words** | 0 | 0 | 0 | 0.0% | N/A |
| **500+ words** | 0 | 0 | 0 | 0.0% | N/A |

### Length Bias Evaluation
> [!IMPORTANT]
> **Key Finding**: Prompt length is **NOT** dominating routing decisions, but it represents a **fatal data coverage gap**.
> 
> 1. **Zero Length Variance**: Every single one of the 15,000 prompts lies between **24 and 80 words** (mean = {res['avg_prompt_len']:.2f} words). There is zero representation of medium (100–250), long (250–500), or extra-long (500+) prompts.
> 2. **Correlation within Bucket**: Within the 0–100 range, `word_count` has a correlation of `+0.37` with the REMOTE label, and `estimated_input_tokens` has `+0.27`.
> 3. **Implication**: A router trained on this dataset will have no concept of prompts longer than 80 words. If deployed in production, it will likely exhibit extreme out-of-distribution instability when faced with typical long-context user queries (e.g., code repositories, documents to summarize).

---

## 4. Routing Decision Analysis

To identify what features drive the routing decisions (LOCAL vs. REMOTE) under the deterministic labeling policy, we trained a Random Forest classifier and calculated Pearson correlations.

### Feature Importance and Correlation Rankings
{imp_table}

### Key Predictors Analysis
1. **`complexity_score` and `reasoning_score` (Dominant)**: Together, these account for over **45%** of the feature importance. This is by design: `reasoning_score` contributes `5` points per level to the routing score. A reasoning score of 5 adds 25 points, immediately crossing the threshold of 23.
2. **`technical_complexity` and `api_keywords`**: Technical domain signals are heavily weighted (+12 points for technical complexity, +3 for API keywords).
3. **`word_count` and `estimated_input_tokens`**: Word count and token pressure are weak predictors (ranked 16th and 17th) because the length of prompts is constrained between 24 and 80 words.

---

## 5. Category Imbalance Detection

We detected severe routing anomalies across major categories, indicating unrealistic routing behavior:

### Anomalous Categories Table
| Category | Total Samples | LOCAL | REMOTE | REMOTE % | Avg Reasoning | Status |
|---|---:|---:|---:|---:|---:|---|
| **Reasoning** | 697 | 0 | 697 | 100.0% | 4.76 | 🚨 Unrealistic |
| **Planning** | 698 | 7 | 691 | 99.0% | 5.57 | 🚨 Unrealistic |
| **Translation** | 698 | 2 | 696 | 99.7% | 4.22 | 🚨 Unrealistic |
| **Mathematics** | 708 | 2 | 706 | 99.7% | 4.71 | 🚨 Unrealistic |
| **General** | 135 | 6 | 129 | 95.6% | 4.21 | 🚨 Unrealistic |
| **Coding** | 698 | 132 | 566 | 81.1% | 3.77 | 🚨 Unrealistic |

### Why these routing ratios are unrealistic:
- **Translation (99.7% REMOTE)**: Translation is a straightforward task that local models excel at. Routing 99.7% of translation queries to a remote model is economically and operationally absurd. It occurs because the translation prompts in the existing project data are synthetically stuffed with technical and clinical terminology (e.g., *"Translate this operational notice about incorrect dosage guidance into French..."*), which artificially inflates the technical complexity score.
- **Reasoning (100% REMOTE) and Planning (99% REMOTE)**: Simple planning or reasoning queries (e.g., making a basic shopping checklist or simple logical comparisons) can easily run locally. They are sent remote because of **instruction inflation** in the templates (e.g., forcing *"Discuss edge cases and failure modes"* or *"Make assumptions explicit for auditability"*), which adds heavy analytical reasoning scores.
- **Mathematics (99.7% REMOTE)**: As analyzed, the threshold is 23. A math prompt gets a +4 task type nudge and a +4 `contains_math` bonus. If its reasoning score is 3 or more (which is the case for 100% of the mathematics prompts in this dataset), its routing score is at least `(3 * 5) + 4 + 4 = 23`. Thus, **every math prompt with reasoning score >= 3 is routed REMOTE**, regardless of how simple the math is (e.g., simple algebra or geometry).

---

## 6. Dataset Quality Audit

A thorough audit of the dataset's quality reveals the following statistics:

- **Duplicate Rate**: **{res['exact_duplicates_rate']:.4f}%** (0 exact duplicates). The initial deduplication step successfully removed exact string duplicates.
- **Near-Duplicate Rate (Global)**: **{res['near_duplicates_rate']:.4f}%** ({res['near_duplicates_count']} pairs found). 
  *Note*: The original dataset preparation script used a local window of 1,500 elements for Jaccard deduplication. Our global audit revealed **{res['near_duplicates_count']} near-duplicate pairs** across the entire 15,000 dataset that slipped past the local deduplication window.
- **Empty Prompts**: **0** (no null, empty, or whitespace-only prompts).
- **Missing Metadata**: **0** (no missing cells or NaNs in the feature columns).
- **Average Prompt Diversity (TTR)**: **{res['ttr']:.4f}** (Type-Token Ratio). This TTR is exceptionally low, indicating extreme vocabulary homogeneity. This is a direct consequence of prompts being synthetically generated via repetitive templates.

### Class Overlap & Label Inconsistency Case Study
Out of the {res['near_duplicates_count']} near-duplicate pairs, we found **{res['inconsistent_pairs_count']} pair** with conflicting routing labels (one LOCAL, one REMOTE).

#### Label Inconsistency Example:
- **Prompt 1 (LOCAL, Routing Score: 19)**: 
  `"Compose a reflective story in which district becomes a symbol for changing processing time. Include practical examples, important trade-offs, and implementation guidance where relevant."`
- **Prompt 2 (REMOTE, Routing Score: 24)**: 
  `"Compose a reflective story in which public dataset becomes a symbol for changing processing time. Include practical examples, important trade-offs, and implementation guidance where relevant."`

#### Root Cause Analysis:
The only lexical difference is the substitution of *"district"* with *"public dataset"*. 
In the feature extractor, the word **`public`** matches Java/C++ access modifier syntax and triggers the `contains_code` flag. This adds **5 points** to the routing score (pushing P2 from 19 to 24), crossing the threshold of 23 and routing P2 to REMOTE. 
This demonstrates that the rule-based labeling pipeline is fragile and introduces labels that are inconsistent and noisy.

---

## 7. Training Readiness

### Is this dataset suitable for production training?
> [!CAUTION]
> **NO**

### Key Reasons for Unsuitability:
1. **Severe Label Bias**: 90.6% of the project-specific data is labeled REMOTE. Training a model on this will result in a model that routes almost everything to REMOTE, destroying the latency/cost savings of a hybrid router.
2. **Artificial Suffix Inflation**: 100% of the imported dataset (representing 66% of the merged dataset) contains the exact same template suffix: `". Include practical examples, important trade-offs, and implementation guidance where relevant."` This artificially adds +15 points to the routing score of all 10,000 chatbot prompts.
3. **No Length Diversity**: The dataset contains zero prompts longer than 80 words. The model will fail to generalize to typical long production inputs.
4. **Keyword Fragility**: The deterministic labeling policy has built-in fragility (e.g., the word "public" triggering a code flag), causing near-identical prompts to get conflicting labels.

---

## 8. Recommendations

To make this dataset suitable for production training, we recommend the following minimal corrections:

1. **Remove Template Suffixes**: Strip the synthetic trailing suffix (*"Include practical examples..."*) from the imported dataset before feature extraction. This will restore the true, uninflated complexity scores of the baseline prompts.
2. **Introduce Long-Context Prompts**: Inject a small, curated set (e.g., 2,000–3,000 prompts) of actual long-context user queries (summarizations, documents, codebases) ranging from 100 to 2,000+ words to ensure length coverage.
3. **Adjust Math & Translation Routing Thresholds**: Mutate the task-type weights or threshold in `RouterConfig` so that simple math (Algebra/Geometry) and standard translation tasks do not unconditionally trigger REMOTE routing.
4. **Refine Code Detection Rules**: Update `_RE_CODE_PATTERN` in the feature extractor to require structural context for keywords like `public` (e.g., `public class` or `public static`) to prevent common English nouns from falsely triggering the code flag.
5. **Run Global Deduplication**: Perform Jaccard deduplication globally over the entire merged dataset, rather than in a sliding window of 1,500, to remove the remaining {res['near_duplicates_count']} near-duplicates.

---

### Conclusion

"The following issues should be corrected before training."
"""

# Write report to workspace root
report_path = "../dataset_audit_report.md"
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report_md)

print(f"Report successfully written to {report_path}")
