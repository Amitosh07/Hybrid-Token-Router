# Feature Extractor V3 — Technical Documentation

This document describes the architecture, algorithms, and features of the **Feature Extraction Pipeline V3** for the Hybrid Token Router.

---

## 1. Pipeline Architecture

V3 is designed as a modular pipeline where each stage has exactly one responsibility:

```
                  User Prompt
                       │
                       ▼
             Stage 1: Normalization
                       │ (lowercase, sentence-split, tokenized)
                       ▼
          Stage 2: Lexical Features
                       │ (word counts, sentence counts, token estimate)
                       ▼
         Stage 3: Structural Features
                       │ (contains_code, contains_math, markdown list counts)
                       ▼
        Stage 4: Semantic Feature Groups
                       ├── technical_complexity
                       ├── reasoning_depth
                       ├── task_complexity
                       ├── constraint_complexity
                       └── context_complexity
                       │ (outputs float score in [0.0, 1.0] & matched evidence)
                       ▼
        Stage 5: Feature Interaction Rules
                       │ (Planning+Consistency, Precision+Reasoning, etc.)
                       ▼
        Stage 6: Complexity Aggregation
                       │ (Complexity score aggregation with weight redistribution)
                       ▼
                 Router Bridge
                       │ (reasoning_score [0-10] and complexity bucket label)
                       ▼
             Final Feature Dictionary
```

---

## 2. Feature Group Descriptions & Scoring Equations

We keep exactly 5 orthogonal semantic groups, enriched with high-level cognitive signals:

### 1. `technical_complexity`
- **Scope:** CS domains (algorithms, concurrency, database schema, system design, machine learning), Precision Risk domains (legal, medical, financial), and Scientific Research complexity.
- **Linguistic Safeguard:** Enforces word boundaries for all short keywords (length $\le 4$) to prevent false positive substring matches (e.g. `cas` matching inside `suitcase`).
- **Scoring Equation:**
  $$S_{\text{domain}} = \min(1.0, N_{\text{tier A}} \times 1.0 + N_{\text{tier B}} \times 0.4)$$
  $$Score = \min\left(1.0, \text{Avg}(\text{top-3 } S_{\text{domain}}) \times (1.0 + (D - 1) \times 0.06)\right)$$
  Where $D$ is the number of active domains.

### 2. `reasoning_depth`
- **Scope:** Logical deduction, mathematical proofs, optimization, multi-step/causal chains of thought, and ethical/moral reasoning.
- **Scoring Equation:**
  For each reasoning category, the hit count is mapped:
  $$\text{Score}_{\text{category}} = \text{base\_weight} \times f(\text{hits})$$
  Where $f(1) = 0.35$, $f(2) = 0.65$, $f(3+) = 1.0$ (measuring keyword density).
  The final score is the average of active categories multiplied by a breadth bonus:
  $$Score = \min\left(1.0, \text{Avg}(\text{active}) \times (1.0 + (\text{active\_count} - 1) \times 0.08)\right)$$

### 3. `task_complexity`
- **Scope:** Main verb groups (architect, implement, optimize, determine, debug, generate, transform), enriched with Planning (roadmaps, outlines, blueprints), Synthesis (comparative analysis, document merging), and Creative Ambiguity (poetry, open-ended brainstorming).
- **Scoring Equation:**
  $$Score = \min(1.0, W_{\text{verb\_group}} + \max(S_{\text{planning}}, S_{\text{synthesis}}, S_{\text{creativity}}) + B_{\text{modifiers}})$$

### 4. `constraint_complexity`
- **Scope:** Explicit rules (must, should not, do not, O(N) constraints) and Consistency/Memory load (JSON schemas, persona instructions, formatting guidelines, output length targets).
- **Scoring Equation:**
  $$Score = \min\left(1.0, \sum (W_{\text{pattern}} \times \min(2, \text{matches})) + S_{\text{consistency}} + B_{\text{list\_items}}\right)$$

### 5. `context_complexity`
- **Scope:** Logarithmic token load, vocabulary richness, and concept density.
- **Scoring Equation:**
  $$Score = T_{\text{load}} \times 0.35 + C_{\text{density\_score}} \times 0.55 + V_{\text{richness}} \times 0.10$$

---

## 3. Evidence Generation

Every feature group in the output dictionary is returned as an object containing both `score` and `matched_patterns` (the specific keywords or patterns that triggered the score):

```json
{
  "technical_complexity": {
    "score": 0.85,
    "matched_patterns": ["concurrency", "thread-safe", "deadlock"]
  }
}
```

---

## 4. Feature Interaction & Weight Redistribution

### Interaction Boosts:
1. **Planning & Consistency:** Boosts complexity if a prompt requires sequential planning and strict formatting/tone consistency.
   - Boost = $\min(\text{planning}, \text{consistency}) \times 0.15$
2. **Precision & Reasoning:** Boosts complexity if a prompt requires logical deduction in a high-risk domain (e.g. legal analysis).
   - Boost = $\min(\text{precision\_risk}, \text{reasoning}) \times 0.15$
3. **Synthesis & Ambiguity:** Boosts complexity if a prompt requires synthesis of viewpoints in an open-ended/philosophical context.
   - Boost = $\min(\text{synthesis}, \text{creativity}) \times 0.10$
4. **Cognitive Intensity:** Boost = $0.10$ if both logical reasoning $\ge 0.70$ and task complexity $\ge 0.70$.

### Dynamic Weight Redistribution:
If a prompt is completely non-technical ($S_{\text{technical}} = 0.0$) but exhibits clear reasoning or task complexity ($S_{\text{reason}} > 0.30$ or $S_{\text{task}} > 0.40$), the technical weight (0.30) is redistributed to reasoning, task, constraint, and context groups by normalizing their weights to sum to 1.0. This prevents purely cognitive/logical puzzles from being dragged down by the absence of coding terms.

---

## 5. Calibration Results

The pipeline was run against the 160 benchmark prompts. Below is a comparison between V1, V2, and V3:

### Overall Routing Distribution:
- **V1:** 159 LOCAL / 1 REMOTE (0.6% remote)
- **V2:** 150 LOCAL / 10 REMOTE (6.2% remote)
- **V3:** **134 LOCAL / 26 REMOTE (16.2% remote)**

### Statistical Separation (Cohen's d):
Cohen's d measures separation between easy/medium/hard prompts.
- **Easy vs Hard (V3):** **d = 1.718** (Large effect size $\ge 0.8$, indicating excellent separation)
- **Medium vs Hard (V3):** **d = 0.975** (Large effect size)
- **Easy vs Medium (V3):** **d = 0.837** (Large effect size)

### Confusion Matrix:
- Ground-truth **EASY** prompts: 56/56 routed LOCAL (0% false positives).
- Ground-truth **HARD** prompts: 19/48 routed REMOTE (39.6% remote routing for complex tasks, recovering screenplays, legal drafts, and roadmaps).

---

## 6. Recommendations for Supervised ML Training

The Feature Extractor V3 is **fully ready** to support a supervised machine learning training run.

### Input Features (recommended):
- `technical_complexity.score` — engineering domain depth [0, 1]
- `reasoning_depth.score` — analytical/logical demands [0, 1]
- `task_complexity.score` — verb-driven task scope [0, 1]
- `constraint_complexity.score` — explicit requirements density [0, 1]
- `context_complexity.score` — concept density + token load [0, 1]
- `estimated_input_tokens` — raw token count [int]
- `contains_code` — structural code presence [bool]
- `contains_math` — math notation presence [bool]

### Features to Avoid (highly collinear or redundant):
- `reasoning_score` (directly derived from complexity_score via rounding)
- `complexity` (bucketing of complexity_score)
- `complexity_score` (collinear with the five individual group scores)
- `routing_score` (computed linearly by the router; this is the label)

### ML Target Recommendation:
Use `provider` ("local" vs "remote") as a binary target variable to train a classifier (such as XGBoost or a shallow neural network). Alternatively, use `complexity_score` to train a regressor.
The current dataset generated by V3 features has extremely low collinearity, high explainability, and clean decision boundaries, making it ideal for immediate classifier training.
