# Active Learning and Error Analysis Report

**Total Samples Evaluated**: 400
**Total ML Router Misclassifications**: 65 (16.25%)

## Error Breakdown by Category

| Category | Misclassifications | Share of Total Errors |
|---|---|---|
| coding | 22 | 33.85% |
| summarization | 20 | 30.77% |
| creative_writing | 10 | 15.38% |
| translation | 7 | 10.77% |
| planning | 2 | 3.08% |
| mathematics | 2 | 3.08% |
| reasoning | 1 | 1.54% |
| general | 1 | 1.54% |

## Error Breakdown by Difficulty

| Difficulty | Misclassifications | Share of Total Errors |
|---|---|---|
| easy | 35 | 53.85% |
| medium | 19 | 29.23% |
| hard | 11 | 16.92% |

## Semantic Disagreement Clusters (KMeans)

We clustered prompt semantic vectors to isolate structural failure regions (blind spots):

### Cluster #0 (Size: 7 samples)
- **Primary Domain/Category**: `translation`
- **Primary Difficulty**: `easy`
- **Exemplar Prompt ID**: `[gen_001804]`
- **Representative Prompt Text**: "Translate this operational notice about incorrect dosage guidance into French for stakeholders using claims feed. Format the response as numbered steps. Provide..."

### Cluster #1 (Size: 7 samples)
- **Primary Domain/Category**: `summarization`
- **Primary Difficulty**: `medium`
- **Exemplar Prompt ID**: `[gen_004549]`
- **Representative Prompt Text**: "Summarize a detailed safety envelope for a robotics engineer, highlighting cycle time, collision risk, and impact on gripper. Include a realistic constraint and..."

### Cluster #2 (Size: 10 samples)
- **Primary Domain/Category**: `creative_writing`
- **Primary Difficulty**: `easy`
- **Exemplar Prompt ID**: `[gen_001502]`
- **Representative Prompt Text**: "Compose a reflective story in which loyalty segment becomes a symbol for changing conversion rate. Include a realistic constraint and one exception case. Format..."

### Cluster #3 (Size: 22 samples)
- **Primary Domain/Category**: `summarization`
- **Primary Difficulty**: `easy`
- **Exemplar Prompt ID**: `[gen_005589]`
- **Representative Prompt Text**: "Summarize a detailed index report for a database administrator, highlighting lock wait time, data loss, and impact on materialized view. Include interacting con..."

### Cluster #4 (Size: 19 samples)
- **Primary Domain/Category**: `coding`
- **Primary Difficulty**: `easy`
- **Exemplar Prompt ID**: `[gen_002785]`
- **Representative Prompt Text**: "Write production-ready code for a principal investigator that processes a survey instrument involving control condition and handles sampling bias. Format the re..."

## Relabeling and Active Learning Recommendations

To resolve these semantic errors in future phases, we recommend prioritizing these samples for audit and relabeling:

- **gen_005553** (coding, medium): Ground Truth=`LOCAL`, Model Pred=`REMOTE`. Prompt: "Write production-ready code for a ML engineer that processes a training run involving validation split and handles overfitting. Include a realistic constraint and one exception case. Format the respon..."
- **gen_001502** (creative_writing, medium): Ground Truth=`LOCAL`, Model Pred=`REMOTE`. Prompt: "Compose a reflective story in which loyalty segment becomes a symbol for changing conversion rate. Include a realistic constraint and one exception case. Format the response as plain paragraphs. Provi..."
- **gen_000549** (summarization, hard): Ground Truth=`LOCAL`, Model Pred=`REMOTE`. Prompt: "Summarize a detailed index report for a database administrator, highlighting lock wait time, data loss, and impact on materialized view. Include interacting constraints, uncertainty, and second-order ..."
- **gen_000161** (coding, hard): Ground Truth=`LOCAL`, Model Pred=`REMOTE`. Prompt: "Write production-ready code for a ML engineer that processes a feature store involving feature pipeline and handles bias amplification. Include interacting constraints, uncertainty, and second-order e..."
- **gen_004037** (summarization, easy): Ground Truth=`LOCAL`, Model Pred=`REMOTE`. Prompt: "Summarize a detailed load balancer for a platform engineer, highlighting CPU utilization, cost overrun, and impact on tenant namespace. Format the response as YAML block. Return the core result using ..."
- **gen_004489** (coding, easy): Ground Truth=`LOCAL`, Model Pred=`REMOTE`. Prompt: "Write production-ready code for a database administrator that processes a index report involving foreign key and handles deadlock. Format the response as checklist. Use accessible language for a mixed..."
- **gen_005547** (reasoning, easy): Ground Truth=`LOCAL`, Model Pred=`REMOTE`. Prompt: "Analyze whether a model risk reviewer should prioritize evaluation notebook improvements or direct mitigation of data leakage. Format the response as annotated code block. Discuss edge cases and failu..."
- **gen_000394** (planning, medium): Ground Truth=`REMOTE`, Model Pred=`LOCAL`. Prompt: "Design a phased rollout for credit model adoption across teams that depend on merchant account. Include a realistic constraint and one exception case. Format the response as numbered steps. Use enough..."
- **gen_003121** (coding, easy): Ground Truth=`LOCAL`, Model Pred=`REMOTE`. Prompt: "Write production-ready code for a legal operations manager that processes a data processing agreement involving regulated entity and handles ambiguous liability. Format the response as plain paragraph..."
- **gen_005225** (coding, hard): Ground Truth=`LOCAL`, Model Pred=`REMOTE`. Prompt: "Write production-ready code for a database administrator that processes a replication stream involving materialized view and handles slow query. Include interacting constraints, uncertainty, and secon..."