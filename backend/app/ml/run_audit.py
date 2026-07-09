import pandas as pd
import numpy as np
import json
import re
import os
from collections import Counter, defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Set paths
CSV_PATH = "app/data/training/training_dataset_merged.csv"
JSON_PATH = "app/data/datasets/10_000_chatbot_prompts.json"

print("Loading dataset...")
df = pd.read_csv(CSV_PATH)
print(f"Loaded CSV with {len(df)} rows and {df.shape[1]} columns.")

# Task 1: Category Distribution
print("\n--- Task 1: Category Distribution ---")
cat_stats = []
categories = df['category'].unique()

for cat in categories:
    cat_df = df[df['category'] == cat]
    total = len(cat_df)
    local_cnt = (cat_df['label'] == 'LOCAL').sum()
    remote_cnt = (cat_df['label'] == 'REMOTE').sum()
    remote_rate = (remote_cnt / total) * 100 if total > 0 else 0
    avg_len = cat_df['word_count'].mean()
    avg_complexity = cat_df['estimated_complexity'].mean()
    avg_reasoning = cat_df['reasoning_score'].mean()
    
    cat_stats.append({
        'Category': cat,
        'Total': int(total),
        'LOCAL': int(local_cnt),
        'REMOTE': int(remote_cnt),
        'REMOTE %': float(remote_rate),
        'Avg Length': float(avg_len),
        'Avg Complexity': float(avg_complexity),
        'Avg Reasoning': float(avg_reasoning)
    })

cat_stats_df = pd.DataFrame(cat_stats).sort_values(by='Total', ascending=False)
print(cat_stats_df.head(20))
cat_stats_df.to_csv("scratch_cat_stats.csv", index=False)

# Task 2: Parent Category Analysis
print("\n--- Task 2: Parent Category Analysis ---")
# Load original chatbot prompts to map them back
with open(JSON_PATH, 'r', encoding='utf-8') as f:
    json_data = json.load(f)

# Helper function to clean text (same as prepare_merged_dataset.py)
def _clean(value):
    return re.sub(r"\s+", " ", str(value or "")).strip()

# Build a mapping from prompt_id to (parent_category, subcategory)
prompt_mapping = {}
for idx, item in enumerate(json_data):
    pid = f"imported_{_clean(item.get('id')) or idx}"
    prompt_mapping[pid] = {
        'parent_category': item.get('parent_category', 'unknown'),
        'subcategory': item.get('subcategory', 'unknown')
    }

# Add parent_category and subcategory columns to the dataframe
df['parent_category'] = df['prompt_id'].map(lambda x: prompt_mapping.get(x, {}).get('parent_category', np.nan))
df['subcategory'] = df['prompt_id'].map(lambda x: prompt_mapping.get(x, {}).get('subcategory', np.nan))

# Filter for rows that actually came from the imported dataset (source == '10k_chatbot_prompts')
imported_df = df[df['source'] == '10k_chatbot_prompts']
print(f"Number of imported prompts: {len(imported_df)}")

parent_stats = []
parents = imported_df['parent_category'].dropna().unique()
for p in parents:
    p_df = imported_df[imported_df['parent_category'] == p]
    total = len(p_df)
    local_cnt = (p_df['label'] == 'LOCAL').sum()
    remote_cnt = (p_df['label'] == 'REMOTE').sum()
    remote_rate = (remote_cnt / total) * 100 if total > 0 else 0
    avg_len = p_df['word_count'].mean()
    avg_complexity = p_df['estimated_complexity'].mean()
    avg_reasoning = p_df['reasoning_score'].mean()
    
    parent_stats.append({
        'Parent Category': p,
        'Total': int(total),
        'LOCAL': int(local_cnt),
        'REMOTE': int(remote_cnt),
        'REMOTE %': float(remote_rate),
        'Avg Length': float(avg_len),
        'Avg Complexity': float(avg_complexity),
        'Avg Reasoning': float(avg_reasoning)
    })

parent_stats_df = pd.DataFrame(parent_stats).sort_values(by='Total', ascending=False)
print(parent_stats_df.head(20))
parent_stats_df.to_csv("scratch_parent_stats.csv", index=False)

# Task 3: Prompt Length Analysis
print("\n--- Task 3: Prompt Length Analysis ---")
def get_bucket(words):
    if words <= 100:
        return "0-100 words"
    elif words <= 250:
        return "100-250"
    elif words <= 500:
        return "250-500"
    else:
        return "500+"

df['length_bucket'] = df['word_count'].apply(get_bucket)
bucket_order = ["0-100 words", "100-250", "250-500", "500+"]

bucket_stats = []
for b in bucket_order:
    b_df = df[df['length_bucket'] == b]
    total = len(b_df)
    local_cnt = (b_df['label'] == 'LOCAL').sum()
    remote_cnt = (b_df['label'] == 'REMOTE').sum()
    avg_routing_score = b_df['routing_score'].mean()
    remote_rate = (remote_cnt / total) * 100 if total > 0 else 0
    
    bucket_stats.append({
        'Bucket': b,
        'Total': int(total),
        'LOCAL': int(local_cnt),
        'REMOTE': int(remote_cnt),
        'REMOTE %': float(remote_rate),
        'Avg Routing Score': float(avg_routing_score)
    })
bucket_stats_df = pd.DataFrame(bucket_stats)
print(bucket_stats_df)

# Task 4: Routing Decision Analysis - Feature Importance
print("\n--- Task 4: Routing Decision Analysis ---")
feature_cols = [
    "prompt_length", "word_count", "estimated_input_tokens", "contains_code", "contains_math",
    "contains_json", "contains_markdown", "contains_numbers", "contains_question",
    "reasoning_score", "technical_complexity", "reasoning_depth", "task_complexity",
    "constraint_complexity", "context_complexity", "complexity_score", "constraint_density",
    "presence_of_tables", "sql_indicators", "api_keywords",
    "system_design_keywords", "algorithmic_complexity", "dependency_between_subtasks",
    "multi_turn_context", "code_indicators", "math_indicators"
]

# Ensure we use columns that exist in the dataframe and are numeric
X_cols = [col for col in feature_cols if col in df.columns]
print(f"Features used for training: {len(X_cols)}")

X = df[X_cols].copy()
# Handing potential dict/string columns by forcing numeric or filling NaNs
for col in X.columns:
    if X[col].dtype == 'object':
        try:
            X[col] = pd.to_numeric(X[col])
        except Exception:
            print(f"Column {col} is object type. Label encoding.")
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    X[col] = X[col].fillna(0)

y = (df['label'] == 'REMOTE').astype(int)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

importances = pd.DataFrame({
    'Feature': X_cols,
    'Importance': rf.feature_importances_
}).sort_values(by='Importance', ascending=False)
print("\nRandom Forest Feature Importance:")
print(importances.head(15))

# Also calculate correlation with target (y)
correlations = []
for col in X_cols:
    try:
        corr = X[col].corr(y)
        correlations.append({'Feature': col, 'Correlation': float(corr)})
    except Exception:
        pass
corr_df = pd.DataFrame(correlations).sort_values(by='Correlation', key=abs, ascending=False)
print("\nCorrelations with REMOTE:")
print(corr_df.head(15))

# Task 5: Category Imbalance Detection
print("\n--- Task 5: Category Imbalance Detection ---")
suspicious_cats = []
for stat in cat_stats:
    if stat['Total'] >= 5: # only consider categories with sufficient samples
        if stat['REMOTE %'] >= 90 or stat['REMOTE %'] <= 10:
            suspicious_cats.append(stat)
            print(f"Suspicious: Category={stat['Category']}, Total={stat['Total']}, REMOTE%={stat['REMOTE %']:.1f}%")

# Task 6: Dataset Quality Audit
print("\n--- Task 6: Dataset Quality Audit ---")
# Duplicate Rate
total_samples = len(df)
exact_duplicates = df.duplicated(subset=['prompt']).sum()
dup_rate = (exact_duplicates / total_samples) * 100
print(f"Exact duplicates inside merged dataset: {exact_duplicates} ({dup_rate:.4f}%)")

# Near duplicates global calculation
print("Computing global near-duplicates (Jaccard similarity threshold 0.82)...")
def normalize_prompt(prompt):
    text = str(prompt).lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def get_shingles(words, size=5):
    if len(words) < size:
        return {tuple(words)}
    return {tuple(words[i:i + size]) for i in range(len(words) - size + 1)}

prompts_normalized = [normalize_prompt(p).split() for p in df['prompt'].tolist()]
shingle_sets = [get_shingles(words) for words in prompts_normalized]

# Build inverted index to speed up near-duplicate search
inverted_index = defaultdict(list)
for idx, s_set in enumerate(shingle_sets):
    for shingle in s_set:
        inverted_index[shingle].append(idx)

# Find near-duplicate pairs
near_dup_pairs = set()
checked_pairs = set()

for idx, s_set in enumerate(shingle_sets):
    candidates = set()
    for shingle in s_set:
        candidates.update(inverted_index[shingle])
    candidates.discard(idx)
    
    for cand in candidates:
        if cand < idx:
            pair = (cand, idx)
        else:
            pair = (idx, cand)
            
        if pair in checked_pairs:
            continue
        checked_pairs.add(pair)
        
        left = s_set
        right = shingle_sets[cand]
        if not left and not right:
            sim = 1.0
        else:
            sim = len(left & right) / max(len(left | right), 1)
            
        if sim >= 0.82:
            near_dup_pairs.add(pair)

near_dup_cnt = len(near_dup_pairs)
near_dup_rate = (near_dup_cnt / total_samples) * 100
print(f"Global near-duplicates found: {near_dup_cnt} ({near_dup_rate:.4f}%)")

# Empty prompts
empty_prompts = df['prompt'].isna().sum() + (df['prompt'].astype(str).str.strip() == '').sum()
print(f"Empty prompts: {empty_prompts}")

# Missing metadata
missing_meta = df[X_cols].isna().sum().sum()
print(f"Missing metadata cells: {missing_meta}")

# Class overlap and label consistency
overlap_cnt = 0
inconsistent_pairs = []
for p1, p2 in near_dup_pairs:
    lbl1 = df.loc[p1, 'label']
    lbl2 = df.loc[p2, 'label']
    if lbl1 != lbl2:
        overlap_cnt += 1
        inconsistent_pairs.append((int(p1), int(p2), df.loc[p1, 'prompt'], df.loc[p2, 'prompt'], lbl1, lbl2))

print(f"Class overlap / Label inconsistency count: {overlap_cnt} pairs out of {near_dup_cnt} near-duplicate pairs ({overlap_cnt/max(near_dup_cnt, 1)*100:.2f}% of near duplicates)")

if overlap_cnt > 0:
    print("\nExamples of label inconsistency:")
    for idx, (p1, p2, pr1, pr2, l1, l2) in enumerate(inconsistent_pairs[:5]):
        print(f"Pair {idx+1}:")
        print(f"  P1: {pr1[:100]}... [Label: {l1}]")
        print(f"  P2: {pr2[:100]}... [Label: {l2}]")

# Prompt Diversity: Vocabulary size & Type-Token Ratio (TTR)
all_words = []
for words in prompts_normalized:
    all_words.extend(words)

unique_words = len(set(all_words))
total_words = len(all_words)
ttr = unique_words / total_words if total_words > 0 else 0
print(f"Total words: {total_words}")
print(f"Unique words: {unique_words}")
print(f"Type-Token Ratio (TTR): {ttr:.4f}")

# Average prompt length in words
avg_prompt_len = df['word_count'].mean()
print(f"Average prompt length: {avg_prompt_len:.2f} words")

# Serialization helper
def make_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return make_serializable(obj.tolist())
    else:
        return obj

print("\nSaving results to json...")
results = {
    'cat_stats': cat_stats,
    'parent_stats': parent_stats,
    'bucket_stats': bucket_stats,
    'importances': importances.to_dict('records'),
    'correlations': corr_df.to_dict('records'),
    'near_duplicates_count': int(near_dup_cnt),
    'near_duplicates_rate': float(near_dup_rate),
    'exact_duplicates_count': int(exact_duplicates),
    'exact_duplicates_rate': float(dup_rate),
    'inconsistent_pairs_count': int(overlap_cnt),
    'total_samples': int(total_samples),
    'ttr': float(ttr),
    'unique_words': int(unique_words),
    'total_words': int(total_words),
    'avg_prompt_len': float(avg_prompt_len),
    'empty_prompts': int(empty_prompts),
    'missing_meta': int(missing_meta)
}

results = make_serializable(results)

with open("scratch_audit_results.json", "w") as f:
    json.dump(results, f, indent=4)
print("Done!")
