# Semantic Embedding-Based Prompt Router

This module implements a semantic embedding-based routing model for the Hybrid Token Router project (Phase 5). It supports pure embedding classification and a hybrid classification approach combining semantic features with handcrafted lexical/structural features.

## Architecture

The module contains the following files:

- **`embedding_utils.py`**: Dynanmic loading of the SentenceTransformer models, hardware device selection (supporting CUDA/ROCm/CPU), and loading latency tracking.
- **`embedding_extractor.py`**: Prompt embedding generation with batching and local file-based caching. Caching uses SHA-256 hashes of prompts along with the model name and its configuration version to automatically invalidate stale entries.
- **`embedding_dataset.py`**: Loads the training datasets, extracts embeddings, generates combined hybrid matrices, and outputs stratified 80/20 train/test splits.
- **`embedding_train.py`**: Compares Logistic Regression, Random Forest, Linear SVM, and Gradient Boosting/XGBoost classifiers on both the pure embedding space and the hybrid space. Model selection is performed using a weighted ranking score:
  - **F1 Score**: 40%
  - **ROC AUC**: 30%
  - **Precision**: 15%
  - **Recall**: 15%
- **`embedding_predict.py`**: Routing decisions at runtime. Exposes `route_embedding` and `route_hybrid` functions mapping to FastAPI response schemas.
- **`embedding_evaluate.py`**: Evaluates all routers (Heuristic, ML, Embedding, Hybrid) on standard validation sets and a new set of 104 unseen prompts. Benchmarks latency, memory, and model file sizes, performing category-level analysis and generating comparison reports.
- **`embedding_visualization.py`**: Generates t-SNE and PCA projections of the prompt embeddings and bar charts comparing router accuracy, F1 scores, and latency.

## Setup and Dependencies

This module requires `sentence-transformers` and its dependencies (`torch`, `transformers`):

```bash
pip install -r backend/requirements.txt
```

## Running the Pipelines

### 1. Model Training
Train and select the best embedding and hybrid routers:
```bash
python backend/app/embedding_router/embedding_train.py
```
This saves:
- Pure embedding router to `backend/models/embedding_classifier.pkl`
- Hybrid router to `backend/models/hybrid_classifier.pkl`

### 2. Model Evaluation and Report Generation
Run holdout validation, unseen evaluation, and feature analysis:
```bash
python backend/app/embedding_router/embedding_evaluate.py
```
This generates:
- `backend/docs/router_comparison.md`
- `backend/docs/unseen_router_comparison.md`
- `backend/docs/embedding_analysis.md`
- Performance charts in `backend/app/ml/plots/`

### 3. Model Projections Visualization
Generate PCA and t-SNE scatter plots:
```bash
python backend/app/embedding_router/embedding_visualization.py
```
This saves `embedding_pca.png` and `embedding_tsne.png` in `backend/app/ml/plots/`.

## Configuration Settings

You can configure the routing mode and the embedding model name in the backend environment file (`backend/.env`):

```env
# Selected Routing Mode: 'heuristic', 'ml', 'embedding', or 'hybrid'
ROUTER_MODE=embedding

# Embedding Model Name (Loaded via HuggingFace SentenceTransformers)
EMBEDDING_MODEL_NAME=BAAI/bge-small-en-v1.5
```
