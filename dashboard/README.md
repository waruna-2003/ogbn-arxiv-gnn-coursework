# Dashboard

`app.py` is the Streamlit application for Coursework Task 08. It reads the generated files in
`artifacts/`.

Run it from the project root:

```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard\app.py
```

## Artifacts

Artifacts are grouped by purpose:

- **Graph data:** `graph_stats.json`, `node_features.pt`, `edge_index.pt`,
  `class_names.json`, `sample_subgraph.png`, `degree_distribution.png`, and
  `connected_components.png`.
- **Models and evaluation:** `gcn_model.pt`, `gat_model.pt`, `model_metrics.json`,
  `gcn_training_log.csv`, `gat_training_log.csv`, `training_curves.png`,
  `model_comparison.png`, `predictions_sample.csv`, and `results_summary.txt`.
- **Embeddings and explainability:** `embeddings_2d.csv`, `tsne_embeddings.png`,
  `attention_weights.png`, `attention_weights_incorrect.png`,
  `attention_quality_summary.json`, and `neighbourhood_influence.csv`.
- **Extended work:** `extended_three_way_comparison.png` and
  `ssl_pretraining_summary.json`.

## Validated run

- Original directed citation edges: **1,166,243**.
- Undirected message-passing edges: **2,315,598**.
- GCN test accuracy / macro F1: **71.71% / 0.5054**.
- GAT test accuracy / macro F1: **70.72% / 0.4781**.
- JSON and CSV files, tensor dimensions, edge symmetry, state dictionaries, dashboard tabs and
  full-graph inference were validated.

These are generated results. If a required artifact is absent, the dashboard reports the missing
file instead of substituting data.

## Screenshot deliverable

Store final dashboard captures in `screenshots/` using its checklist. Keep screenshots out of
`artifacts/`, which is reserved for notebook-generated data.
