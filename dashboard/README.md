# Dashboard

`app.py` is the Streamlit application for Coursework Task 08. `artifacts/` contains the real
outputs it displays.

Run it from the project root:

```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard\app.py
```

## Artifacts

The 24 files are grouped by purpose:

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
- All JSON and CSV files, tensor dimensions, edge symmetry, saved state dictionaries, all seven
  dashboard tabs, and a real full-graph inference were validated successfully.

These are generated results, not manually maintained source files. The dashboard has no synthetic
fallback: if a required artifact is absent, it reports the missing file.

## Screenshot deliverable

The coursework also requires dashboard screenshots. Store the final images in `screenshots/` using
the filenames and capture checklist documented in `screenshots/README.md`. Do not place screenshots
inside `artifacts/`; that directory is reserved for notebook-generated data.
