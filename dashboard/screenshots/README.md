# Dashboard Screenshot Checklist

Save the Task 08 dashboard captures here, separate from `dashboard/artifacts/`.

Use these filenames:

1. `01_graph_overview.png` - Overview tab showing graph statistics and graph visualizations.
2. `02_model_performance.png` - Models tab showing GCN/GAT metrics and comparison.
3. `03_node_predictions.png` - Predictions tab showing real sampled classification results.
4. `04_embedding_visualization.png` - Embeddings tab showing the interactive PCA or t-SNE plot.
5. `05_explainability.png` - Explanations tab showing attention or neighbourhood influence.
6. `06_live_inference.png` - Live test tab after **Run live inference** succeeds (recommended).

Before capturing, run from the project root:

```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard\app.py
```

Use a wide browser window and keep the title, selected tab, values, charts and success message
readable. Exclude unrelated tabs, notifications, account details and desktop content.
