# Dashboard Screenshot Checklist

Task 08 requires a working Streamlit dashboard **and screenshots**. Save the final captures here so
they remain separate from notebook-generated files in `dashboard/artifacts/`.

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

Use a wide browser window, keep the sidebar collapsed unless it adds useful context, and make sure
the page title, selected tab, values, charts, and success message are readable. Avoid including
unrelated browser tabs, notifications, account details, or desktop content.
