# CCS4354 OGBN-Arxiv GNN Coursework

This repository contains the completed group implementation for the CCS4354 Tensors and Graphs
coursework. The project performs node classification on the OGBN-Arxiv citation network using a
Graph Convolutional Network (GCN), Graph Attention Network (GAT), a bonus Graph Transformer, and
self-supervised feature pre-training.

The five member sections have been merged into `main`. The canonical notebook contains the
integrated implementation and saved execution outputs, while the Streamlit dashboard displays the
generated experimental artifacts.

## Repository structure

```text
.
|-- README.md
|-- requirements.txt
|-- notebook/
|   `-- CCS4354_OGBN_Arxiv_Coursework.ipynb
|       Completed integrated and executed coursework notebook
|-- dashboard/
|   |-- app.py
|   |   Streamlit dashboard for inspecting the graph and model results
|   |-- README.md
|   |   Dashboard-specific guidance
|   |-- artifacts/
|   |   Notebook-generated metrics, charts, predictions, tensors and models
|   `-- screenshots/
|       Location for final dashboard evidence images
|-- data/
|   `-- ogbn_arxiv/
|       Local OGBN-Arxiv dataset; ignored by Git and downloaded when required
`-- .venv/
    Local Python environment; ignored by Git
```

The technical report, presentation, video, contribution declaration and final submission archive
are separate coursework deliverables. They should be added or packaged only after the group has
approved their final versions.

## Current project status

- All five member implementations have been merged into `main`.
- The canonical notebook contains 58 implementation cells and saved outputs from the final run.
- GCN, GAT, Graph Transformer and self-supervised experiments are included.
- The dashboard uses real notebook-generated artifacts and does not fabricate fallback results.
- Artifacts required for the dashboard and full-graph live inference are present.
- The final submission-ZIP cell is intentionally left for the submission packaging stage.

## Main results

| Model | Test accuracy | Macro F1 | Training time | Parameters |
|---|---:|---:|---:|---:|
| GCN | 71.71% | 0.5054 | 129.1 s | 110,120 |
| GAT | 70.72% | 0.4781 | 234.4 s | 110,200 |
| Graph Transformer | 70.79% | 0.4863 | 417.1 s | 437,408 |

Additional graph information:

- Nodes: **169,343**
- Original directed citation edges: **1,166,243**
- Undirected message-passing edges: **2,315,598**
- Node features: **128**
- Target classes: **40**

The GCN is the strongest practical model in this experiment: it achieves the highest test
accuracy and macro F1 while training substantially faster than GAT and the Graph Transformer.

## Group contributions

| Member | Main responsibility | Integrated sections |
|---|---|---|
| Binara Hansaka | Tensor fundamentals and graph analysis | Tasks 01–02 |
| Dilshan Kodithuwakku | Graph preparation and GCN | Task 03 and GCN implementation |
| Pramudi Biyonika | GAT and controlled model training | GAT and Task 05 |
| Tharindu Kothalawala | Evaluation and explainability | Tasks 06–07 |
| Waruna Silva | Dashboard, bonus models, integration and final execution | Task 08, extended work and final integration |

The individual contributions and pull-request merges remain visible in the Git history. Future
changes should still be made on short-lived feature branches and merged through reviewed pull
requests rather than committed directly to `main`.

## Recreate the environment

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The main dependencies are PyTorch, PyTorch Geometric, OGB, Streamlit, pandas, NumPy, Plotly,
Matplotlib, scikit-learn, NetworkX, SciPy, seaborn and JupyterLab.

## Open the notebook

```powershell
.\.venv\Scripts\python.exe -m jupyter lab notebook\CCS4354_OGBN_Arxiv_Coursework.ipynb
```

Start Jupyter from the repository root because project paths are relative to this directory.

The committed notebook already contains final outputs. A complete retraining run is computationally
expensive, particularly for the full undirected GAT and bonus models. Google Colab with a T4 GPU is
recommended; a local GPU is not required to view the notebook or run the dashboard.

## Run the dashboard

```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard\app.py
```

Then open the local Streamlit address displayed in the terminal, normally
`http://localhost:8501`.

The standard tabs load saved results immediately. The **Run live inference** button performs a real
full-graph forward pass using the saved GCN and GAT weights and may take time on a CPU.

## Regenerate dashboard artifacts

Regenerate artifacts only when the integrated notebook or experimental configuration changes:

1. Upload `notebook/CCS4354_OGBN_Arxiv_Coursework.ipynb` to Google Colab.
2. Select **Runtime > Change runtime type > T4 GPU**.
3. Run the notebook from the first cell to the final artifact export cell.
4. Confirm that training, evaluation and artifact validation finish without errors.
5. Download the generated `dashboard_artifacts_undirected.zip` archive.
6. Back up the existing `dashboard/artifacts/` directory.
7. Extract the verified archive into `dashboard/artifacts/`.
8. Run the dashboard and confirm that every tab loads the new results.

Do not replace the committed artifacts with partial outputs from an interrupted notebook run.

## Git workflow for further changes

```powershell
git switch main
git pull origin main
git switch -c feature/short-description
```

Make the scoped change, test it, commit it, push the branch, and open a pull request. Because Jupyter
notebooks are JSON documents, avoid having multiple people edit the same notebook cells at the same
time. Pull the latest `main` before starting and keep notebook changes limited to the assigned
section.

## Submission checklist

Before creating the final submission package:

- Confirm all member names and student IDs in the technical report.
- Add the approved technical report and presentation.
- Capture final dashboard screenshots.
- Verify that the notebook opens and its saved outputs are visible.
- Confirm the dashboard launches using the committed artifacts.
- Review GitHub contribution history and contribution percentages.
- Include the required demonstration video or video link.
- Run the final packaging cell only after the deliverable set is complete.
