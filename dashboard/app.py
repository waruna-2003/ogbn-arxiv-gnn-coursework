"""Streamlit dashboard for the OGBN-Arxiv coursework results.

The adjacent ``artifacts`` directory contains outputs exported by the notebook.
Required files are validated before the interface loads.
"""

import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")

st.set_page_config(
    page_title="Graph Intelligence Dashboard | OGBN-Arxiv",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Class indices are not alphabetical; use the official mapping when available.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --bg-deep: #05070f;
    --bg-panel: rgba(255,255,255,0.04);
    --accent1: #7C5CFF;
    --accent2: #22D3EE;
    --accent3: #F472B6;
    --accent-green: #34D399;
    --text-main: #EDEFFB;
    --text-dim: #9AA1C4;
    --border: rgba(255,255,255,0.09);
}

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
    color: var(--text-main);
}

.stApp {
    background: radial-gradient(circle at 15% 10%, rgba(124,92,255,0.20), transparent 45%),
                radial-gradient(circle at 85% 0%, rgba(34,211,238,0.16), transparent 40%),
                radial-gradient(circle at 50% 100%, rgba(244,114,182,0.12), transparent 45%),
                var(--bg-deep);
    background-size: 200% 200%;
    animation: bgFloat 22s ease-in-out infinite;
}
@keyframes bgFloat {
    0%   { background-position: 0% 0%; }
    50%  { background-position: 100% 60%; }
    100% { background-position: 0% 0%; }
}

.hero {
    padding: 2.1rem 2.4rem;
    border-radius: 22px;
    background: linear-gradient(120deg, rgba(124,92,255,0.22), rgba(34,211,238,0.14) 55%, rgba(244,114,182,0.14));
    border: 1px solid var(--border);
    box-shadow: 0 20px 60px rgba(76,50,180,0.25);
    margin-bottom: 1.6rem;
    position: relative;
    overflow: hidden;
    animation: heroIn 0.9s cubic-bezier(0.16,1,0.3,1);
}
.hero::after {
    content: "";
    position: absolute; inset: 0;
    background: linear-gradient(100deg, transparent 30%, rgba(255,255,255,0.10) 45%, transparent 60%);
    background-size: 250% 250%;
    animation: shimmer 6s ease-in-out infinite;
}
@keyframes shimmer {
    0% { background-position: 200% 0%; }
    100% { background-position: -50% 0%; }
}
@keyframes heroIn {
    from { opacity: 0; transform: translateY(-18px); }
    to   { opacity: 1; transform: translateY(0); }
}
.hero h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.35rem;
    font-weight: 700;
    margin: 0 0 0.35rem 0;
    background: linear-gradient(90deg, #ffffff, var(--accent2) 60%, var(--accent1));
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}
.hero p { color: var(--text-dim); font-size: 1.02rem; margin: 0; }
.badge-row { margin-top: 0.9rem; display: flex; gap: 0.5rem; flex-wrap: wrap; }
.badge {
    padding: 0.28rem 0.75rem; border-radius: 999px; font-size: 0.78rem; font-weight: 600;
    background: rgba(255,255,255,0.07); border: 1px solid var(--border); color: var(--text-main);
    display: inline-flex; align-items: center; gap: 0.35rem;
}
.dot { width: 7px; height: 7px; border-radius: 50%; background: var(--accent-green);
       box-shadow: 0 0 8px var(--accent-green); animation: pulse 1.6s ease-in-out infinite; }
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.35; } }

.glass {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 1.25rem 1.4rem;
    backdrop-filter: blur(14px);
    transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
    animation: fadeUp 0.6s cubic-bezier(0.16,1,0.3,1) both;
}
.glass:hover {
    transform: translateY(-4px);
    box-shadow: 0 14px 34px rgba(124,92,255,0.22);
    border-color: rgba(124,92,255,0.45);
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
}

.metric-label { font-size: 0.78rem; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600; }
.metric-value { font-family: 'Space Grotesk', sans-serif; font-size: 2.1rem; font-weight: 700; margin-top: 0.15rem; }
.metric-delta { font-size: 0.8rem; margin-top: 0.2rem; font-weight: 600; }
.metric-up { color: var(--accent-green); }
.metric-down { color: var(--accent3); }

.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.35rem; font-weight: 700; margin: 1.6rem 0 0.7rem 0;
    display: flex; align-items: center; gap: 0.5rem;
}
.section-title .bar { width: 5px; height: 22px; border-radius: 4px;
    background: linear-gradient(180deg, var(--accent1), var(--accent2)); display: inline-block; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(12,14,28,0.96), rgba(6,8,18,0.98));
    border-right: 1px solid var(--border);
}

.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] {
    background: rgba(255,255,255,0.04); border-radius: 10px 10px 0 0; padding: 0.55rem 1.1rem;
    border: 1px solid var(--border); border-bottom: none; color: var(--text-dim); font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, rgba(124,92,255,0.35), rgba(34,211,238,0.20));
    color: #fff;
}

.conf-bar-track { width: 100%; background: rgba(255,255,255,0.06); border-radius: 999px; height: 10px; overflow: hidden; }
.conf-bar-fill { height: 100%; border-radius: 999px;
    background: linear-gradient(90deg, var(--accent1), var(--accent2));
    background-size: 200% 100%; animation: fillMove 2.2s ease-in-out infinite; }
@keyframes fillMove { 0% {background-position: 0% 0%;} 100% {background-position: 100% 0%;} }

footer, #MainMenu { visibility: hidden; }

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--bg-panel) !important;
    border: 1px solid var(--border) !important;
    border-radius: 18px !important;
    backdrop-filter: blur(14px);
    transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 14px 34px rgba(124,92,255,0.18);
    border-color: rgba(124,92,255,0.4) !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
:root {
    --bg-deep: #f4f7fb;
    --bg-panel: #ffffff;
    --accent1: #3157d5;
    --accent2: #168a9f;
    --accent3: #b4235a;
    --accent-green: #16824b;
    --text-main: #172033;
    --text-dim: #596579;
    --border: #dce3ee;
}
html, body, [class*="css"] { color: var(--text-main); }
.stApp { background: var(--bg-deep); animation: none; }
.block-container { max-width: 1320px; padding-top: 1.5rem; padding-bottom: 3rem; }
.hero {
    padding: 1.4rem 1.6rem; border-radius: 12px; margin-bottom: 1rem;
    background: #ffffff; border: 1px solid var(--border); box-shadow: none;
    animation: none;
}
.hero::after { display: none; }
.hero h1 { color: #172033; background: none; font-size: 2rem; }
.hero p { color: var(--text-dim); }
.badge { background: #eef3ff; border-color: #d6e0fa; color: #253b78; }
.dot { animation: none; box-shadow: none; }
.glass {
    background: #ffffff; border: 1px solid var(--border); border-radius: 10px;
    padding: 1rem 1.1rem; box-shadow: none; animation: none; backdrop-filter: none;
}
.glass:hover { transform: none; box-shadow: none; border-color: var(--border); }
.metric-label { color: var(--text-dim); letter-spacing: 0; text-transform: none; font-size: .86rem; }
.metric-value { color: #172033; font-size: 1.65rem; }
.section-title { color: #172033; font-size: 1.25rem; margin-top: 1.8rem; }
.section-title .bar { background: #3157d5; }
section[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid var(--border); }
.stTabs [data-baseweb="tab-list"] { gap: 2px; overflow-x: auto; }
.stTabs [data-baseweb="tab"] {
    background: transparent; border: none; border-bottom: 3px solid transparent;
    border-radius: 0; color: #596579; padding: .65rem .8rem; font-weight: 600;
}
.stTabs [aria-selected="true"] { background: transparent; color: #2347b2; border-bottom-color: #3157d5; }
.conf-bar-track { background: #e7ecf3; }
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #ffffff !important; border: 1px solid var(--border) !important;
    border-radius: 10px !important; backdrop-filter: none; box-shadow: none;
}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    transform: none; box-shadow: none; border-color: var(--border) !important;
}
@media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
</style>
""", unsafe_allow_html=True)


# Required notebook exports.
REQUIRED_ARTIFACTS = {
    "graph_stats.json": "graph statistics",
    "model_metrics.json": "model performance metrics (accuracy/precision/recall/F1)",
    "predictions_sample.csv": "node classification predictions",
    "embeddings_2d.csv": "2D t-SNE/PCA embedding projections",
    "gcn_training_log.csv": "GCN training curve log",
    "gat_training_log.csv": "GAT training curve log",
}


@st.cache_data(show_spinner=False)
def load_artifacts():
    art = {}
    path = lambda name: os.path.join(ARTIFACT_DIR, name)

    missing = [fname for fname in REQUIRED_ARTIFACTS if not os.path.exists(path(fname))]
    if missing:
        return {"missing": missing}

    with open(path("graph_stats.json")) as f:
        art["graph_stats"] = json.load(f)

    # A missing mapping uses neutral labels rather than guessed subject names.
    num_classes = art["graph_stats"]["num_classes"]
    class_names_loaded = None
    if os.path.exists(path("class_names.json")):
        with open(path("class_names.json")) as f:
            class_names_loaded = json.load(f)
    if class_names_loaded and len(class_names_loaded) == num_classes:
        art["class_names"] = class_names_loaded
        art["class_names_are_real"] = True
    else:
        art["class_names"] = [f"Class {i}" for i in range(num_classes)]
        art["class_names_are_real"] = False

    with open(path("model_metrics.json")) as f:
        art["model_metrics"] = json.load(f)

    art["predictions"] = pd.read_csv(path("predictions_sample.csv"))
    art["embeddings"] = pd.read_csv(path("embeddings_2d.csv"))
    art["gcn_log"] = pd.read_csv(path("gcn_training_log.csv"))
    art["gat_log"] = pd.read_csv(path("gat_training_log.csv"))

    infl_path = path("neighbourhood_influence.csv")
    art["neighbourhood_influence"] = pd.read_csv(infl_path) if os.path.exists(infl_path) else None

    attn_csv_path = path("attention_weights.csv")
    art["attention_table"] = pd.read_csv(attn_csv_path) if os.path.exists(attn_csv_path) else None

    attn_quality_path = path("attention_quality_summary.json")
    if os.path.exists(attn_quality_path):
        with open(attn_quality_path) as f:
            art["attention_quality"] = json.load(f)
    else:
        art["attention_quality"] = None

    ssl_path = path("ssl_pretraining_summary.json")
    if os.path.exists(ssl_path):
        with open(ssl_path) as f:
            art["ssl_summary"] = json.load(f)
    else:
        art["ssl_summary"] = None

    return art


data = load_artifacts()

if "missing" in data:
    st.error(
        "**Real model artifacts were not found.** This dashboard only displays genuine results "
        "from the coursework notebook — it does not fall back to synthetic data.\n\n"
        "Please run the coursework notebook's Section 8 export cell first, so that "
        f"`{ARTIFACT_DIR}/` sits next to `app.py` and contains:"
    )
    for fname in data["missing"]:
        st.markdown(f"- `{fname}` — {REQUIRED_ARTIFACTS[fname]}")
    st.stop()

gs = data["graph_stats"]
mm = data["model_metrics"]
CLASS_NAMES = data["class_names"]

def class_name(class_id):
    """Return the subject name for a numeric class identifier."""
    i = int(class_id)
    return CLASS_NAMES[i] if 0 <= i < len(CLASS_NAMES) else f"Class {i}"

live_badge = '<span class="badge"><span class="dot"></span> Verified notebook results</span>'
class_name_badge = ('<span class="badge">Official arXiv class names</span>' if data.get("class_names_are_real")
                     else '<span class="badge">Generic class labels</span>')

st.markdown(f"""
<div class="hero">
    <h1>OGBN-Arxiv Model Results</h1>
    <p>Coursework dashboard for understanding the graph, comparing GCN and GAT, and inspecting predictions.</p>
    <div class="badge-row">
        {live_badge}
        {class_name_badge}
        <span class="badge">{gs['num_nodes']:,} papers</span>
        <span class="badge">{gs['num_edges']:,} citations</span>
        <span class="badge">{gs['num_classes']} classes</span>
    </div>
</div>
""", unsafe_allow_html=True)

with st.container(border=True):
    st.markdown("#### Start here")
    guide_left, guide_right = st.columns([2, 1])
    guide_left.markdown(
        "**Question:** Can citation links and paper features predict each paper's subject area?  \n"
        f"**Result:** GCN achieved **{mm['GCN']['test']['accuracy']*100:.2f}%** test accuracy, "
        f"compared with **{mm['GAT']['test']['accuracy']*100:.2f}%** for GAT."
    )
    guide_right.markdown(
        "**Suggested path**  \n"
        "1. Overview  \n2. Models  \n3. Predictions  \n4. Explanations"
    )

# Sidebar
with st.sidebar:
    st.markdown("### View settings")
    split_choice = st.selectbox(
        "Metrics split",
        ["test", "validation"],
        index=0,
        help="Use test for final performance and validation for model-selection performance.",
    )
    st.markdown("---")
    st.markdown("### How to read this")
    st.markdown(
        "- **Overview:** what the graph contains\n"
        "- **Models:** which model performed better\n"
        "- **Predictions:** inspect individual results\n"
        "- **Embeddings:** see learned clusters\n"
        "- **Explanations:** understand model evidence\n"
        "- **Extra work:** bonus experiments\n"
        "- **Live test:** run saved models"
    )
    st.markdown("---")
    st.success("Real coursework artifacts loaded")
    with st.expander("Technical data source"):
        st.code(ARTIFACT_DIR, language=None)

tab_overview, tab_perf, tab_predict, tab_embed, tab_explain, tab_bonus, tab_live = st.tabs(
    ["1  Overview", "2  Models", "3  Predictions", "4  Embeddings",
     "5  Explanations", "6  Extra work", "7  Live test"]
)

def metric_card(label, value, delta=None, delta_positive=True, col=None):
    target = col or st
    target.metric(
        label=label,
        value=value,
        delta=delta,
        delta_color="normal" if delta_positive else "inverse",
        border=True,
    )


# Graph overview
with tab_overview:
    st.markdown('<div class="section-title"><span class="bar"></span>Graph Statistics</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    metric_card("Total Papers (Nodes)", f"{gs['num_nodes']:,}", col=c1)
    metric_card("Citation Edges", f"{gs['num_edges']:,}", col=c2)
    metric_card("Feature Dimension", gs["num_features"], col=c3)
    metric_card("Subject Areas", gs["num_classes"], col=c4)

    with st.expander("More graph statistics"):
        c5, c6, c7, c8 = st.columns(4)
        metric_card("Graph Density", f"{gs['density']*100:.5f}%", col=c5)
        metric_card("Average In-Degree", f"{gs['avg_in_degree']:.2f}", col=c6)
        metric_card("Maximum In-Degree", f"{gs['max_in_degree']:,}", col=c7)
        metric_card("Largest Component", f"{gs['largest_component_pct']:.1f}%", col=c8)
        st.caption("In-degree is the number of incoming citation links recorded for a paper.")

    st.markdown('<div class="section-title"><span class="bar"></span>Train / Validation / Test Split</div>', unsafe_allow_html=True)
    split_df = pd.DataFrame({
        "Split": ["Train", "Validation", "Test"],
        "Nodes": [gs["train_size"], gs["valid_size"], gs["test_size"]],
    })
    fig_split = px.pie(
        split_df, names="Split", values="Nodes", hole=0.62,
        color="Split", color_discrete_map={"Train": "#7C5CFF", "Validation": "#22D3EE", "Test": "#F472B6"},
    )
    fig_split.update_traces(textinfo="percent+label", pull=[0.02, 0.02, 0.02],
                             marker=dict(line=dict(color="#05070f", width=2)))
    fig_split.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#172033", showlegend=True, height=360,
        annotations=[dict(text=f"{gs['num_nodes']:,}<br>papers", x=0.5, y=0.5, font_size=15, showarrow=False)],
    )
    colA, colB = st.columns([1.1, 1])
    colA.plotly_chart(fig_split, width="stretch")
    with colB:
        # Native containers keep widgets inside the visible card.
        with st.container(border=True):
            st.markdown("##### 🧭 About the split")
            st.write(
                "OGBN-Arxiv uses a **temporal split**: training papers were published earliest, "
                "test papers are the most recent. The temporal split evaluates how well models "
                "trained primarily on older papers generalize to newer papers."
            )
            st.write(f"- **Train:** {gs['train_size']:,} papers ({gs['train_size']/gs['num_nodes']*100:.1f}%)")
            st.write(f"- **Validation:** {gs['valid_size']:,} papers ({gs['valid_size']/gs['num_nodes']*100:.1f}%)")
            st.write(f"- **Test:** {gs['test_size']:,} papers ({gs['test_size']/gs['num_nodes']*100:.1f}%)")

    st.markdown('<div class="section-title"><span class="bar"></span>Sub-graph Snapshot</div>', unsafe_allow_html=True)
    img_path = os.path.join(ARTIFACT_DIR, "sample_subgraph.png")
    if os.path.exists(img_path):
        st.image(img_path, caption="A 2-hop ego-network sampled from the citation graph (from the notebook, Section 2.3).", width="stretch")
    else:
        st.info("Run the notebook's Section 2.3 to generate a real sub-graph snapshot here.")

    st.markdown('<div class="section-title"><span class="bar"></span>Degree &amp; Connectivity</div>', unsafe_allow_html=True)
    dg1, dg2 = st.columns(2)
    with dg1:
        deg_img = os.path.join(ARTIFACT_DIR, "degree_distribution.png")
        if os.path.exists(deg_img):
            st.image(deg_img, caption="In-degree and out-degree distributions (notebook Section 2.5).", width="stretch")
        else:
            st.info("Run the notebook to generate the degree distribution chart.")
    with dg2:
        cc_img = os.path.join(ARTIFACT_DIR, "connected_components.png")
        if os.path.exists(cc_img):
            st.image(cc_img, caption="Weak and strong connected-component analysis (notebook Section 2.7).", width="stretch")
        else:
            st.info("Run the notebook to generate the connected components chart.")

# Model performance
with tab_perf:
    st.markdown('<div class="section-title"><span class="bar"></span>Head-to-Head Metrics</div>', unsafe_allow_html=True)

    gcn_m = mm["GCN"][split_choice]
    gat_m = mm["GAT"][split_choice]

    cols = st.columns(4)
    labels = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1"]
    keys = ["accuracy", "precision", "recall", "f1"]
    for c, lab, k in zip(cols, labels, keys):
        winner = "GAT" if gat_m[k] >= gcn_m[k] else "GCN"
        diff = abs(gat_m[k] - gcn_m[k]) * 100
        metric_card(
            lab,
            f"{max(gcn_m[k], gat_m[k])*100:.2f}%",
            delta=f"{winner} leads by {diff:.2f} points",
            delta_positive=True,
            col=c,
        )
        c.caption(f"GCN {gcn_m[k]*100:.2f}% · GAT {gat_m[k]*100:.2f}%")

    st.markdown("")
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(name="GCN", x=labels, y=[gcn_m[k] for k in keys],
                              marker_color="#7C5CFF", marker_line_width=0))
    fig_bar.add_trace(go.Bar(name="GAT", x=labels, y=[gat_m[k] for k in keys],
                              marker_color="#22D3EE", marker_line_width=0))
    fig_bar.update_layout(
        barmode="group", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#172033", height=400, yaxis=dict(range=[0, 1], gridcolor="#e6eaf0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"),
        title=f"GCN vs GAT — {split_choice.capitalize()} Set Performance",
    )
    st.plotly_chart(fig_bar, width="stretch")

    best_model = "GCN" if gcn_m["accuracy"] > gat_m["accuracy"] else "GAT"
    st.info(
        f"On the {split_choice} split, {best_model} achieved the higher overall accuracy. "
        "Macro-F1 should also be considered because OGBN-Arxiv contains multiple classes "
        "with different frequencies."
    )

    # Keep the headline result fixed to the test split.
    gcn_test, gat_test = mm["GCN"]["test"], mm["GAT"]["test"]
    test_keys = ["accuracy", "precision", "recall", "f1"]
    gcn_wins = sum(1 for k in test_keys if gcn_test[k] >= gat_test[k])
    overall_best = "GCN" if gcn_wins >= len(test_keys) / 2 else "GAT"
    consistent = gcn_wins in (0, len(test_keys))

    if consistent:
        st.success(
            f"**Best-performing model: {overall_best}.** {overall_best} achieves higher test-set "
            f"accuracy, macro precision, macro recall, and macro F1 than "
            f"{'GAT' if overall_best == 'GCN' else 'GCN'} under the current configuration — a "
            "consistent result across every metric, not just one."
        )
    else:
        st.success(
            f"**Best-performing model (by accuracy): {overall_best}.** The two models don't agree "
            "on every metric — check the table above to see where each one leads before treating "
            "this as a clean win."
        )

    colL, colR = st.columns(2)
    with colL:
        st.markdown('<div class="section-title"><span class="bar"></span>Training Curves</div>', unsafe_allow_html=True)
        fig_curve = go.Figure()
        fig_curve.add_trace(go.Scatter(x=data["gcn_log"]["epoch"], y=data["gcn_log"]["val_acc"],
                                        name="GCN val acc", line=dict(color="#7C5CFF", width=3)))
        fig_curve.add_trace(go.Scatter(x=data["gat_log"]["epoch"], y=data["gat_log"]["val_acc"],
                                        name="GAT val acc", line=dict(color="#22D3EE", width=3)))
        fig_curve.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#172033",
            height=340, xaxis_title="Epoch", yaxis_title="Validation Accuracy",
            xaxis=dict(gridcolor="#e6eaf0"), yaxis=dict(gridcolor="#e6eaf0"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"),
        )
        st.plotly_chart(fig_curve, width="stretch")

    with colR:
        st.markdown('<div class="section-title"><span class="bar"></span>Efficiency</div>', unsafe_allow_html=True)
        eff_df = pd.DataFrame({
            "Model": ["GCN", "GAT"],
            "Params": [mm["GCN"]["num_params"], mm["GAT"]["num_params"]],
            "Train Time (s)": [mm["GCN"]["train_time_sec"], mm["GAT"]["train_time_sec"]],
        })
        fig_eff = px.bar(eff_df, x="Model", y="Train Time (s)", color="Model",
                          color_discrete_map={"GCN": "#7C5CFF", "GAT": "#22D3EE"}, text="Params")
        fig_eff.update_traces(texttemplate="%{text:,} params", textposition="outside")
        fig_eff.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#172033",
            height=340, showlegend=False, yaxis=dict(gridcolor="#e6eaf0"),
        )
        st.plotly_chart(fig_eff, width="stretch")

    with st.expander("📎 Static notebook exports (model_comparison.png, training_curves.png)"):
        se1, se2 = st.columns(2)
        with se1:
            cmp_img = os.path.join(ARTIFACT_DIR, "model_comparison.png")
            if os.path.exists(cmp_img):
                st.image(cmp_img, caption="Notebook-generated GCN vs GAT comparison (matplotlib export).", width="stretch")
            else:
                st.info("Run the notebook to generate model_comparison.png.")
        with se2:
            curves_img = os.path.join(ARTIFACT_DIR, "training_curves.png")
            if os.path.exists(curves_img):
                st.image(curves_img, caption="Notebook-generated training curves (matplotlib export).", width="stretch")
            else:
                st.info("Run the notebook to generate training_curves.png.")

# Predictions
with tab_predict:
    st.markdown('<div class="section-title"><span class="bar"></span>Explore Predictions</div>', unsafe_allow_html=True)
    st.caption(
        "Filter the saved test examples, compare each model with the true subject, and inspect "
        "confidence. Use the Live test tab when you want to execute the models again."
    )
    preds = data["predictions"]

    f1, f2, f3 = st.columns([1, 1, 1])
    correctness_filter = f1.selectbox("Filter", ["All papers", "Correct only", "Misclassified only"])
    model_for_filter = f2.selectbox("Judge correctness by", ["GCN", "GAT"])
    n_rows = f3.slider("Rows to show", 10, min(200, len(preds)), 25)

    col_correct = f"{model_for_filter.lower()}_correct"
    view = preds.copy()
    if correctness_filter == "Correct only":
        view = view[view[col_correct]]
    elif correctness_filter == "Misclassified only":
        view = view[~view[col_correct]]

    acc_shown = view[col_correct].mean() if len(view) else 0
    m1, m2, m3 = st.columns(3)
    metric_card("Papers shown", f"{len(view):,}", col=m1)
    metric_card("Accuracy for selected rows", f"{acc_shown*100:.1f}%", col=m2)
    metric_card("Avg. confidence", f"{view[f'{model_for_filter.lower()}_confidence'].mean()*100:.1f}%", col=m3)

    display_df = pd.DataFrame({
        "Paper ID": view["paper_id"].head(n_rows),
        "True Subject": view["true_class"].head(n_rows).apply(class_name),
        "GCN Prediction": view["gcn_prediction"].head(n_rows).apply(class_name),
        "GCN Confidence": view["gcn_confidence"].head(n_rows),
        "GCN ✓": view["gcn_correct"].head(n_rows).map({True: "✅", False: "❌"}),
        "GAT Prediction": view["gat_prediction"].head(n_rows).apply(class_name),
        "GAT Confidence": view["gat_confidence"].head(n_rows),
        "GAT ✓": view["gat_correct"].head(n_rows).map({True: "✅", False: "❌"}),
    })

    st.dataframe(
        display_df.style.background_gradient(
            subset=[f"{model_for_filter.upper()} Confidence"], cmap="Purples"
        ),
        width="stretch", height=420, hide_index=True,
    )

    st.markdown('<div class="section-title"><span class="bar"></span>Inspect a Single Paper Prediction</div>', unsafe_allow_html=True)
    if len(preds):
        pick = st.selectbox("Pick a paper ID from the sample", preds["paper_id"].tolist())
        row = preds[preds["paper_id"] == pick].iloc[0]

        cc1, cc2 = st.columns(2)
        for c, mdl in zip([cc1, cc2], ["gcn", "gat"]):
            pred_class = int(row[f"{mdl}_prediction"])
            true_class = int(row["true_class"])
            conf = float(row[f"{mdl}_confidence"])
            correct = pred_class == true_class
            badge_color = "#34D399" if correct else "#F472B6"
            c.markdown(f"""
            <div class="glass">
                <div class="metric-label">{mdl.upper()} prediction</div>
                <div class="metric-value" style="color:{badge_color}">{class_name(pred_class)}</div>
                <div style="color:var(--text-dim); font-size:0.85rem; margin:0.3rem 0 0.5rem 0;">
                    True label: <b>{class_name(true_class)}</b> &nbsp;•&nbsp; {"✅ Correct" if correct else "❌ Incorrect"}
                </div>
                <div class="conf-bar-track"><div class="conf-bar-fill" style="width:{conf*100:.0f}%"></div></div>
                <div style="text-align:right; font-size:0.78rem; color:var(--text-dim); margin-top:0.25rem;">{conf*100:.1f}% confidence</div>
            </div>
            """, unsafe_allow_html=True)

# Embeddings
with tab_embed:
    st.markdown('<div class="section-title"><span class="bar"></span>Learned Node Embeddings</div>', unsafe_allow_html=True)
    st.caption("Each point is a paper. Points that cluster together were mapped to similar representations by the GNN — colour shows the true subject area.")

    proj_choice = st.radio("Projection", ["t-SNE", "PCA"], horizontal=True)
    if proj_choice == "t-SNE":
        st.caption("t-SNE provides a nonlinear two-dimensional visualization of learned node embeddings, allowing qualitative inspection of class separation and overlap.")
    else:
        st.caption("PCA provides a linear dimensionality-reduction baseline for comparison with t-SNE.")
    xcol, ycol = ("tsne_x", "tsne_y") if proj_choice == "t-SNE" else ("pca_x", "pca_y")

    emb = data["embeddings"].copy()
    emb["class_name"] = emb["true_class"].apply(class_name)

    available_classes = sorted(emb["class_name"].unique())
    default_classes = emb["class_name"].value_counts().head(8).index.tolist()
    selected_classes = st.multiselect(
        "Subject areas to display",
        options=available_classes,
        default=default_classes,
        help="The eight most common sampled classes are selected initially to keep the chart readable.",
    )
    filtered_emb = emb[emb["class_name"].isin(selected_classes)] if selected_classes else emb

    fig_emb = px.scatter(
        filtered_emb, x=xcol, y=ycol, color="class_name",
        opacity=0.75, height=560,
        color_discrete_sequence=px.colors.qualitative.Light24,
    )
    fig_emb.update_traces(marker=dict(size=6, line=dict(width=0)))
    fig_emb.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#172033",
        legend=dict(font=dict(size=9)), xaxis=dict(gridcolor="#e6eaf0", zeroline=False),
        yaxis=dict(gridcolor="#e6eaf0", zeroline=False),
        title=f"{proj_choice} Projection of Node Embeddings ({len(filtered_emb):,} of {len(emb):,} sampled papers shown)",
    )
    st.plotly_chart(fig_emb, width="stretch")

# Explainability
with tab_explain:
    st.markdown('<div class="section-title"><span class="bar"></span>Why did the model predict that?</div>', unsafe_allow_html=True)

    e1, e2 = st.columns(2)
    with e1:
        with st.container(border=True):
            st.markdown("##### 🎯 GAT Attention Weights")
            attn_img = os.path.join(ARTIFACT_DIR, "attention_weights.png")
            if os.path.exists(attn_img):
                st.image(attn_img, width="stretch", caption="Correctly-classified test paper.")
            else:
                st.info("Run notebook Section 7.2 to generate the real attention-weight chart for a chosen paper.")
            st.caption("The attention weights indicate which neighbouring papers received greater weighting during message aggregation — a higher bar does not by itself prove that citation caused the prediction, which is why it's cross-checked against the neighbourhood-influence ablation on the right.")

            attn_table = data.get("attention_table")
            if attn_table is not None and not attn_table.empty:
                st.markdown("Top attended neighbours:")
                st.dataframe(attn_table.head(10), width="stretch", hide_index=True)

            attn_incorrect_img = os.path.join(ARTIFACT_DIR, "attention_weights_incorrect.png")
            if os.path.exists(attn_incorrect_img):
                st.markdown("**Contrasting example — a misclassified test paper:**")
                st.image(attn_incorrect_img, width="stretch", caption="Incorrectly-classified test paper (notebook Section 7.2).")

            attn_quality = data.get("attention_quality")
            if attn_quality:
                st.markdown("**Does high attention actually mean 'same subject area'?**")
                aq1, aq2 = st.columns(2)
                aq1.metric("Top-attention edges", f"{attn_quality['top_attention_same_class_rate']*100:.1f}%",
                           help=f"Same-class rate among the top {attn_quality['top_k']} highest-attention citation edges (notebook Section 7.2b).")
                aq2.metric("Random baseline", f"{attn_quality['random_baseline_same_class_rate']*100:.1f}%",
                           help="Same-class rate among a random sample of the same size.")

    with e2:
        with st.container(border=True):
            st.markdown("##### 🕸️ Neighbourhood Influence")
            st.write(
                "Removing a neighbour and re-running the model shows how much that citation "
                "shifts the predicted class probability — the notebook's Section 7.3 ranks each "
                "neighbour of a chosen paper by this influence score, which sanity-checks the "
                "attention-based explanation with a separate edge-ablation method."
            )
            influence_df = data.get("neighbourhood_influence")
            if influence_df is not None and not influence_df.empty:
                infl_view = influence_df.sort_values("prob_drop", ascending=False).head(15).copy()
                x_col = "prob_drop" if "prob_drop" in infl_view.columns else infl_view.columns[-1]
                raw_y_col = "removed_neighbour" if "removed_neighbour" in infl_view.columns else infl_view.columns[0]
                # String labels prevent Plotly treating node IDs as a continuous axis.
                infl_view["neighbour_label"] = "Paper #" + infl_view[raw_y_col].astype(str)
                if "target_node" in infl_view.columns:
                    st.caption(f"Explaining paper #{int(infl_view['target_node'].iloc[0])} "
                               f"(predicted class {int(infl_view['target_predicted_class'].iloc[0])}).")
                fig_infl = px.bar(infl_view, x=x_col, y="neighbour_label", orientation="h",
                                   color=x_col, color_continuous_scale=["#22D3EE", "#7C5CFF", "#F472B6"])
                fig_infl.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                        font_color="#172033", height=340, coloraxis_showscale=False,
                                        yaxis=dict(autorange="reversed", type="category"),
                                        xaxis_title="Probability drop when neighbour is removed")
                st.plotly_chart(fig_infl, width="stretch")
                st.caption("Real ranking from notebook Section 7.3's `influence_df`, exported to `neighbourhood_influence.csv`.")
            else:
                st.info("Run notebook Section 7.3 to generate real neighbourhood influence results (`neighbourhood_influence.csv`).")

    st.markdown('<div class="section-title"><span class="bar"></span>t-SNE / PCA (from notebook)</div>', unsafe_allow_html=True)
    tsne_img = os.path.join(ARTIFACT_DIR, "tsne_embeddings.png")
    if os.path.exists(tsne_img):
        st.image(tsne_img, width="stretch", caption="Static export from the notebook (Section 7.1) — the Embedding Space tab above is the interactive version.")

# Extended experiments
with tab_bonus:
    st.caption(
        "The two sections below are supplementary experiments that go beyond the core GCN vs GAT "
        "comparison — exploring one additional architecture and one additional training strategy. "
        "They are not a replacement for the primary GCN/GAT results in the Model Performance tab."
    )

    st.markdown('<div class="section-title"><span class="bar"></span>Alternative Architecture — Graph Transformer</div>', unsafe_allow_html=True)
    st.write(
        "A 3-layer Graph Transformer (PyTorch Geometric's `TransformerConv`) is trained with the "
        "same protocol as the GCN/GAT baselines — same depth, comparable hidden width, same "
        "optimizer and training loop. This controls model scale and training budget, while each "
        "architecture retains its own layer operations."
    )
    bonus_img = os.path.join(ARTIFACT_DIR, "extended_three_way_comparison.png")
    if not os.path.exists(bonus_img):
        bonus_img = os.path.join(ARTIFACT_DIR, "additional_three_way_comparison.png")
    if os.path.exists(bonus_img):
        st.image(bonus_img, width="stretch",
                 caption="GCN vs GAT vs Graph Transformer — test-set performance.")
    else:
        st.info("Run the notebook's Graph Transformer section to generate the three-way comparison chart.")

    st.markdown('<div class="section-title"><span class="bar"></span>Alternative Training Strategy — Self-Supervised Pre-training</div>', unsafe_allow_html=True)
    ssl_summary = data.get("ssl_summary")
    if ssl_summary:
        st.write(
            f"A GCN encoder is pre-trained to reconstruct masked node features "
            f"({ssl_summary['mask_rate']*100:.0f}% of nodes masked per epoch, "
            f"{ssl_summary['pretrain_epochs_run']} pretraining epochs) using only the graph "
            "structure — no labels involved. The encoder is then fine-tuned on the official "
            "labelled split and compared against an identical architecture trained from scratch, "
            "isolating the effect of the pretext task from the architecture itself."
        )

        compare_df = pd.DataFrame([
            {"Model": "GCN (primary)", "Test Accuracy": mm["GCN"]["test"]["accuracy"], "Test Macro F1": mm["GCN"]["test"]["f1"]},
            {"Model": "GAT (primary)", "Test Accuracy": mm["GAT"]["test"]["accuracy"], "Test Macro F1": mm["GAT"]["test"]["f1"]},
            {"Model": "GCN, SSL-pretrained (extended)", "Test Accuracy": ssl_summary["ssl_pretrained_test_accuracy"], "Test Macro F1": ssl_summary["ssl_pretrained_test_f1"]},
            {"Model": "GCN, from scratch (extended)", "Test Accuracy": ssl_summary["from_scratch_test_accuracy"], "Test Macro F1": ssl_summary["from_scratch_test_f1"]},
        ])
        compare_df["Test Accuracy"] = (compare_df["Test Accuracy"] * 100).round(2).astype(str) + "%"
        compare_df["Test Macro F1"] = (compare_df["Test Macro F1"] * 100).round(2).astype(str) + "%"
        st.dataframe(compare_df, width="stretch", hide_index=True)

        sc1, sc2, sc3, sc4 = st.columns(4)
        metric_card("SSL-Pretrained Test Accuracy", f"{ssl_summary['ssl_pretrained_test_accuracy']*100:.2f}%", col=sc1)
        metric_card("From-Scratch Test Accuracy", f"{ssl_summary['from_scratch_test_accuracy']*100:.2f}%", col=sc2)
        metric_card("SSL-Pretrained Test Macro F1", f"{ssl_summary['ssl_pretrained_test_f1']*100:.2f}%", col=sc3)
        metric_card("From-Scratch Test Macro F1", f"{ssl_summary['from_scratch_test_f1']*100:.2f}%", col=sc4)

        acc_delta = ssl_summary["test_accuracy_improvement_pp"]
        f1_delta = ssl_summary["test_f1_improvement"]
        acc_word = "improvement" if acc_delta > 0 else ("decline" if acc_delta < 0 else "no change")
        f1_word = "increased" if f1_delta > 0 else ("reduced" if f1_delta < 0 else "left unchanged")
        if acc_delta > 0 and f1_delta > 0:
            conclusion = "the effect of pretraining was a consistent improvement across both metrics."
        elif acc_delta <= 0 and f1_delta <= 0:
            conclusion = "pretraining did not improve either metric over training from scratch on this run."
        else:
            conclusion = "the effect of pretraining was mixed rather than a consistent improvement."
        st.info(
            f"Self-supervised pretraining produced a small {acc_delta:+.2f} percentage-point "
            f"{acc_word} in test accuracy, but {f1_word} macro-F1 by {abs(f1_delta):.4f}. "
            f"Therefore, {conclusion} Note that even the improved run (**{ssl_summary['ssl_pretrained_test_accuracy']*100:.2f}%** "
            f"accuracy) still trails the primary GCN model (**{mm['GCN']['test']['accuracy']*100:.2f}%**), so this experiment "
            "is best read as a study of the pretraining effect, not a new best-performing model."
        )
    else:
        st.info("Run the notebook's self-supervised pre-training section to generate `ssl_pretraining_summary.json`.")

# Live inference requires PyTorch Geometric, node features and the full edge index.
LIVE_REQUIRED = {
    "node_features.pt": "all node feature vectors (data.x), shape [num_nodes, 128]",
    "edge_index.pt": "the full citation graph edge_index, shape [2, num_edges]",
}

with tab_live:
    st.markdown('<div class="section-title"><span class="bar"></span>Run the Trained Models Live</div>', unsafe_allow_html=True)
    st.write(
        "Everywhere else in this dashboard shows **precomputed** results from `predictions_sample.csv`. "
        "This tab is different: pick any paper in the graph and it runs an **actual forward pass** "
        "through the loaded `gcn_model.pt` / `gat_model.pt` weights, right now, using that paper's real "
        "citation neighbourhood."
    )

    if st.button("Run live inference", type="primary"):
        st.session_state["live_inference_requested"] = True
    if not st.session_state.get("live_inference_requested", False):
        st.info(
            "Live inference is intentionally paused until requested. It processes the full "
            "citation graph and can take time on a CPU; the other dashboard tabs remain instant."
        )
        st.stop()

    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch_geometric.nn import GCNConv, GATConv
        TORCH_OK = True
        torch_import_error = None
    except ImportError as e:
        TORCH_OK = False
        torch_import_error = str(e)

    missing_graph_files = [f for f in LIVE_REQUIRED if not os.path.exists(os.path.join(ARTIFACT_DIR, f))]

    if not TORCH_OK:
        st.warning(
            "**PyTorch / PyTorch Geometric aren't installed**, so this tab can't run real inference.\n\n"
            f"Import error: `{torch_import_error}`\n\n"
            "Install them (CPU build is enough for this):\n"
            "```bash\npip install torch --index-url https://download.pytorch.org/whl/cpu\n"
            "pip install torch_geometric\n```"
        )
    elif missing_graph_files:
        st.warning(
            "**The full graph isn't exported yet**, so a genuine forward pass isn't possible — "
            "`predictions_sample.csv` only stores the *result* of past predictions, not the raw "
            "128-dim features / edges GCN and GAT need to compute a new one. Add this cell to the "
            "notebook (after the `data`/`dataset` object exists) and re-run the export section:"
        )
        st.code(
            "import torch, os\n"
            "os.makedirs('dashboard/artifacts', exist_ok=True)\n"
            "torch.save(data.x, 'dashboard/artifacts/node_features.pt')\n"
            "torch.save(data.edge_index, 'dashboard/artifacts/edge_index.pt')",
            language="python",
        )
        st.caption(f"Missing: {', '.join(missing_graph_files)}")
    else:
        # These definitions must match the saved state dictionaries exactly.
        class GCN(nn.Module):
            def __init__(self, in_dim=128, hidden=256, out_dim=40, dropout=0.5):
                super().__init__()
                self.convs = nn.ModuleList([
                    GCNConv(in_dim, hidden),
                    GCNConv(hidden, hidden),
                    GCNConv(hidden, out_dim),
                ])
                self.bns = nn.ModuleList([nn.BatchNorm1d(hidden), nn.BatchNorm1d(hidden)])
                self.dropout = dropout

            def forward(self, x, edge_index):
                for i in range(2):
                    x = self.convs[i](x, edge_index)
                    x = self.bns[i](x)
                    x = F.relu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
                return self.convs[2](x, edge_index)

        class GAT(nn.Module):
            def __init__(self, in_dim=128, hidden=32, heads=8, out_dim=40, dropout=0.5):
                super().__init__()
                self.convs = nn.ModuleList([
                    GATConv(in_dim, hidden, heads=heads, concat=True),
                    GATConv(hidden * heads, hidden, heads=heads, concat=True),
                    GATConv(hidden * heads, out_dim, heads=1, concat=False),
                ])
                self.dropout = dropout

            def forward(self, x, edge_index):
                for i in range(2):
                    x = self.convs[i](x, edge_index)
                    x = F.elu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
                return self.convs[2](x, edge_index)

        @st.cache_resource(show_spinner="Loading graph + trained weights into memory (once per session)…")
        def load_live_inference_state():
            load_options = {"map_location": "cpu", "weights_only": True}
            x = torch.load(os.path.join(ARTIFACT_DIR, "node_features.pt"), **load_options)
            edge_index = torch.load(os.path.join(ARTIFACT_DIR, "edge_index.pt"), **load_options)

            gcn = GCN()
            gcn.load_state_dict(
                torch.load(os.path.join(ARTIFACT_DIR, "gcn_model.pt"), **load_options),
                strict=True,
            )
            gcn.eval()

            gat = GAT()
            gat.load_state_dict(
                torch.load(os.path.join(ARTIFACT_DIR, "gat_model.pt"), **load_options),
                strict=True,
            )
            gat.eval()

            with torch.no_grad():
                gcn_logits_all = gcn(x, edge_index)
                gat_logits_all = gat(x, edge_index)
            return x, edge_index, gcn_logits_all, gat_logits_all

        try:
            x, edge_index, gcn_logits_all, gat_logits_all = load_live_inference_state()
            st.success(
                f"Real forward pass complete over the full graph ({x.shape[0]:,} nodes, "
                f"{edge_index.shape[1]:,} edges) using the exact weights in `gcn_model.pt` / `gat_model.pt`.",
                icon="✅",
            )

            all_paper_ids = list(range(x.shape[0]))
            sample_ids = data["predictions"]["paper_id"].tolist()
            source = st.radio(
                "Pick from", ["Sample papers (has a known true label to check against)", "Any node ID in the graph"],
                horizontal=True,
            )
            if source.startswith("Sample"):
                node_id = st.selectbox("Paper ID", sample_ids)
            else:
                node_id = st.number_input("Paper (node) ID", min_value=0, max_value=x.shape[0] - 1, value=0, step=1)

            node_id = int(node_id)
            gcn_probs = F.softmax(gcn_logits_all[node_id], dim=0)
            gat_probs = F.softmax(gat_logits_all[node_id], dim=0)
            gcn_pred = int(torch.argmax(gcn_probs).item())
            gat_pred = int(torch.argmax(gat_probs).item())
            num_neighbours = int(((edge_index[0] == node_id) | (edge_index[1] == node_id)).sum().item())

            lc1, lc2, lc3 = st.columns(3)
            lc1.metric("Citation neighbours used", f"{num_neighbours}")
            lc2.metric("GCN prediction", class_name(gcn_pred), f"{gcn_probs[gcn_pred]*100:.1f}% confidence")
            lc3.metric("GAT prediction", class_name(gat_pred), f"{gat_probs[gat_pred]*100:.1f}% confidence")

            row = data["predictions"][data["predictions"]["paper_id"] == node_id]
            if not row.empty:
                true_class = int(row.iloc[0]["true_class"])
                st.caption(
                    f"True label: **{class_name(true_class)}**  ·  Precomputed CSV said GCN → "
                    f"**{class_name(int(row.iloc[0]['gcn_prediction']))}**, GAT → "
                    f"**{class_name(int(row.iloc[0]['gat_prediction']))}** — compare that against the live "
                    "numbers above; they should match, since it's the same weights and the same graph."
                )

            top5 = pd.DataFrame({
                "Subject": CLASS_NAMES,
                "GCN probability": gcn_probs.detach().numpy(),
                "GAT probability": gat_probs.detach().numpy(),
            }).sort_values("GCN probability", ascending=False).head(5)
            fig_live = px.bar(top5, x="Subject", y=["GCN probability", "GAT probability"], barmode="group")
            fig_live.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                    font_color="#172033", height=340, legend_title_text="")
            st.plotly_chart(fig_live, width="stretch")

        except Exception as e:
            st.error(
                "Loading the graph / running inference failed — the exported tensors likely don't "
                f"match the architecture reconstructed from the `.pt` state_dict. Raw error: `{e}`"
            )

st.markdown("""
<div style="text-align:center; color:var(--text-dim); font-size:0.8rem; margin-top:2.5rem; padding-bottom:1rem;">
    Graph Intelligence Dashboard · OGBN-Arxiv Node Classification
</div>
""", unsafe_allow_html=True)
