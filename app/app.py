import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# ── Config page ──
st.set_page_config(
    page_title="Bankia — Risque de Défaut",
    page_icon="🏦",
    layout="wide"
)

# ── Couleurs Bankia ──
VIOLET       = "#7B4FD4"
VIOLET_LIGHT = "#9B7FE8"
VIOLET_DARK  = "#6B3FC4"
AMBER        = "#F59E0B"
BG           = "#f8f8fb"
CARD         = "#ffffff"

# ── Style CSS ──
st.markdown(f"""
<style>
    [data-testid="stSidebar"] {{
        background-color: {VIOLET_DARK};
    }}
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] span {{
        color: white !important;
    }}
    [data-testid="stSidebar"] input {{
        color: #1a1a1a !important;
        background-color: white !important;
    }}
    .stMetric {{
        background-color: {CARD};
        border: 1px solid #e0d7f7;
        border-radius: 12px;
        padding: 12px;
    }}
    .risk-badge {{
        font-size: 1.2em;
        font-weight: bold;
        padding: 12px 24px;
        border-radius: 10px;
        text-align: center;
        margin-top: 10px;
    }}
    .low    {{ background-color: #d4edda; color: #155724; }}
    .medium {{ background-color: #fff3cd; color: #856404; }}
    .high   {{ background-color: #f8d7da; color: #721c24; }}
    h1, h2, h3 {{ color: {VIOLET}; }}
    .stButton > button {{
        background-color: {AMBER};
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 8px;
    }}
</style>
""", unsafe_allow_html=True)

# ── Chargement ──
MODEL_PATH  = "/Users/ivanwinograd/IdeaProjects/Personnel/MLOps/Projet-MLops/src/Models/model.pkl"
SCALER_PATH = "/Users/ivanwinograd/IdeaProjects/Personnel/MLOps/Projet-MLops/src/Models/scaler.pkl"
DATA_PATH   = "/Users/ivanwinograd/IdeaProjects/Personnel/MLOps/Projet-MLops/Data/Loan_Data.csv"

model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df       = load_data()
features = ["credit_lines_outstanding", "loan_amt_outstanding",
            "total_debt_outstanding", "income", "years_employed", "fico_score"]
labels   = {
    "credit_lines_outstanding": "Lignes de crédit",
    "loan_amt_outstanding"    : "Prêt restant",
    "total_debt_outstanding"  : "Dette totale",
    "income"                  : "Revenu",
    "years_employed"          : "Années d'emploi",
    "fico_score"              : "FICO Score"
}
moyennes = df.groupby("default")[features].mean()
pop_mean = df[features].mean()

# ── Sidebar ──
with st.sidebar:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    st.header("🧾 Informations Client")

    credit_lines_outstanding = st.number_input("Lignes de crédit actives",    min_value=0, value=2,     help="Nombre de lignes de crédit en cours")
    loan_amt_outstanding     = st.number_input("Montant du prêt restant (€)", min_value=0, value=10000, help="Capital restant dû sur le prêt")
    total_debt_outstanding   = st.number_input("Dette totale (€)",            min_value=0, value=15000, help="Ensemble des dettes du client")
    income                   = st.number_input("Revenu annuel (€)",           min_value=0, value=50000, help="Revenu annuel brut")
    years_employed           = st.number_input("Années d'emploi",             min_value=0, max_value=50, value=5, help="Ancienneté dans l'emploi actuel")
    fico_score               = st.number_input("FICO Score",                  min_value=300, max_value=850, value=650, help="Score de crédit (300 = très risqué, 850 = excellent)")

    st.markdown("---")
    predict_btn = st.button("🔍 Prédire le risque", use_container_width=True)

# ── Header ──
st.markdown("""
    <div style="display: flex; align-items: center; gap: 20px; padding: 10px 0 20px 0;">
        <svg width="220" height="90" viewBox="0 0 320 90" xmlns="http://www.w3.org/2000/svg">
          <rect x="10" y="5" width="65" height="80" rx="8" fill="#4a2a9a"/>
          <rect x="20" y="22" width="18" height="13" rx="4" fill="#F59E0B"/>
          <rect x="20" y="42" width="40" height="4" rx="2" fill="#9B7FE8"/>
          <rect x="20" y="50" width="28" height="4" rx="2" fill="#9B7FE8"/>
          <text x="88" y="55" font-family="Arial, sans-serif" font-size="48" font-weight="700" fill="#7B4FD4">Bank<tspan fill="#F59E0B">ia</tspan></text>
          <text x="88" y="75" font-family="Arial, sans-serif" font-size="11" font-weight="400" fill="#9B7FE8" letter-spacing="2">VOTRE BANQUE, PARTOUT, TOUJOURS</text>
        </svg>
        <div style="flex: 1; text-align: center;">
            <div style="font-size: 2.2em; font-weight: 700; color: #1a1a2e;">Prédiction du Risque de Défaut</div>
            <div style="font-size: 0.95em; color: #6b7280; margin-top: 4px;">Outil d'aide à la décision pour l'analyse du risque crédit client.</div>
        </div>
    </div>
""", unsafe_allow_html=True)
st.markdown("---")

client = {
    "customer_id"              : 0,
    "credit_lines_outstanding" : credit_lines_outstanding,
    "loan_amt_outstanding"     : loan_amt_outstanding,
    "total_debt_outstanding"   : total_debt_outstanding,
    "income"                   : income,
    "years_employed"           : years_employed,
    "fico_score"               : fico_score
}
data        = pd.DataFrame([client])
data_scaled = scaler.transform(data)

if predict_btn:
    proba = model.predict_proba(data_scaled)[0][1]

    # ── Feu tricolore ──
    if proba < 0.3:
        niveau, css_class, emoji = "FAIBLE", "low", "🟢"
    elif proba < 0.6:
        niveau, css_class, emoji = "MODÉRÉ", "medium", "🟡"
    else:
        niveau, css_class, emoji = "ÉLEVÉ", "high", "🔴"

    # ── Percentile ──
    df_scored = df[features].copy()
    df_scored.insert(0, "customer_id", 0)
    scores_pop = model.predict_proba(scaler.transform(df_scored))[:, 1]
    percentile = int(np.mean(scores_pop <= proba) * 100)

    # ── Métriques + badge ──
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Probabilité de défaut", f"{proba:.1%}")
        st.markdown(f'<div class="risk-badge {css_class}">{emoji} Risque {niveau}</div>', unsafe_allow_html=True)
    with col2:
        st.metric("📈 Percentile de risque", f"{percentile}e percentile")
    with col3:
        st.metric("🎯 Seuil de décision", "50%")

    st.markdown("---")

    # ── Jauge ──
    fig_j, ax_j = plt.subplots(figsize=(8, 0.8))
    fig_j.patch.set_facecolor(BG)
    ax_j.set_facecolor(BG)

    color = "#2ecc71" if proba < 0.3 else AMBER if proba < 0.6 else "#e74c3c"
    ax_j.barh([""], [proba],     color=color,    height=0.5, zorder=2)
    ax_j.barh([""], [1 - proba], left=[proba],   color="#e0d7f7", height=0.5, zorder=2)
    ax_j.axvline(0.3, color="#2ecc71", linestyle=":", linewidth=1.2, zorder=3)
    ax_j.axvline(0.6, color=AMBER,     linestyle=":", linewidth=1.2, zorder=3)
    ax_j.axvline(0.5, color=VIOLET,    linestyle="--", linewidth=1,  zorder=3, alpha=0.5)
    ax_j.set_xlim(0, 1)
    ax_j.set_xticks([0, 0.3, 0.5, 0.6, 1])
    ax_j.set_xticklabels(["0%", "30%", "50%", "60%", "100%"], color=VIOLET, fontsize=9)
    ax_j.set_yticks([])
    for spine in ax_j.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    st.pyplot(fig_j)

    st.markdown("---")

    # ── Graphique + Tableau ──
    col_graph, col_table = st.columns([2, 1])

    with col_graph:
        st.subheader("📊 Profil client vs population")
        st.caption("Valeurs normalisées en % par rapport à la moyenne globale (100% = moyenne)")

        client_norm     = [client[f] / pop_mean[f] * 100 for f in features]
        default_norm    = [moyennes.loc[1, f] / pop_mean[f] * 100 for f in features]
        no_default_norm = [moyennes.loc[0, f] / pop_mean[f] * 100 for f in features]

        x     = np.arange(len(features))
        width = 0.25

        fig, ax = plt.subplots(figsize=(9, 4))
        fig.patch.set_facecolor(CARD)
        ax.set_facecolor(CARD)

        ax.bar(x - width, no_default_norm, width, label="Non-défaillants", color=VIOLET_LIGHT, alpha=0.9)
        ax.bar(x,         default_norm,    width, label="Défaillants",     color="#e74c3c",    alpha=0.85)
        ax.bar(x + width, client_norm,     width, label="Ce client",       color=AMBER,        alpha=0.95)
        ax.axhline(100, color=VIOLET, linestyle="--", linewidth=0.8, alpha=0.5, label="Moyenne")

        ax.set_xticks(x)
        ax.set_xticklabels([labels[f] for f in features], fontsize=9, color=VIOLET)
        ax.set_ylabel("% par rapport à la moyenne", color=VIOLET)
        ax.tick_params(colors=VIOLET)
        ax.legend(fontsize=8, facecolor=CARD, labelcolor=VIOLET)
        ax.set_title("Comparaison normalisée du profil client", fontsize=11,
                     fontweight="bold", color=VIOLET)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color("#e0d7f7")

        plt.tight_layout()
        st.pyplot(fig)

    with col_table:
        st.subheader("📋 Récapitulatif")
        recap = pd.DataFrame({
            "Indicateur"  : [labels[f] for f in features],
            "Client"      : [client[f] for f in features],
            "Moy. défaut" : [round(moyennes.loc[1, f], 1) for f in features],
        })
        st.dataframe(recap, hide_index=True, use_container_width=True)