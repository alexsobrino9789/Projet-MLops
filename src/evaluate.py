import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)

# ── Import du preprocessing ──
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import run_preprocessing

# 🔹 CONFIGURATION
MODEL_PATH  = "Models/model.pkl"
MODELS_DIR  = "Models"


# ─────────────────────────────────────────────────────────────────────────────

def charger_modele():
    modele = joblib.load(MODEL_PATH)
    print("✅ Modèle chargé")
    return modele


def faire_predictions(modele, X_test):
    y_pred  = modele.predict(X_test)
    y_proba = modele.predict_proba(X_test)[:, 1]
    print("✅ Prédictions effectuées")
    return y_pred, y_proba


def afficher_rapport(y_test, y_pred):
    print("\n📊 RAPPORT DE CLASSIFICATION\n")
    print(classification_report(y_test, y_pred,
                                target_names=["No Default", "Default"]))


def afficher_matrice_confusion(y_test, y_pred):
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(
        confusion_matrix(y_test, y_pred),
        display_labels=["No Default", "Default"]
    ).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Matrice de Confusion — Meilleur Modèle")
    plt.tight_layout()
    path = os.path.join(MODELS_DIR, "eval_confusion_matrix.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.show()
    print("✅ Matrice de confusion sauvegardée")


def afficher_courbe_roc(y_test, y_proba):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc          = roc_auc_score(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#378ADD", linewidth=2,
            label=f"AUC-ROC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Modèle aléatoire")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#378ADD")
    ax.set_xlabel("Taux de faux positifs")
    ax.set_ylabel("Taux de vrais positifs")
    ax.set_title("Courbe ROC — Meilleur Modèle")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(MODELS_DIR, "eval_courbe_roc.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"✅ Courbe ROC sauvegardée | AUC-ROC = {auc:.4f}")


def afficher_courbe_precision_recall(y_test, y_proba):
    precision, recall, seuils = precision_recall_curve(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="#D85A30", linewidth=2)
    ax.fill_between(recall, precision, alpha=0.1, color="#D85A30")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Courbe Precision-Recall — Meilleur Modèle")
    plt.tight_layout()
    path = os.path.join(MODELS_DIR, "eval_precision_recall.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.show()
    print("✅ Courbe Precision-Recall sauvegardée")


def afficher_resume(y_test, y_pred, y_proba):
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    print("\n" + "=" * 45)
    print(f"{'RÉSUMÉ FINAL':^45}")
    print("=" * 45)
    print(f"  AUC-ROC          : {roc_auc_score(y_test, y_proba):.4f}")
    print(f"  Vrais Positifs   : {tp}  (defaults détectés ✅)")
    print(f"  Faux Négatifs    : {fn}  (defaults manqués ⚠️)")
    print(f"  Faux Positifs    : {fp}  (fausses alarmes)")
    print(f"  Vrais Négatifs   : {tn}")
    print("=" * 45)


def run_evaluation():
    print("\n🚀 DÉBUT DE L'ÉVALUATION\n")

    # 1. Données
    X_train, X_test, y_train, y_test, scaler = run_preprocessing()

    # 2. Modèle
    modele = charger_modele()

    # 3. Prédictions
    y_pred, y_proba = faire_predictions(modele, X_test)

    # 4. Rapport
    afficher_rapport(y_test, y_pred)

    # 5. Graphiques
    afficher_matrice_confusion(y_test, y_pred)
    afficher_courbe_roc(y_test, y_proba)
    afficher_courbe_precision_recall(y_test, y_proba)

    # 6. Résumé
    afficher_resume(y_test, y_pred, y_proba)

    print("\n✅ ÉVALUATION TERMINÉE\n")


# ── POINT D'ENTRÉE ──
if __name__ == "__main__":
    run_evaluation()