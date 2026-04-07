import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model   import LogisticRegression
from sklearn.tree           import DecisionTreeClassifier
from sklearn.ensemble       import RandomForestClassifier
from sklearn.metrics        import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay,
)

import mlflow
import mlflow.sklearn

# ── Import du preprocessing ──
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import run_preprocessing

# 🔹 CONFIGURATION
MODELS_DIR  = "Models"
MLRUNS_DIR  = "MLruns"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(MLRUNS_DIR, exist_ok=True)

mlruns_abs = os.path.abspath(MLRUNS_DIR).replace("\\", "/")
mlflow.set_tracking_uri(f"file:///{mlruns_abs}")


# ─────────────────────────────────────────────────────────────────────────────

def definir_modeles():
    modeles = [
        {
            "nom"    : "Logistic_Regression",
            "modele" : LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42, C=1.0),
            "params" : {"max_iter": 1000, "C": 1.0, "class_weight": "balanced"},
        },
        {
            "nom"    : "Decision_Tree",
            "modele" : DecisionTreeClassifier(max_depth=6, min_samples_leaf=10, class_weight="balanced", random_state=42),
            "params" : {"max_depth": 6, "min_samples_leaf": 10, "class_weight": "balanced"},
        },
        {
            "nom"    : "Random_Forest",
            "modele" : RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1),
            "params" : {"n_estimators": 200, "max_depth": 10, "class_weight": "balanced"},
        },
    ]
    print("✅ Modèles définis")
    return modeles


def calculer_metriques(y_test, y_pred, y_proba):
    metriques = {
        "accuracy"  : round(accuracy_score(y_test, y_pred), 4),
        "precision" : round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall"    : round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1"        : round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc"   : round(roc_auc_score(y_test, y_proba), 4),
    }
    print("✅ Métriques calculées")
    return metriques


def sauvegarder_graphiques(modele, nom, X_test, y_test, y_pred):
    # Matrice de confusion
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(
        confusion_matrix(y_test, y_pred),
        display_labels=["No Default", "Default"]
    ).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {nom.replace('_', ' ')}")
    cm_path = os.path.join(MODELS_DIR, f"cm_{nom}.png")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=120, bbox_inches="tight")
    plt.close()

    # Courbe ROC
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(modele, X_test, y_test, ax=ax,
                                   name=nom.replace("_", " "))
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_title(f"ROC Curve — {nom.replace('_', ' ')}")
    roc_path = os.path.join(MODELS_DIR, f"roc_{nom}.png")
    plt.tight_layout()
    plt.savefig(roc_path, dpi=120, bbox_inches="tight")
    plt.close()

    # Feature importance
    fi_path = None
    if hasattr(modele, "feature_importances_"):
        importances = modele.feature_importances_
    elif hasattr(modele, "coef_"):
        importances = np.abs(modele.coef_[0])
    else:
        importances = None

    if importances is not None:                                      # ✅ indentación correcta
        if hasattr(X_test, "columns"):
            feature_names = list(X_test.columns)
        else:
            feature_names = [f"feature_{i}" for i in range(len(importances))]

        idx = np.argsort(importances)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh([feature_names[i] for i in idx], importances[idx], color="#378ADD")
        ax.set_title(f"Feature Importance — {nom.replace('_', ' ')}")
        fi_path = os.path.join(MODELS_DIR, f"importance_{nom}.png")
        plt.tight_layout()
        plt.savefig(fi_path, dpi=120, bbox_inches="tight")
        plt.close()

    print("✅ Graphiques sauvegardés")
    return cm_path, roc_path, fi_path              # ✅ dentro de la función


def entrainer_modele(cfg, X_train, X_test, y_train, y_test):
    nom    = cfg["nom"]
    modele = cfg["modele"]
    params = cfg["params"]

    print(f"\n🚀 Entraînement : {nom}\n")

    mlflow.set_experiment(f"LoanDefault_{nom}")

    with mlflow.start_run(run_name=f"run_{nom}") as run:

        # Entraînement
        modele.fit(X_train, y_train)
        y_pred  = modele.predict(X_test)
        y_proba = modele.predict_proba(X_test)[:, 1]

        # Métriques
        metriques = calculer_metriques(y_test, y_pred, y_proba)

        # Log MLflow — paramètres
        mlflow.log_params(params)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size",  len(X_test))

        # Log MLflow — métriques
        mlflow.log_metrics(metriques)

        # Log MLflow — artefacts graphiques
        cm_path, roc_path, fi_path = sauvegarder_graphiques(
            modele, nom, X_test, y_test, y_pred
        )
        mlflow.log_artifact(cm_path,  artifact_path="plots")
        mlflow.log_artifact(roc_path, artifact_path="plots")
        if fi_path:
            mlflow.log_artifact(fi_path, artifact_path="plots")

        # Log MLflow — métriques JSON
        metrics_path = os.path.join(MODELS_DIR, f"metrics_{nom}.json")
        with open(metrics_path, "w") as f:
            json.dump(metriques, f, indent=2)
        mlflow.log_artifact(metrics_path, artifact_path="metrics")

        # Log MLflow — modèle
        mlflow.sklearn.log_model(
            modele,
            artifact_path="model",
            registered_model_name=f"LoanDefault_{nom}",
        )

        # Sauvegarde locale
        joblib.dump(modele, os.path.join(MODELS_DIR, f"{nom}.pkl"))

        print(f"✅ {nom} terminé | Run ID: {run.info.run_id}")
        print(f"   → AUC-ROC: {metriques['roc_auc']} | F1: {metriques['f1']}")

    return metriques, modele


def selectionner_meilleur_modele(resultats):
    meilleur = max(resultats, key=lambda r: r["metriques"]["roc_auc"])
    src  = os.path.join(MODELS_DIR, f"{meilleur['nom']}.pkl")
    dest = os.path.join(MODELS_DIR, "model.pkl")
    joblib.dump(joblib.load(src), dest)
    print(f"\n🏆 Meilleur modèle : {meilleur['nom']}")
    print(f"   → AUC-ROC : {meilleur['metriques']['roc_auc']}")
    print(f"   → Sauvegardé comme model.pkl")
    return meilleur


def afficher_comparatif(resultats):
    print("\n" + "=" * 65)
    print(f"{'COMPARATIF DES MODÈLES':^65}")
    print("=" * 65)
    print(f"{'Modèle':<25} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'AUC-ROC':>9}")
    print("-" * 65)
    for r in resultats:
        m = r["metriques"]
        print(
            f"{r['nom']:<25} {m['accuracy']:>9.4f} {m['precision']:>10.4f} "
            f"{m['recall']:>8.4f} {m['f1']:>8.4f} {m['roc_auc']:>9.4f}"
        )
    print("=" * 65)


def run_training():
    print("\n🚀 DÉBUT DE L'ENTRAÎNEMENT\n")

    # 1. Prétraitement
    X_train, X_test, y_train, y_test, scaler = run_preprocessing()
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))

    # 2. Définir les modèles
    modeles = definir_modeles()

    # 3. Entraîner chaque modèle
    resultats = []
    for cfg in modeles:
        metriques, modele_entraine = entrainer_modele(
            cfg, X_train, X_test, y_train, y_test
        )
        resultats.append({
            "nom"      : cfg["nom"],
            "metriques": metriques,
            "modele"   : modele_entraine,
        })

    # 4. Comparatif
    afficher_comparatif(resultats)

    # 5. Meilleur modèle → model.pkl
    meilleur = selectionner_meilleur_modele(resultats)

    print("\n✅ ENTRAÎNEMENT TERMINÉ\n")
    print("📊 Pour visualiser dans MLflow UI :")
    print(f"   mlflow ui --backend-store-uri file:///{mlruns_abs}")
    print("   → http://127.0.0.1:5000")

    return resultats, meilleur


# ── POINT D'ENTRÉE ──
if __name__ == "__main__":
    resultats, meilleur = run_training()