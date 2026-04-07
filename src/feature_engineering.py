import pandas as pd
import numpy as np

# 🔹 CONFIGURATION
DATA_PATH = "Data/Loan_Data.csv"


def charger_donnees(path):
    df = pd.read_csv(path)
    print("✅ Données chargées")
    return df


def feature_engineering(df):

    # Suppression de l'identifiant client
    df = df.drop("customer_id", axis=1, errors="ignore")
    print("✅ customer_id supprimé")

    # Ratios financiers
    df["debt_to_income"] = df["total_debt_outstanding"] / df["income"]
    df["loan_to_income"] = df["loan_amt_outstanding"] / df["income"]
    print("✅ Ratios financiers créés")

    # Segmentation du FICO score
    df["fico_category"] = pd.cut(df["fico_score"],
                                 bins=[0, 580, 670, 740, 800, 850],
                                 labels=["Mauvais", "Passable", "Bon", "Très bon", "Exceptionnel"])
    print("✅ Catégories FICO créées")

    # Score de risque composite
    df["risk_score"] = (
            (df["fico_score"] < 600).astype(int) +
            (df["credit_lines_outstanding"] >= 3).astype(int) +
            (df["debt_to_income"] > 0.5).astype(int)
    ).astype(int)
    print("✅ Score de risque composite créé")

    return df


# ── POINT D'ENTRÉE ──
if __name__ == "__main__":
    df = charger_donnees(DATA_PATH)
    df = feature_engineering(df)
    print("\n📊 Aperçu des nouvelles variables :")
    print(df[["debt_to_income", "loan_to_income",
              "fico_category", "risk_score"]].head(10))
    print(f"\n✅ Dataset final : {df.shape[0]} lignes, {df.shape[1]} colonnes")
