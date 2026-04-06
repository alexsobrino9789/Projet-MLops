import streamlit as st
import pandas as pd
import joblib

# Charger le modèle
MODEL_PATH = "Models/model.pkl"
model = joblib.load(MODEL_PATH)

st.title("📊 Prédiction du Risque de Défaut")

st.write("Cette application prédit la probabilité qu’un client fasse défaut sur son prêt.")

# Formulaire utilisateur
st.header("🧾 Informations Client")

age = st.number_input("Âge", min_value=18, max_value=100, value=30)
income = st.number_input("Revenu annuel", min_value=0, value=50000)
loan_amt_outstanding = st.number_input("Montant du prêt restant", min_value=0, value=10000)
total_debt_outstanding = st.number_input("Dette totale", min_value=0, value=15000)
fico_score = st.number_input("FICO Score", min_value=300, max_value=850, value=650)
years_employed = st.number_input("Années d'emploi", min_value=0, max_value=50, value=5)

# Features dérivées
debt_to_income = total_debt_outstanding / income if income > 0 else 0
loan_to_income = loan_amt_outstanding / income if income > 0 else 0

# Préparation du dataframe
data = pd.DataFrame([{
    "age": age,
    "income": income,
    "loan_amt_outstanding": loan_amt_outstanding,
    "total_debt_outstanding": total_debt_outstanding,
    "fico_score": fico_score,
    "years_employed": years_employed,
    "debt_to_income": debt_to_income,
    "loan_to_income": loan_to_income
}])

# Bouton prédiction
if st.button("🔍 Prédire le risque"):
    proba = model.predict_proba(data)[0][1]
    st.subheader(f"Probabilité de défaut : **{proba:.2f}**")

    if proba > 0.5:
        st.error("⚠️ Risque ÉLEVÉ de défaut")
    else:
        st.success("🟢 Risque FAIBLE de défaut")
