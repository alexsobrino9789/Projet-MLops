import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 🔹 CONFIGURATION
DATA_PATH = "Data/Loan_Data.csv"
TARGET = "default"


def charger_donnees(path):
    df = pd.read_csv(path)
    print("✅ Données chargées")
    return df


def nettoyer_donnees(df):
    df = df.drop_duplicates()
    df = df.fillna(df.median(numeric_only=True))
    print("✅ Données nettoyées")
    return df


def encoder_donnees(df):
    df = pd.get_dummies(df, drop_first=True)
    print("✅ Données encodées")
    return df


def diviser_donnees(df):
    X = df.drop(TARGET, axis=1)
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("✅ Données divisées (train/test)")
    return X_train, X_test, y_train, y_test


def normaliser_donnees(X_train, X_test):
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("✅ Données normalisées")
    return X_train, X_test, scaler


def run_preprocessing():
    print("\n🚀 DÉBUT DU PRÉTRAITEMENT\n")

    df = charger_donnees(DATA_PATH)
    df = nettoyer_donnees(df)
    df = encoder_donnees(df)

    X_train, X_test, y_train, y_test = diviser_donnees(df)
    X_train, X_test, scaler = normaliser_donnees(X_train, X_test)

    print("\n✅ PRÉTRAITEMENT TERMINÉ\n")

    return X_train, X_test, y_train, y_test, scaler


# ── POINT D’ENTRÉE ──
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler = run_preprocessing()

    print("📊 RÉSUMÉ FINAL")
    print(f"X_train : {X_train.shape}")
    print(f"X_test  : {X_test.shape}")

    print("\nDistribution de la variable cible :")
    print(f"Train → 0: {(y_train==0).sum()} | 1: {(y_train==1).sum()}")
    print(f"Test  → 0: {(y_test==0).sum()} | 1: {(y_test==1).sum()}")