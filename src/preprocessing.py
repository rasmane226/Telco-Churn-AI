
"""import pandas as pd
from sklearn.model_selection import train_test_split
# Charger le dataset
df = pd.read_csv("../data/churn.csv")

print("Dimensions initiales :", df.shape)
print(df.head())

# Convertir TotalCharges en valeurs numériques
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Remplacer les valeurs manquantes par 0
df["TotalCharges"] = df["TotalCharges"].fillna(0)

# Vérifier TotalCharges
print("\nVérification de TotalCharges :")
print("Type :", df["TotalCharges"].dtype)
print("Valeurs manquantes :", df["TotalCharges"].isnull().sum())

# Supprimer la colonne d'identification
df = df.drop("customerID", axis=1)

print("\nDimensions après suppression de customerID :", df.shape)
print(df.head())

# Transformer la variable cible Churn en valeurs numériques
df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

print("\nRépartition de Churn après transformation :")
print(df["Churn"].value_counts())

# Vérifier les types
print("\nTypes des colonnes :")
print(df.dtypes)

print("\nPremières valeurs de Churn :")
print(df["Churn"].head(10))

# Séparer les variables explicatives et la variable cible

X = df.drop("Churn", axis=1)
y = df["Churn"]

print("\nDimensions de X :", X.shape)
print("Dimensions de y :", y.shape)

print("\nColonnes de X :")
print(X.columns)

print("\nPremières valeurs de y :")
print(y.head())

# Encoder les variables catégorielles

X_encoded = pd.get_dummies(
    X,
    drop_first=True,
    dtype=int
)

print("\nDimensions de X après encodage :", X_encoded.shape)

print("\nColonnes de X après encodage :")
print(X_encoded.columns)

print("\nAperçu de X après encodage :")
print(X_encoded.head())

print("\nTypes des colonnes après encodage :")
print(X_encoded.dtypes)

print("\nNombre total de colonnes non numériques :")
print(
    X_encoded.select_dtypes(exclude=["number"]).shape[1]
)

# Séparer les données en ensemble d'entraînement et ensemble de test

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nDimensions de X_train :", X_train.shape)
print("Dimensions de X_test :", X_test.shape)
print("Dimensions de y_train :", y_train.shape)
print("Dimensions de y_test :", y_test.shape)

print("\nRépartition de y_train :")
print(y_train.value_counts())

print("\nRépartition de y_test :")
print(y_test.value_counts())"""






import pandas as pd


def load_and_prepare_data():

    # ==============================
    # 1. Charger le dataset
    # ==============================

    df = pd.read_csv("../Data/churn.csv")

    print("Dimensions initiales :", df.shape)


    # ==============================
    # 2. Nettoyer TotalCharges
    # ==============================

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    df["TotalCharges"] = df["TotalCharges"].fillna(0)


    # ==============================
    # 3. Supprimer customerID
    # ==============================

    df = df.drop("customerID", axis=1)


    # ==============================
    # 4. Transformer Churn
    # ==============================

    df["Churn"] = df["Churn"].map({
        "No": 0,
        "Yes": 1
    })


    # ==============================
    # 5. Séparer X et y
    # ==============================

    X = df.drop("Churn", axis=1)

    y = df["Churn"]


    # ==============================
    # 6. Afficher les informations
    # ==============================

    print("\nDataset préparé avec succès.")

    print("Dimensions de X :", X.shape)
    print("Dimensions de y :", y.shape)

    print("\nVariables de X :")
    print(X.columns.tolist())

    print("\nRépartition de y :")
    print(y.value_counts())


    return X, y

if __name__ == "__main__":
    X, y = load_and_prepare_data()