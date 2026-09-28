import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns




df = pd.read_csv("../data/churn.csv")

print(df.head())
print("Dimension:", df.shape)
print("Columns:", df.columns)
print("\nInformations sur le dataset :")
df.info()

print("\nValeurs problématiques de TotalCharges :")

# Conversion de TotalCharges en numérique
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Remplacement des valeurs manquantes par 0
df["TotalCharges"] = df["TotalCharges"].fillna(0)

# Vérifications
print("\nType de TotalCharges :")
print(df["TotalCharges"].dtype)

print("\nValeurs manquantes dans TotalCharges :")
print(df["TotalCharges"].isnull().sum())

print("\nValeurs manquantes dans chaque colonne :")
print(df.isnull().sum())

print("\nChaînes vides dans les colonnes textuelles :")

for col in df.select_dtypes(include=["object", "string"]).columns:
    nombre_vides = (df[col].str.strip() == "").sum()
    print(col, ":", nombre_vides)

print("\nNombre de doublons :", df.duplicated().sum())

"""#Analysons la colonne churn
print("\nRépartition du Churn :")
print(df["Churn"].value_counts())
print("\nPourcentage de chaque catégorie de Churn :")
print(df["Churn"].value_counts(normalize=True) * 100)

sns.countplot(data=df, x="Churn")

plt.title("Répartition des clients selon le Churn")
plt.xlabel("Churn")
plt.ylabel("Nombre de clients")

plt.show()

print("\nTaux de Churn selon le type de contrat :")

contract_churn = pd.crosstab(
    df["Contract"],
    df["Churn"],
    normalize="index"
) * 100

print(contract_churn)

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Contract",
    hue="Churn"
)

plt.title("Churn selon le type de contrat")
plt.xlabel("Type de contrat")
plt.ylabel("Nombre de clients")

plt.show()

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Churn",
    y="tenure"
)

plt.title("Durée d'abonnement selon le Churn")
plt.xlabel("Churn")
plt.ylabel("Durée d'abonnement en mois")

plt.show()

print("\nDurée moyenne d'abonnement selon le Churn :")
print(df.groupby("Churn")["tenure"].mean())


plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Churn",
    y="MonthlyCharges"
)

plt.title("Frais mensuels selon le Churn")
plt.xlabel("Churn")
plt.ylabel("Frais mensuels")

plt.show()
print("\nFrais mensuels moyens selon le Churn :")
print(df.groupby("Churn")["MonthlyCharges"].mean())"""

print("\nTaux de Churn selon le service Internet :")

internet_churn = pd.crosstab(
    df["InternetService"],
    df["Churn"],
    normalize="index"
) * 100

print(internet_churn)


print("\nTaux de Churn selon le support technique :")

techsupport_churn = pd.crosstab(
    df["TechSupport"],
    df["Churn"],
    normalize="index"
) * 100

print(techsupport_churn)

print("\nTaux de Churn selon le mode de paiement :")

payment_churn = pd.crosstab(
    df["PaymentMethod"],
    df["Churn"],
    normalize="index"
) * 100

print(payment_churn)

print("\nTaux de Churn selon la facturation électronique :")

paperless_churn = pd.crosstab(
    df["PaperlessBilling"],
    df["Churn"],
    normalize="index"
) * 100

print(paperless_churn)

print("\nStatistiques descriptives des variables numériques :")

print(
    df[
        ["tenure", "MonthlyCharges", "TotalCharges"]
    ].describe()
)

print("\nMatrice de corrélation :")

correlation_matrix = df[
    ["tenure", "MonthlyCharges", "TotalCharges"]
].corr()

print(correlation_matrix)

plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Matrice de corrélation des variables numériques")
plt.show()

print("\nAnalyse des valeurs aberrantes :")

colonnes_numeriques = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

for col in colonnes_numeriques:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    limite_inferieure = Q1 - 1.5 * IQR
    limite_superieure = Q3 + 1.5 * IQR

    outliers = df[
        (df[col] < limite_inferieure) |
        (df[col] > limite_superieure)
    ]

    print(f"\nVariable : {col}")
    print(f"Q1 : {Q1:.2f}")
    print(f"Q3 : {Q3:.2f}")
    print(f"IQR : {IQR:.2f}")
    print(f"Limite inférieure : {limite_inferieure:.2f}")
    print(f"Limite supérieure : {limite_superieure:.2f}")
    print(f"Nombre d'outliers : {len(outliers)}")