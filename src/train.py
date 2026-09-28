
"""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)


# ==========================================
# 1. Charger le dataset
# ==========================================

df = pd.read_csv("../data/churn.csv")

print("Dimensions initiales :", df.shape)


# ==========================================
# 2. Nettoyer TotalCharges
# ==========================================

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df["TotalCharges"] = df["TotalCharges"].fillna(0)


# ==========================================
# 3. Supprimer customerID
# ==========================================

df = df.drop("customerID", axis=1)


# ==========================================
# 4. Transformer Churn en valeurs numériques
# ==========================================

df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})


# ==========================================
# 5. Séparer X et y
# ==========================================

X = df.drop("Churn", axis=1)
y = df["Churn"]


# ==========================================
# 6. Encoder les variables catégorielles
# ==========================================

X_encoded = pd.get_dummies(
    X,
    drop_first=True,
    dtype=int
)

print("Dimensions après encodage :", X_encoded.shape)


# ==========================================
# 7. Séparer les données train/test
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Dimensions de X_train :", X_train.shape)
print("Dimensions de X_test :", X_test.shape)


#Logistique Regressşon
# ==========================================
# 8. Créer le modèle
# ==========================================

model = LogisticRegression(
    max_iter=1000
)


# ==========================================
# 9. Entraîner le modèle
# ==========================================

model.fit(X_train, y_train)

print("\nLe modèle a été entraîné avec succès.")


# ==========================================
# 10. Faire des prédictions
# ==========================================

y_pred = model.predict(X_test)

print("\nPremières prédictions :")
print(y_pred[:20])


# ==========================================
# 11. Évaluer le modèle
# ==========================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n========== ÉVALUATION DU MODÈLE ==========")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")


# ==========================================
# 12. Matrice de confusion
# ==========================================

cm = confusion_matrix(y_test, y_pred)

print("\nMatrice de confusion :")
print(cm)


# ==========================================
# 13. Rapport de classification
# ==========================================

print("\nRapport de classification :")
print(classification_report(y_test, y_pred))





#Decision Tree

# ==========================================
# 8. Tester plusieurs profondeurs
# ==========================================

profondeurs = [3, 5, 7, 10, None]

print("\n========== COMPARAISON DES PROFONDEURS ==========")

for profondeur in profondeurs:

    model = DecisionTreeClassifier(
        max_depth=profondeur,
        random_state=42
    )

    # Entraîner le modèle
    model.fit(X_train, y_train)

    # Faire les prédictions
    y_pred = model.predict(X_test)

    # Calculer les métriques
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\nProfondeur : {profondeur}")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")

#Random Forest
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
# ==========================================
# 9. Entraîner le modèle
# ==========================================

model.fit(X_train, y_train)

print("\nLe modèle Random Forest a été entraîné avec succès.")


# ==========================================
# 10. Faire des prédictions
# ==========================================

y_pred = model.predict(X_test)

print("\nPremières prédictions :")
print(y_pred[:20])


# ==========================================
# 11. Évaluer le modèle
# ==========================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n========== ÉVALUATION DU RANDOM FOREST ==========")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")


# ==========================================
# 12. Matrice de confusion
# ==========================================

cm = confusion_matrix(y_test, y_pred)

print("\nMatrice de confusion :")
print(cm)


# ==========================================
# 13. Rapport de classification
# ==========================================

print("\nRapport de classification :")
print(classification_report(y_test, y_pred))

#random forest avec gridsearchcv

# ==========================================
# 8. Créer le modèle de base
# ==========================================

model = RandomForestClassifier(
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)


# ==========================================
# 9. Définir les hyperparamètres à tester
# ==========================================

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [5, 7, 10],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2, 4]
}


# ==========================================
# 10. Créer GridSearchCV
# ==========================================

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
    verbose=1
)


# ==========================================
# 11. Rechercher les meilleurs paramètres
# ==========================================

print("\n========== OPTIMISATION DU RANDOM FOREST ==========")

grid_search.fit(X_train, y_train)

print("\nMeilleurs paramètres :")
print(grid_search.best_params_)

print("\nMeilleur score F1 obtenu pendant la validation croisée :")
print(grid_search.best_score_)


# ==========================================
# 12. Récupérer le meilleur modèle
# ==========================================

best_model = grid_search.best_estimator_

print("\nLe meilleur modèle a été sélectionné.")


# ==========================================
# 13. Faire des prédictions sur le test
# ==========================================

y_pred = best_model.predict(X_test)

# Probabilités de churn
y_proba = best_model.predict_proba(X_test)[:, 1]

print("\nExemples de probabilités de churn :")
print(y_proba[:20])

resultats = pd.DataFrame({
    "Probabilite_Churn": y_proba,
    "Valeur_Reelle": y_test.values
})

print(resultats.head(20))

# ==========================================
# 14. Évaluer le meilleur modèle
# ==========================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

# Calcul des points de la courbe ROC
fpr, tpr, thresholds = roc_curve(y_test, y_proba)

# Création du graphique
plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"Random Forest optimisé (AUC = {roc_auc:.3f})"
)

# Ligne correspondant au hasard
plt.plot(
    [0, 1],
    [0, 1],
    "--",
    label="Classification aléatoire"
)

plt.xlabel("Taux de faux positifs (FPR)")
plt.ylabel("Taux de vrais positifs (TPR / Recall)")
plt.title("Courbe ROC - Random Forest optimisé")
plt.legend()
plt.grid()
plt.show()


print("\n========== ÉVALUATION DU MEILLEUR MODÈLE ==========")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")



# ==========================================
# 15. Matrice de confusion
# ==========================================

cm = confusion_matrix(y_test, y_pred)

print("\nMatrice de confusion :")
print(cm)


# ==========================================
# 16. Rapport de classification
# ==========================================

print("\nRapport de classification :")
print(classification_report(y_test, y_pred))


# ==========================================
# TESTONS POUR AVOIR LE MEILLEUR SEUIL
# ==========================================

seuils = np.arange(0.01, 1.00, 0.01)

resultats = []

for seuil in seuils:

    y_pred_seuil = (y_proba >= seuil).astype(int)

    precision = precision_score(
        y_test,
        y_pred_seuil,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred_seuil,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred_seuil,
        zero_division=0
    )

    resultats.append(
        [seuil, precision, recall, f1]
    )


# ==========================================
# RECHERCHE DU MEILLEUR SEUIL
# ==========================================

resultats = np.array(resultats)

indice_meilleur = np.argmax(resultats[:, 3])

meilleur_seuil = resultats[indice_meilleur, 0]
meilleure_precision = resultats[indice_meilleur, 1]
meilleur_recall = resultats[indice_meilleur, 2]
meilleur_f1 = resultats[indice_meilleur, 3]


print("\n========== MEILLEUR SEUIL SELON F1 ==========")

print(f"Seuil     : {meilleur_seuil:.2f}")
print(f"Precision : {meilleure_precision:.4f}")
print(f"Recall    : {meilleur_recall:.4f}")
print(f"F1-score  : {meilleur_f1:.4f}")




# ==========================================
# VISUALISONS LE CHOIX DU SEUIL
# ==========================================

plt.figure(figsize=(10, 6))

plt.plot(
    resultats[:, 0],
    resultats[:, 1],
    label="Precision"
)

plt.plot(
    resultats[:, 0],
    resultats[:, 2],
    label="Recall"
)

plt.plot(
    resultats[:, 0],
    resultats[:, 3],
    label="F1-score"
)

# Marquer le meilleur seuil
plt.axvline(
    meilleur_seuil,
    linestyle="--",
    label=f"Meilleur seuil = {meilleur_seuil:.2f}"
)

plt.xlabel("Seuil de décision")
plt.ylabel("Score")
plt.title("Precision, Recall et F1-score selon le seuil")

plt.legend()
plt.grid()

plt.show()



# ==========================================
# Après le graphique, nous allons appliquer notre seuil retenu selon le F1 :
# ==========================================
seuil_final = meilleur_seuil

y_pred_final = (y_proba >= seuil_final).astype(int)


cm = confusion_matrix(y_test, y_pred_final)

print("\n========== MATRICE DE CONFUSION ==========")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Non-Churn", "Churn"]
)

disp.plot()

plt.title(f"Matrice de confusion - seuil {seuil_final:.2f}")

plt.show()




# ==========================================
# ENTRAINONS LE MODEL XGBOOST
# ==========================================


xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    random_state=42,
    eval_metric="logloss"
)

xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)
y_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]

print("\n========== XGBOOST ==========")

print(
    f"Accuracy  : {accuracy_score(y_test, y_pred_xgb):.4f}"
)

print(
    f"Precision : {precision_score(y_test, y_pred_xgb):.4f}"
)

print(
    f"Recall    : {recall_score(y_test, y_pred_xgb):.4f}"
)

print(
    f"F1-score  : {f1_score(y_test, y_pred_xgb):.4f}"
)

print(
    f"ROC-AUC   : {roc_auc_score(y_test, y_proba_xgb):.4f}"
)

print("\nMatrice de confusion :")
print(confusion_matrix(y_test, y_pred_xgb))




# Probabilités de churn
y_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]

seuils_xgb = np.arange(0.01, 1.00, 0.01)

resultats_xgb = []

for seuil in seuils_xgb:

    y_pred_seuil = (y_proba_xgb >= seuil).astype(int)

    precision = precision_score(
        y_test,
        y_pred_seuil,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred_seuil,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred_seuil,
        zero_division=0
    )

    resultats_xgb.append(
        [seuil, precision, recall, f1]
    )


# Conversion en tableau NumPy
resultats_xgb = np.array(resultats_xgb)


# Recherche du meilleur F1
indice_meilleur_xgb = np.argmax(
    resultats_xgb[:, 3]
)

meilleur_seuil_xgb = resultats_xgb[
    indice_meilleur_xgb, 0
]

meilleure_precision_xgb = resultats_xgb[
    indice_meilleur_xgb, 1
]

meilleur_recall_xgb = resultats_xgb[
    indice_meilleur_xgb, 2
]

meilleur_f1_xgb = resultats_xgb[
    indice_meilleur_xgb, 3
]


print("\n========== MEILLEUR SEUIL XGBOOST ==========")

print(
    f"Seuil     : {meilleur_seuil_xgb:.2f}"
)

print(
    f"Precision : {meilleure_precision_xgb:.4f}"
)

print(
    f"Recall    : {meilleur_recall_xgb:.4f}"
)

print(
    f"F1-score  : {meilleur_f1_xgb:.4f}"
)



# ==========================================
# XGBOOST + GRIDSEARCHCV
# ==========================================

xgb_model = XGBClassifier(
    random_state=42,
    eval_metric="logloss",
    n_jobs=-1
)

param_grid_xgb = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "min_child_weight": [1, 3],
    "subsample": [0.8, 1.0]
}

grid_search_xgb = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid_xgb,
    cv=5,
    scoring="f1",
    n_jobs=-1,
    verbose=1
)

grid_search_xgb.fit(X_train, y_train)

# Meilleur modèle
best_xgb = grid_search_xgb.best_estimator_

print("\n========== XGBOOST OPTIMISÉ ==========")

print("Meilleurs paramètres :")
print(grid_search_xgb.best_params_)

print("\nMeilleur score F1 obtenu pendant la validation croisée :")
print(grid_search_xgb.best_score_)


# ==========================================
# ÉVALUATION DU MEILLEUR XGBOOST
# ==========================================

y_pred_xgb_best = best_xgb.predict(X_test)

y_proba_xgb_best = best_xgb.predict_proba(X_test)[:, 1]

print("\n========== ÉVALUATION XGBOOST OPTIMISÉ ==========")

print(
    f"Accuracy  : {accuracy_score(y_test, y_pred_xgb_best):.4f}"
)

print(
    f"Precision : {precision_score(y_test, y_pred_xgb_best):.4f}"
)

print(
    f"Recall    : {recall_score(y_test, y_pred_xgb_best):.4f}"
)

print(
    f"F1-score  : {f1_score(y_test, y_pred_xgb_best):.4f}"
)

print(
    f"ROC-AUC   : {roc_auc_score(y_test, y_proba_xgb_best):.4f}"
)

print("\nMatrice de confusion :")
print(
    confusion_matrix(
        y_test,
        y_pred_xgb_best
    )
)



# Probabilités prédites par le XGBoost optimisé
y_proba_xgb_best = best_xgb.predict_proba(X_test)[:, 1]

# Seuils de 0.01 à 0.99
seuils_xgb_best = np.arange(0.01, 1.00, 0.01)

resultats_xgb_best = []

for seuil in seuils_xgb_best:

    y_pred_seuil = (
        y_proba_xgb_best >= seuil
    ).astype(int)

    precision = precision_score(
        y_test,
        y_pred_seuil,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred_seuil,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred_seuil,
        zero_division=0
    )

    resultats_xgb_best.append(
        [seuil, precision, recall, f1]
    )


# Conversion en NumPy
resultats_xgb_best = np.array(
    resultats_xgb_best
)


# Trouver le meilleur F1
indice_meilleur_xgb_best = np.argmax(
    resultats_xgb_best[:, 3]
)

meilleur_seuil_xgb_best = (
    resultats_xgb_best[
        indice_meilleur_xgb_best, 0
    ]
)

meilleure_precision_xgb_best = (
    resultats_xgb_best[
        indice_meilleur_xgb_best, 1
    ]
)

meilleur_recall_xgb_best = (
    resultats_xgb_best[
        indice_meilleur_xgb_best, 2
    ]
)

meilleur_f1_xgb_best = (
    resultats_xgb_best[
        indice_meilleur_xgb_best, 3
    ]
)


print(
    "\n========== MEILLEUR SEUIL XGBOOST OPTIMISÉ =========="
)

print(
    f"Seuil     : {meilleur_seuil_xgb_best:.2f}"
)

print(
    f"Precision : {meilleure_precision_xgb_best:.4f}"
)

print(
    f"Recall    : {meilleur_recall_xgb_best:.4f}"
)

print(
    f"F1-score  : {meilleur_f1_xgb_best:.4f}"
)"""











# ==========================================
# LE PROJET AVEC PIPELINE
# ==========================================

from preprocessing import load_and_prepare_data
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import os

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve
)

import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd


# ==============================
# 1. Charger les données
# ==============================

X, y = load_and_prepare_data()


# ==============================
# 2. Train / Validation / Test
# ==============================

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.25,
    random_state=42,
    stratify=y_train_val
)


print("\n========== TRAIN / VALIDATION / TEST ==========")

print("X_train :", X_train.shape)
print("X_val   :", X_val.shape)
print("X_test  :", X_test.shape)

print("y_train :", y_train.shape)
print("y_val   :", y_val.shape)
print("y_test  :", y_test.shape)


# ==============================
# 3. Répartition des classes
# ==============================

print("\nRépartition de y_train :")
print(y_train.value_counts())

print("\nRépartition de y_val :")
print(y_val.value_counts())

print("\nRépartition de y_test :")
print(y_test.value_counts())


# ==============================
# 4. Identifier les variables
# ==============================

numeric_features = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

categorical_features = [
    col for col in X.columns
    if col not in numeric_features
]


print("\n========== VARIABLES ==========")

print("Variables numériques :")
print(numeric_features)

print("\nVariables catégorielles :")
print(categorical_features)


# ==============================
# 5. Preprocessing
# ==============================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            "passthrough",
            numeric_features
        ),
        (
            "cat",
            OneHotEncoder(
                drop="first",
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ]
)

print("\n========== PREPROCESSING ==========")
print("ColumnTransformer créé avec succès.")


# ==============================
# 6. Random Forest
# ==============================

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=2,
    min_samples_split=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("\n========== MODEL ==========")
print("Random Forest créé avec succès.")


# ==============================
# 7. Pipeline
# ==============================

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

print("\n========== PIPELINE ==========")
print("Pipeline créé avec succès.")


# ==============================
# 8. Entraîner le modèle
# ==============================

print("\n========== ENTRAÎNEMENT ==========")

pipeline.fit(
    X_train,
    y_train
)

print("Entraînement terminé avec succès.")




# ==============================
# 8.1. FEATURE IMPORTANCE
# ==============================

print("\n========== FEATURE IMPORTANCE ==========")

# Récupérer le modèle Random Forest entraîné
rf_model = pipeline.named_steps["model"]

# Récupérer les noms des variables après preprocessing
feature_names = (
    pipeline.named_steps["preprocessor"]
    .get_feature_names_out()
)

# Récupérer l'importance de chaque variable
importances = rf_model.feature_importances_

# Créer un DataFrame
feature_importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

# Trier les variables par ordre décroissant
feature_importance_df = (
    feature_importance_df
    .sort_values(by="Importance", ascending=False)
)

# Afficher les 15 variables les plus importantes
print("\nTop 15 des variables les plus importantes :")
print(feature_importance_df.head(15).to_string(index=False))

# Visualiser les 15 variables les plus importantes
top_features = feature_importance_df.head(15).sort_values(
    by="Importance",
    ascending=True
)

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Variables")
plt.title("Top 15 des variables importantes - Random Forest")

plt.tight_layout()

# Créer le dossier Models s'il n'existe pas
os.makedirs("../Models", exist_ok=True)

# Enregistrer le graphique
plt.savefig(
    "../Models/feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nGraphique sauvegardé dans Models/feature_importance.png")





# ==============================
# 9. Validation
# ==============================

y_val_proba = pipeline.predict_proba(
    X_val
)[:, 1]

print("\n========== VALIDATION ==========")

print("Premières probabilités de churn :")
print(y_val_proba[:20])


# ==============================
# 10. Recherche du meilleur seuil
# ==============================

thresholds_to_test = np.arange(
    0.01,
    1.00,
    0.01
)

best_threshold = 0
best_f1 = 0

print("\n========== RECHERCHE DU SEUIL ==========")

for threshold in thresholds_to_test:

    y_val_pred = (
        y_val_proba >= threshold
    ).astype(int)

    f1 = f1_score(
        y_val,
        y_val_pred,
        zero_division=0
    )

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold


print("\n========== MEILLEUR SEUIL ==========")

print(
    f"Seuil optimal : {best_threshold:.2f}"
)

print(
    f"F1-score validation : {best_f1:.4f}"
)


# ==============================
# 11. Évaluation finale sur Test
# ==============================

y_test_proba = pipeline.predict_proba(
    X_test
)[:, 1]

y_test_pred = (
    y_test_proba >= best_threshold
).astype(int)


accuracy = accuracy_score(
    y_test,
    y_test_pred
)

precision = precision_score(
    y_test,
    y_test_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_test_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_test_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_test_proba
)


print("\n========== ÉVALUATION FINALE ==========")

print(
    f"Seuil utilisé : {best_threshold:.2f}"
)

print(
    f"Accuracy      : {accuracy:.4f}"
)

print(
    f"Precision     : {precision:.4f}"
)

print(
    f"Recall        : {recall:.4f}"
)

print(
    f"F1-score      : {f1:.4f}"
)

print(
    f"ROC-AUC       : {roc_auc:.4f}"
)


# ==============================
# 12. Matrice de confusion
# ==============================

cm = confusion_matrix(
    y_test,
    y_test_pred
)

print("\n========== MATRICE DE CONFUSION ==========")
print(cm)


disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "No Churn",
        "Churn"
    ]
)

disp.plot()

plt.title(
    "Matrice de confusion - Random Forest"
)

plt.xlabel("Prédiction")
plt.ylabel("Réalité")

plt.tight_layout()

plt.savefig(
    "../Models/cm.png"
)

plt.show()


# ==============================
# 13. Courbe ROC
# ==============================

fpr, tpr, _ = roc_curve(
    y_test,
    y_test_proba
)

plt.figure(
    figsize=(8, 6)
)

plt.plot(
    fpr,
    tpr,
    label=f"Random Forest (AUC = {roc_auc:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "Courbe ROC - Random Forest"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "../Models/roc_curve.png"
)

plt.show()


# ==============================
# 14. Sauvegarder le modèle
# ==============================

model_data = {
    "pipeline": pipeline,
    "threshold": best_threshold
}

joblib.dump(
    model_data,
    "../Models/churn_pipeline.pkl"
)

print("\n========== SAUVEGARDE ==========")

print(
    "Pipeline et seuil sauvegardés avec succès."
)

print(
    f"Seuil sauvegardé : {best_threshold:.2f}"
)