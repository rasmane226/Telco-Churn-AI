import os
import joblib
import shap
import pandas as pd


# ==========================================
# 1. CHARGEMENT DU MODELE
# ==========================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "Models",
    "churn_pipeline.pkl"
)

model_data = joblib.load(MODEL_PATH)

pipeline = model_data["pipeline"]
threshold = model_data["threshold"]

preprocessor = pipeline.named_steps["preprocessor"]
rf_model = pipeline.named_steps["model"]


# ==========================================
# 2. FONCTION DE PREDICTION
# ==========================================

def predict_churn(client_df):

    probability = pipeline.predict_proba(
        client_df
    )[:, 1][0]

    prediction = int(probability >= threshold)

    if prediction == 1:
        result = "CHURN"
    else:
        result = "NO CHURN"

    return result, probability


# ==========================================
# 3. EXPLICATION SHAP
# ==========================================

def explain_churn(client_df):
    """
    Explique la prédiction d'un client avec SHAP.

    Retourne :
    - Un DataFrame contenant les contributions SHAP
    - La valeur de base
    - La probabilité de churn
    """

    # Transformer les données du client
    X_transformed = preprocessor.transform(client_df)

    # Convertir en tableau dense si nécessaire
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    # Récupérer les noms des variables
    feature_names = preprocessor.get_feature_names_out()

    # Créer l'explainer SHAP
    explainer = shap.TreeExplainer(rf_model)

    # Calculer les valeurs SHAP
    shap_values = explainer(X_transformed)

    values = shap_values.values
    base_values = shap_values.base_values

    # Sélectionner les contributions de la classe Churn
    if values.ndim == 3:
        contributions = values[0, :, 1]
    elif values.ndim == 2:
        contributions = values[0, :]
    else:
        raise ValueError(
            f"Format SHAP inattendu : {values.shape}"
        )

    # Récupérer la valeur de base de la classe Churn
    if getattr(base_values, "ndim", 0) == 2:
        base_value = float(base_values[0, 1])
    elif getattr(base_values, "ndim", 0) == 1:
        if len(base_values) > 1:
            base_value = float(base_values[1])
        else:
            base_value = float(base_values[0])
    else:
        base_value = float(base_values)

    # Créer le tableau des contributions
    explanation_df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP_Value": contributions
    })

    # Trier par importance absolue
    explanation_df["Importance"] = (
        explanation_df["SHAP_Value"].abs()
    )

    explanation_df = explanation_df.sort_values(
        by="Importance",
        ascending=False
    ).reset_index(drop=True)

    # Calculer la probabilité de churn
    probability = float(
        pipeline.predict_proba(client_df)[0, 1]
    )

    return explanation_df, base_value, probability


# ==========================================
# 4. TEST
# ==========================================

if __name__ == "__main__":

    print("Pipeline chargé avec succès.")
    print(f"Seuil chargé : {threshold:.2f}")
    print("SHAP prêt à être utilisé.")