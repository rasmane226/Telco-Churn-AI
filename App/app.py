import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from datetime import datetime


# ============================================================
# CONFIGURATION DE LA PAGE
# ============================================================

st.set_page_config(
    page_title="Telco Churn AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CHEMIN DU PROJET
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

# Permet à Python de trouver le dossier src
sys.path.insert(
    0,
    PROJECT_ROOT
)


# ============================================================
# IMPORT DU MODELE
# ============================================================

from src.predict import predict_churn, explain_churn, threshold


# ============================================================
# HISTORIQUE DES PREDICTIONS
# ============================================================

HISTORY_PATH = os.path.join(PROJECT_ROOT, "Data", "predictions_history.csv")


def save_prediction(client_data, result, probability):
    """Enregistre une prédiction dans le fichier CSV."""
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
    row = {
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **client_data,
        "Prediction": result,
        "Probabilite_Churn": round(float(probability) * 100, 2)
    }
    new_row = pd.DataFrame([row])

    if os.path.isfile(HISTORY_PATH):
        try:
            old_history = pd.read_csv(HISTORY_PATH)
            history = pd.concat([old_history, new_row], ignore_index=True)
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            history = new_row
    else:
        history = new_row

    history.to_csv(HISTORY_PATH, index=False, encoding="utf-8-sig")




# ============================================================
# STYLE DE L'APPLICATION
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       FOND GENERAL
       ======================================================== */

    .stApp {
        background-color: #f5f8fc;
    }


    /* ========================================================
       SIDEBAR
       ======================================================== */

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #102a43 0%, #0b2239 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    /* Espacement et lisibilité du contenu de la barre latérale */
    [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        padding-top: 1.5rem;
    }

    [data-testid="stSidebar"] * {
        color: white;
    }

    /* Titre de l'application */
    [data-testid="stSidebar"] h1 {
        font-size: 1.55rem !important;
        line-height: 1.35 !important;
        font-weight: 750 !important;
        padding: 0.25rem 0 1rem 0;
        letter-spacing: 0.2px;
    }

    /* Espacement entre les éléments du menu */
    [data-testid="stSidebar"] [role="radiogroup"] {
        gap: 0.55rem !important;
        padding: 0.5rem 0 0.8rem 0;
    }

    [data-testid="stSidebar"] [role="radiogroup"] > label {
        background: rgba(255, 255, 255, 0.055);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 0.68rem 0.75rem !important;
        margin: 0 0 0.45rem 0 !important;
        min-height: 2.8rem;
        transition: background 0.18s ease, border-color 0.18s ease,
                    transform 0.18s ease;
    }

    [data-testid="stSidebar"] [role="radiogroup"] > label:hover {
        background: rgba(255, 255, 255, 0.13);
        border-color: rgba(255, 255, 255, 0.22);
        transform: translateX(2px);
    }

    [data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) {
        background: rgba(56, 139, 253, 0.24);
        border: 1px solid rgba(125, 180, 255, 0.65);
        box-shadow: inset 3px 0 0 #60a5fa;
    }

    [data-testid="stSidebar"] [role="radiogroup"] > label p {
        font-weight: 550;
        font-size: 0.94rem;
        margin: 0;
    }

    [data-testid="stSidebar"] hr {
        margin: 1.25rem 0;
        border-color: rgba(255, 255, 255, 0.14);
    }

    [data-testid="stSidebar"] h3 {
        font-size: 1.08rem !important;
        margin-top: 0.25rem;
        margin-bottom: 0.55rem;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        line-height: 1.55;
    }


    /* ========================================================
       TITRES
       ======================================================== */

    h1,
    h2,
    h3 {
        color: #102a43;
    }


    /* ========================================================
       METRIQUES
       ======================================================== */

    [data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #d9e2ec;
        border-radius: 12px;
        padding: 12px;
    }


    /* Nom de la métrique */

    [data-testid="stMetricLabel"] {
        font-size: 13px !important;
        color: #486581 !important;
    }


    /* Valeur de la métrique */

    [data-testid="stMetricValue"] {
        color: #1769e0 !important;
        font-size: 27px !important;
        font-weight: 700 !important;
    }


    /* ========================================================
       BOUTONS
       ======================================================== */

    .stButton > button {
        border-radius: 8px;
        min-height: 48px;
        font-weight: 600;
    }


    /* ========================================================
       INPUTS
       ======================================================== */

    .stSelectbox,
    .stNumberInput {
        margin-bottom: 5px;
    }


    /* ========================================================
       SEPARATEURS
       ======================================================== */

    hr {
        border-color: #d9e2ec;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("# 📊 Telco Churn AI")

    st.divider()

    selected_page = st.radio(
        "Navigation",
        [
            "🏠 Prédiction",
            "📊 Analyse du dataset",
            "📈 Visualisations",
            "ℹ️ À propos",
            "📜 Historique des prédictions",
            "🧠 Explication SHAP"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("### 💡 Objectif")

    st.write(
        "Identifier les clients présentant un risque "
        "de résiliation afin de permettre des actions "
        "de rétention ciblées."
    )





# ============================================================
# PAGE : HISTORIQUE DES PREDICTIONS
# ============================================================

if selected_page == "📜 Historique des prédictions":

    st.title("📜 Historique des prédictions")
    st.write(
        "Consultez les prédictions réalisées dans l'application. "
        "Chaque prédiction est enregistrée dans un fichier CSV local."
    )

    if os.path.isfile(HISTORY_PATH):
        try:
            history_df = pd.read_csv(HISTORY_PATH)

            if history_df.empty:
                st.info("Aucune prédiction n'a encore été enregistrée.")
            else:
                st.subheader("📌 Indicateurs de l'historique")
                total_predictions = len(history_df)
                churn_predictions = (history_df["Prediction"] == "CHURN").sum()
                no_churn_predictions = (history_df["Prediction"] == "NO CHURN").sum()

                col1, col2, col3 = st.columns(3)
                col1.metric("Nombre de prédictions", total_predictions)
                col2.metric("Churn prédit", int(churn_predictions))
                col3.metric("No Churn prédit", int(no_churn_predictions))

                st.divider()
                st.subheader("🔍 Détail des prédictions")
                st.dataframe(
                    history_df.iloc[::-1],
                    use_container_width=True,
                    hide_index=True
                )

                csv_data = history_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="⬇️ Télécharger l'historique (CSV)",
                    data=csv_data,
                    file_name="predictions_history.csv",
                    mime="text/csv",
                    use_container_width=True
                )

                st.divider()
                st.subheader("🗑️ Gestion de l'historique")
                st.warning(
                    "La suppression efface définitivement le fichier local "
                    "contenant les prédictions enregistrées."
                )
                if st.button("Effacer l'historique"):
                    os.remove(HISTORY_PATH)
                    st.success("L'historique a été supprimé.")
                    st.rerun()

        except (pd.errors.EmptyDataError, pd.errors.ParserError, KeyError) as error:
            st.error(f"Impossible de lire l'historique : {error}")
    else:
        st.info(
            "L'historique est vide pour le moment. "
            "Effectuez une prédiction pour commencer à l'alimenter."
        )

    st.stop()


# ============================================================
# PAGE : EXPLICATION SHAP
# ============================================================

if selected_page == "🧠 Explication SHAP":

    st.title("🧠 Explication des prédictions avec SHAP")
    st.write(
        "Sélectionnez une prédiction de l'historique pour comprendre les "
        "facteurs qui ont contribué au résultat. Vous pouvez aussi consulter "
        "la dernière prédiction de la session."
    )

    # Les variables attendues par le modèle entraîné.
    feature_columns = [
        "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
        "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
        "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
        "MonthlyCharges", "TotalCharges"
    ]

    selected_client_data = None
    selected_result = None
    selected_probability = None
    selected_date = None

    # 1. Proposer les prédictions enregistrées dans le CSV.
    if os.path.isfile(HISTORY_PATH):
        try:
            history_for_shap = pd.read_csv(HISTORY_PATH)

            missing_columns = [
                column for column in feature_columns + [
                    "Date", "Prediction", "Probabilite_Churn"
                ]
                if column not in history_for_shap.columns
            ]

            if not history_for_shap.empty and not missing_columns:
                history_for_shap = history_for_shap.reset_index(drop=True)
                history_for_shap["_history_id"] = history_for_shap.index
                history_for_shap = history_for_shap.sort_values(
                    "Date", ascending=False
                ).reset_index(drop=True)

                st.subheader("📂 Sélectionner une prédiction")
                history_options = []
                for _, history_row in history_for_shap.iterrows():
                    history_options.append(
                        f"{history_row['Date']} | "
                        f"{history_row['Prediction']} | "
                        f"{float(history_row['Probabilite_Churn']):.2f}%"
                    )

                selected_option = st.selectbox(
                    "Prédictions enregistrées",
                    history_options,
                    key="shap_history_selection"
                )
                selected_index = history_options.index(selected_option)
                selected_row = history_for_shap.iloc[selected_index]

                selected_client_data = {
                    column: selected_row[column]
                    for column in feature_columns
                }
                selected_result = str(selected_row["Prediction"])
                selected_probability = (
                    float(selected_row["Probabilite_Churn"]) / 100.0
                )
                selected_date = str(selected_row["Date"])

            elif not history_for_shap.empty and missing_columns:
                st.warning(
                    "Certaines anciennes lignes de l'historique ne contiennent "
                    "pas toutes les informations nécessaires à SHAP. "
                    "Colonnes manquantes : " + ", ".join(missing_columns)
                )

        except (pd.errors.EmptyDataError, pd.errors.ParserError, ValueError, TypeError) as error:
            st.warning(f"Impossible de lire l'historique : {error}")

    # 2. Si l'historique n'est pas exploitable, utiliser la dernière
    # prédiction faite dans la session.
    if selected_client_data is None:
        last_prediction = st.session_state.get("last_prediction")
        if last_prediction:
            selected_client_data = last_prediction["client_data"].copy()
            selected_result = last_prediction["result"]
            selected_probability = float(last_prediction["probability"])
            selected_date = "Dernière prédiction de cette session"

    if selected_client_data is None:
        st.info(
            "Aucune prédiction n'est disponible pour l'explication. "
            "Effectuez d'abord une prédiction dans la page « Prédiction »."
        )
        st.stop()

    # Restaurer les types numériques attendus par le pipeline.
    selected_client_data["SeniorCitizen"] = int(
        float(selected_client_data["SeniorCitizen"])
    )
    selected_client_data["tenure"] = int(
        float(selected_client_data["tenure"])
    )
    selected_client_data["MonthlyCharges"] = float(
        selected_client_data["MonthlyCharges"]
    )
    selected_client_data["TotalCharges"] = float(
        selected_client_data["TotalCharges"]
    )

    client_df = pd.DataFrame([selected_client_data])
    probability_percent = float(selected_probability) * 100

    st.caption(f"Prédiction sélectionnée : {selected_date}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Prédiction", selected_result)
    col2.metric("Probabilité de churn", f"{probability_percent:.2f}%")
    col3.metric("Seuil de décision", f"{threshold:.2f}")

    with st.expander("👤 Voir les informations du client analysé"):
        client_details_df = pd.DataFrame(
            [
                {"Variable": column, "Valeur": value}
                for column, value in selected_client_data.items()
            ]
        )
        st.dataframe(
            client_details_df,
            use_container_width=True,
            hide_index=True
        )

    st.divider()
    st.subheader("Comment lire le graphique ?")
    st.write(
        "Une contribution SHAP positive pousse la sortie du modèle vers le "
        "churn ; une contribution négative la pousse dans le sens opposé. "
        "Ces contributions expliquent le modèle pour ce client, sans démontrer "
        "de lien de causalité."
    )

    try:
        explanation_df, shap_base_value, shap_probability = explain_churn(client_df)

        top_shap = (
            explanation_df.assign(
                Absolute_SHAP=explanation_df["SHAP_Value"].abs()
            )
            .nlargest(10, "Absolute_SHAP")
            .sort_values("SHAP_Value", ascending=True)
            .copy()
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(top_shap["Feature"], top_shap["SHAP_Value"])
        ax.axvline(0, linewidth=1)
        ax.set_xlabel("Contribution SHAP (classe Churn)")
        ax.set_ylabel("Variables")
        ax.set_title("Facteurs influençant cette prédiction")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        positive = (
            explanation_df[explanation_df["SHAP_Value"] > 0]
            .sort_values("SHAP_Value", ascending=False)
            .head(3)
        )
        negative = (
            explanation_df[explanation_df["SHAP_Value"] < 0]
            .sort_values("SHAP_Value", ascending=True)
            .head(3)
        )

        st.subheader("📝 Interprétation automatique")
        st.write(
            f"Le modèle classe ce client dans la catégorie **{selected_result}** "
            f"avec une probabilité estimée de **{probability_percent:.2f}%**. "
            f"Le seuil de décision utilisé est **{threshold:.2f}**."
        )

        col_positive, col_negative = st.columns(2)
        with col_positive:
            st.markdown("**Facteurs poussant vers le churn**")
            if positive.empty:
                st.write("Aucune contribution positive parmi les variables calculées.")
            else:
                for _, row in positive.iterrows():
                    st.write(
                        f"• **{row['Feature']}** "
                        f"(contribution : +{row['SHAP_Value']:.4f})"
                    )

        with col_negative:
            st.markdown("**Facteurs poussant dans le sens opposé**")
            if negative.empty:
                st.write("Aucune contribution négative parmi les variables calculées.")
            else:
                for _, row in negative.iterrows():
                    st.write(
                        f"• **{row['Feature']}** "
                        f"(contribution : {row['SHAP_Value']:.4f})"
                    )

        with st.expander("Voir toutes les valeurs SHAP"):
            st.dataframe(
                explanation_df[["Feature", "SHAP_Value"]].rename(columns={
                    "Feature": "Variable",
                    "SHAP_Value": "Contribution SHAP"
                }),
                use_container_width=True,
                hide_index=True
            )

        st.caption(
            "Les valeurs SHAP décrivent la contribution des variables à la "
            "sortie du modèle. Elles ne prouvent pas une causalité et ne "
            "constituent pas une certitude sur le comportement futur du client."
        )

    except Exception as error:
        st.error(f"Impossible de calculer l'explication SHAP : {error}")

    st.stop()


# ============================================================
# PAGE : ANALYSE DU DATASET
# ============================================================

if selected_page == "📊 Analyse du dataset":

    st.title("📊 Analyse du dataset")

    st.write(
        "Explorez les données des clients et "
        "les caractéristiques associées à la résiliation."
    )

    # Charger le dataset
    dataset_path = os.path.join(
        PROJECT_ROOT,
        "Data",
        "churn.csv"
    )

    df = pd.read_csv(dataset_path)

    # Nettoyage des données
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Statistiques principales
    total_clients = len(df)
    churn_clients = (df["Churn"] == "Yes").sum()
    non_churn_clients = (df["Churn"] == "No").sum()
    churn_rate = churn_clients / total_clients * 100

    st.subheader("📌 Indicateurs clés")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Nombre de clients", f"{total_clients:,}")
    col2.metric("Clients résiliés", f"{churn_clients:,}")
    col3.metric("Clients fidèles", f"{non_churn_clients:,}")
    col4.metric("Taux de résiliation", f"{churn_rate:.2f}%")

    st.divider()

    # Répartition du churn
    st.subheader("📊 Répartition des clients")

    churn_distribution = (
        df["Churn"]
        .value_counts()
        .rename(index={"No": "No Churn", "Yes": "Churn"})
    )

    st.bar_chart(churn_distribution)

    st.divider()

    # Taux de churn par contrat
    st.subheader("📄 Taux de résiliation selon le contrat")

    contract_churn = (
        df.groupby("Contract")["Churn"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .sort_values(ascending=False)
    )

    st.bar_chart(contract_churn)

    st.divider()

    # Ancienneté moyenne selon le churn
    st.subheader("⏳ Ancienneté moyenne des clients")

    tenure_by_churn = (
        df.groupby("Churn")["tenure"]
        .mean()
        .rename(index={"No": "No Churn", "Yes": "Churn"})
    )

    st.bar_chart(tenure_by_churn)

    st.divider()

    # Aperçu des données
    st.subheader("🔍 Aperçu du dataset")

    st.dataframe(df.head(10), use_container_width=True)

    st.caption(f"Dimensions du dataset : {df.shape[0]} lignes × {df.shape[1]} colonnes")

    st.stop()


# ============================================================
# PAGE : VISUALISATIONS
# ============================================================

if selected_page == "📈 Visualisations":

    st.title("📈 Visualisations du modèle")

    st.write(
        "Cette page présente les performances du modèle "
        "Random Forest sur le jeu de test."
    )

    st.subheader("📊 Métriques du modèle")

    metric1, metric2, metric3, metric4, metric5 = st.columns(5)

    metric1.metric("Accuracy", "76.72%")
    metric2.metric("Precision", "54.75%")
    metric3.metric("Recall", "70.86%")
    metric4.metric("F1-score", "61.77%")
    metric5.metric("ROC-AUC", "84.14%")

    st.divider()

    # Deux colonnes pour les graphiques
    col1, col2 = st.columns(2, gap="large")

    # Matrice de confusion
    with col1:

        st.subheader("🔢 Matrice de confusion")

        confusion_matrix_paths = [
            os.path.join(PROJECT_ROOT, "Models", "cm.png"),
            os.path.join(PROJECT_ROOT, "Models", "cm"),
            os.path.join(PROJECT_ROOT, "Models", "confusion_matrix.png")
        ]

        confusion_matrix_path = next(
            (
                path for path in confusion_matrix_paths
                if os.path.isfile(path)
            ),
            None
        )

        if confusion_matrix_path:
            st.image(
                confusion_matrix_path,
                use_container_width=True
            )
        else:
            st.warning(
                "La matrice de confusion est introuvable. "
                "Vérifiez le dossier Models."
            )

    # Courbe ROC
    with col2:

        st.subheader("📈 Courbe ROC")

        roc_paths = [
            os.path.join(PROJECT_ROOT, "Models", "roc_curve.png"),
            os.path.join(PROJECT_ROOT, "Models", "roc_curve")
        ]

        roc_path = next(
            (
                path for path in roc_paths
                if os.path.isfile(path)
            ),
            None
        )

        if roc_path:
            st.image(
                roc_path,
                use_container_width=True
            )
        else:
            st.warning(
                "La courbe ROC est introuvable. "
                "Vérifiez le dossier Models."
            )

    st.divider()

    # Importance des variables (Explainable AI)
    st.subheader("🧠 Importance des variables — Random Forest")
    st.write(
        "Ce graphique présente les variables qui contribuent le plus "
        "aux décisions globales du modèle. Une importance élevée ne "
        "signifie pas qu'une variable cause directement la résiliation."
    )

    feature_importance_path = os.path.join(
        PROJECT_ROOT,
        "Models",
        "feature_importance.png"
    )

    if os.path.isfile(feature_importance_path):
        st.image(
            feature_importance_path,
            use_container_width=True
        )
        st.caption(
            "Importance calculée à partir du modèle Random Forest entraîné."
        )
    else:
        st.warning(
            "Le graphique d'importance des variables est introuvable. "
            "Exécutez src/train.py pour le générer."
        )

    st.stop()


# ============================================================
# PAGE : À PROPOS
# ============================================================

if selected_page == "ℹ️ À propos":

    st.title("ℹ️ À propos de Telco Churn AI")

    st.write(
        "Telco Churn AI est une application de Machine Learning "
        "qui estime la probabilité qu'un client résilie son "
        "abonnement téléphonique."
    )

    st.divider()

    st.subheader("🎯 Objectif du projet")

    st.write(
        "L'objectif est d'aider les entreprises de télécommunications "
        "à identifier les clients susceptibles de résilier leur "
        "abonnement et à envisager des actions de fidélisation."
    )

    st.subheader("⚙️ Fonctionnement")

    st.markdown(
        """
        1. L'utilisateur renseigne les caractéristiques d'un client.
        2. Les données sont transmises au modèle entraîné.
        3. Le modèle estime la probabilité de résiliation.
        4. L'application affiche la prédiction et le niveau de risque.
        """
    )

    st.subheader("🧰 Technologies utilisées")

    st.markdown(
        """
        - **Python** : langage de programmation.
        - **Pandas** : préparation et analyse des données.
        - **Scikit-learn** : entraînement et évaluation du modèle.
        - **Random Forest** : algorithme de classification.
        - **Streamlit** : création de l'interface web.
        - **Matplotlib** : génération des graphiques.
        """
    )

    st.subheader("📁 Dataset")

    st.write(
        "Le projet utilise le dataset IBM Telco Customer Churn, "
        "qui contient des informations sur les clients, leurs "
        "abonnements, leurs services et leur résiliation."
    )

    st.divider()

    st.caption("Telco Churn AI — Projet de Machine Learning")

    st.stop()




# ============================================================
# TITRE PRINCIPAL
# ============================================================

st.title(
    "📊 Telco Churn AI"
)

st.subheader(
    "Prédiction du risque de résiliation client"
)

st.write(
    "Cette application utilise un modèle de Machine Learning "
    "pour estimer la probabilité qu'un client résilie son abonnement."
)


# ============================================================
# CREATION DES DEUX COLONNES PRINCIPALES
# ============================================================

left_column, right_column = st.columns(
    [1, 1],
    gap="large"
)


# ============================================================
# ============================================================
# COLONNE GAUCHE
# INFORMATIONS CLIENT
# ============================================================
# ============================================================

with left_column:

    st.header(
        "👤 Informations client"
    )

    st.write(
        "Remplissez les informations ci-dessous "
        "pour prédire le risque de churn."
    )


    # ========================================================
    # INFORMATIONS GENERALES
    # ========================================================

    st.markdown(
        "### 👤 Informations générales"
    )

    col1, col2, col3 = st.columns(3)


    with col1:

        gender = st.selectbox(
            "Genre",
            [
                "Male",
                "Female"
            ]
        )


    with col2:

        senior_citizen = st.selectbox(
            "Senior Citizen",
            [0, 1],
            format_func=lambda x:
                "Oui" if x == 1 else "Non"
        )


    with col3:

        partner = st.selectbox(
            "Partenaire",
            [
                "Yes",
                "No"
            ]
        )


    col4, col5 = st.columns(2)


    with col4:

        dependents = st.selectbox(
            "Personnes à charge",
            [
                "Yes",
                "No"
            ]
        )


    with col5:

        tenure = st.number_input(
            "Ancienneté (mois)",
            min_value=0,
            max_value=100,
            value=12,
            step=1
        )


    # ========================================================
    # SERVICES TELEPHONIQUES
    # ========================================================

    st.markdown(
        "### 📞 Services téléphoniques"
    )

    col1, col2 = st.columns(2)


    with col1:

        phone_service = st.selectbox(
            "Service téléphonique",
            [
                "Yes",
                "No"
            ]
        )


    with col2:

        multiple_lines = st.selectbox(
            "Lignes multiples",
            [
                "Yes",
                "No",
                "No phone service"
            ]
        )


    # ========================================================
    # SERVICES INTERNET
    # ========================================================

    st.markdown(
        "### 🌐 Services Internet"
    )

    col1, col2, col3 = st.columns(3)


    with col1:

        internet_service = st.selectbox(
            "Type de connexion internet",
            [
                "DSL",
                "Fiber optic",
                "No"
            ]
        )


    with col2:

        online_security = st.selectbox(
            "Sécurité en ligne",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )


    with col3:

        online_backup = st.selectbox(
            "Sauvegarde en ligne",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )


    col1, col2, col3 = st.columns(3)


    with col1:

        device_protection = st.selectbox(
            "Protection des appareils",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )


    with col2:

        tech_support = st.selectbox(
            "Support technique",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )


    with col3:

        streaming_tv = st.selectbox(
            "Streaming TV",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )


    streaming_movies = st.selectbox(
        "Streaming Movies",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )


    # ========================================================
    # CONTRAT ET FACTURATION
    # ========================================================

    st.markdown(
        "### 📄 Contrat et facturation"
    )

    col1, col2, col3 = st.columns(3)


    with col1:

        contract = st.selectbox(
            "Type de contrat",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ]
        )


    with col2:

        paperless_billing = st.selectbox(
            "Facturation sans papier",
            [
                "Yes",
                "No"
            ]
        )


    with col3:

        payment_method = st.selectbox(
            "Méthode de paiement",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )


    # ========================================================
    # INFORMATIONS FINANCIERES
    # ========================================================

    st.markdown(
        "### 💰 Informations financières"
    )

    col1, col2 = st.columns(2)


    with col1:

        monthly_charges = st.number_input(
            "Charges mensuelles ($)",
            min_value=0.0,
            max_value=200.0,
            value=70.0,
            step=0.5
        )


    with col2:

        total_charges = st.number_input(
            "Charges totales ($)",
            min_value=0.0,
            max_value=10000.0,
            value=350.0,
            step=10.0
        )


    # ========================================================
    # BOUTON DE PREDICTION
    # ========================================================

    st.divider()

    predict_button = st.button(
        "🚀 Prédire le risque de Churn",
        use_container_width=True
    )


# ============================================================
# ============================================================
# COLONNE DROITE
# RESULTAT
# ============================================================
# ============================================================

with right_column:

    st.header(
        "🎯 Résultat de la prédiction"
    )


    # ========================================================
    # AVANT LA PREDICTION
    # ========================================================

    if not predict_button:

        st.info(
            "👆 Remplissez les informations du client "
            "puis cliquez sur « Prédire le risque de Churn »."
        )

        st.divider()

        st.header(
            "📊 Métriques du modèle"
        )

        st.caption(
            "Performances obtenues sur le jeu de test."
        )


        metric1, metric2 = st.columns(2)


        with metric1:

            st.metric(
                "Accuracy",
                "76.72%"
            )


        with metric2:

            st.metric(
                "Precision",
                "54.75%"
            )


        metric3, metric4 = st.columns(2)


        with metric3:

            st.metric(
                "Recall",
                "70.86%"
            )


        with metric4:

            st.metric(
                "F1-score",
                "61.77%"
            )


        metric5 = st.columns(1)[0]


        with metric5:

            st.metric(
                "ROC-AUC",
                "84.14%"
            )


    # ========================================================
    # APRES LA PREDICTION
    # ========================================================

    else:

        # ====================================================
        # CREATION DES DONNEES CLIENT
        # ====================================================

        client_data = {

            "gender": gender,

            "SeniorCitizen": senior_citizen,

            "Partner": partner,

            "Dependents": dependents,

            "tenure": tenure,

            "PhoneService": phone_service,

            "MultipleLines": multiple_lines,

            "InternetService": internet_service,

            "OnlineSecurity": online_security,

            "OnlineBackup": online_backup,

            "DeviceProtection": device_protection,

            "TechSupport": tech_support,

            "StreamingTV": streaming_tv,

            "StreamingMovies": streaming_movies,

            "Contract": contract,

            "PaperlessBilling": paperless_billing,

            "PaymentMethod": payment_method,

            "MonthlyCharges": monthly_charges,

            "TotalCharges": total_charges
        }


        # ====================================================
        # DATAFRAME
        # ====================================================

        client_df = pd.DataFrame(
            [client_data]
        )


        # ====================================================
        # PREDICTION
        # ====================================================

        result, probability = predict_churn(
            client_df
        )


        # ====================================================
        # PROBABILITE
        # ====================================================

        probability = float(
            probability
        )

        # Conserver la dernière prédiction pour la page SHAP séparée.
        st.session_state["last_prediction"] = {
            "client_data": client_data.copy(),
            "result": result,
            "probability": probability
        }

        # Enregistrer la prédiction après le clic sur le bouton.
        save_prediction(
            client_data=client_data,
            result=result,
            probability=probability
        )

        probability_percent = (
            probability * 100
        )


        # ====================================================
        # NIVEAU DE RISQUE
        # ====================================================

        if probability_percent >= 70:

            risk_level = "ÉLEVÉ"
            risk_icon = "🔴"

        elif probability_percent >= 40:

            risk_level = "MODÉRÉ"
            risk_icon = "🟠"

        else:

            risk_level = "FAIBLE"
            risk_icon = "🟢"


        # ====================================================
        # RESULTAT PRINCIPAL
        # ====================================================

        if result == "CHURN":

            st.error(
                "⚠️ RISQUE ÉLEVÉ DE RÉSILIATION — CHURN"
            )

        else:

            st.success(
                "✅ RISQUE DE RÉSILIATION NON DÉTECTÉ — NO CHURN"
            )


        # ====================================================
        # PREDICTION ET PROBABILITE
        # ====================================================

        result_col1, result_col2 = st.columns(2)


        with result_col1:

            st.metric(
                "Prédiction",
                result
            )


        with result_col2:

            st.metric(
                "Probabilité de Churn",
                f"{probability_percent:.2f}%"
            )


        # ====================================================
        # NIVEAU DE RISQUE
        # ====================================================

        st.subheader(
            f"{risk_icon} Niveau de risque : {risk_level}"
        )

        st.progress(
            probability
        )

        st.caption(
            f"Probabilité estimée : "
            f"{probability_percent:.2f}%"
        )


        # ====================================================
        # RECOMMANDATION
        # ====================================================

        st.subheader(
            "💡 Recommandation"
        )


        if result == "CHURN":

            st.warning(
                "Ce client présente un risque élevé de "
                "résiliation. Des actions de rétention "
                "ciblées peuvent être envisagées : "
                "offres spéciales, meilleur support ou "
                "avantages personnalisés."
            )

        else:

            st.info(
                "Le modèle ne détecte pas actuellement "
                "un risque élevé de résiliation pour ce client."
            )


        # ====================================================
        # METRIQUES DU MODELE
        # ====================================================

        st.divider()

        st.header(
            "📊 Métriques du modèle"
        )

        st.caption(
            "Performances obtenues sur le jeu de test."
        )


        # Première ligne
        metric1, metric2, metric3 = st.columns(3)


        with metric1:

            st.metric(
                "Accuracy",
                "76.72%"
            )


        with metric2:

            st.metric(
                "Precision",
                "54.75%"
            )


        with metric3:

            st.metric(
                "Recall",
                "70.86%"
            )


        # Deuxième ligne
        metric4, metric5 = st.columns(2)


        with metric4:

            st.metric(
                "F1-score",
                "61.77%"
            )


        with metric5:

            st.metric(
                "ROC-AUC",
                "84.14%"
            )


        # ====================================================
        # EVALUATION DU MODELE
        # ====================================================

        st.divider()

        st.header(
            "📈 Évaluation du modèle"
        )


        # ====================================================
        # COLONNES POUR LES GRAPHIQUES
        # ====================================================

        chart_col1, chart_col2 = st.columns(
            2,
            gap="medium"
        )


        # ====================================================
        # MATRICE DE CONFUSION
        # ====================================================

        with chart_col1:

            st.subheader(
                "🔢 Matrice de confusion"
            )


            confusion_matrix_paths = [

                os.path.join(
                    PROJECT_ROOT,
                    "Models",
                    "cm.png"
                ),

                os.path.join(
                    PROJECT_ROOT,
                    "Models",
                    "cm"
                ),

                os.path.join(
                    PROJECT_ROOT,
                    "Models",
                    "confusion_matrix.png"
                )
            ]


            confusion_matrix_path = None


            for path in confusion_matrix_paths:

                if os.path.exists(path):

                    confusion_matrix_path = path

                    break


            if confusion_matrix_path is not None:

                st.image(
                    confusion_matrix_path,
                    use_container_width=True
                )

            else:

                st.warning(
                    "La matrice de confusion "
                    "n'a pas été trouvée dans le dossier Models."
                )


        # ====================================================
        # COURBE ROC
        # ====================================================

        with chart_col2:

            st.subheader(
                "📈 Courbe ROC"
            )


            roc_paths = [

                os.path.join(
                    PROJECT_ROOT,
                    "Models",
                    "roc_curve.png"
                ),

                os.path.join(
                    PROJECT_ROOT,
                    "Models",
                    "roc_curve"
                )
            ]


            roc_path = None


            for path in roc_paths:

                if os.path.exists(path):

                    roc_path = path

                    break


            if roc_path is not None:

                st.image(
                    roc_path,
                    use_container_width=True
                )

            else:

                st.warning(
                    "La courbe ROC "
                    "n'a pas été trouvée dans le dossier Models."
                )

