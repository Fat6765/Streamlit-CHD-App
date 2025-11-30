

import streamlit as st
import pandas as pd
import joblib
import os

# --- 1. CONFIGURATION DE LA PAGE (Style du tutoriel) ---
st.set_page_config(
    page_title="Prédiction du risque de CHD",
    page_icon="🫀",
    layout="centered"
)

# --- 2. FONCTION PERSONNALISÉE (INDISPENSABLE pour votre modèle) ---
# Cette partie n'est pas dans le tuto générique, mais est obligatoire pour VOTRE modèle
def lowercase_variable(X):
    return X.apply(lambda x: x.str.lower() if x.dtype == "object" else x)

# --- 3. CHARGEMENT DU MODÈLE (Avec Cache comme dans le tuto) ---
@st.cache_resource
def load_model():
    # On utilise le chemin complet qui fonctionne chez vous
    path = 'Model.pkl'
    return joblib.load(path)

try:
    model = load_model()
except Exception as e:
    st.error(f"Erreur de chargement du modèle : {e}")
    st.stop()

# --- 4. INTERFACE UTILISATEUR ---
st.title("🩺 Prédiction du risque cardiaque")
st.markdown("""
Cette application utilise un modèle de Machine Learning (Pipeline + ACP) 
pour estimer le risque de maladie cardiaque (CHD).
""")

st.subheader("Saisissez les paramètres cliniques :")

# Organisation en colonnes pour un rendu plus pro (comme souvent dans les tutos)
col1, col2 = st.columns(2)

with col1:
    sbp = st.number_input("Pression artérielle (sbp)", value=130, min_value=80, max_value=250)
    ldl = st.number_input("Cholestérol LDL", value=4.0, format="%.2f")
    adiposity = st.number_input("Adiposité", value=25.0, format="%.2f")

with col2:
    famhist = st.selectbox("Antécédents familiaux", ["Present", "Absent"])
    obesity = st.number_input("Obésité", value=25.0, format="%.2f")
    age = st.number_input("Âge", value=45, min_value=15, max_value=100)

# Création du DataFrame
input_data = pd.DataFrame({
    'sbp': [sbp],
    'ldl': [ldl],
    'adiposity': [adiposity],
    'famhist': [famhist],
    'obesity': [obesity],
    'age': [age]
})

# --- 5. PRÉDICTION ET RÉSULTATS ---
if st.button("Lancer la prédiction 🚀", type="primary"):
    try:
        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0][1]
        
        st.divider() # Ligne de séparation
        
        if prediction == 1:
            st.error(f"⚠️ **RÉSULTAT : Risque Élevé détecté**")
            st.write(f"Probabilité estimée : **{proba:.1%}**")
            st.info("Conseil : Veuillez consulter un cardiologue pour des examens approfondis.")
        else:
            st.success(f"✅ **RÉSULTAT : Faible risque**")
            st.write(f"Probabilité estimée : **{proba:.1%}**")
            
    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")
