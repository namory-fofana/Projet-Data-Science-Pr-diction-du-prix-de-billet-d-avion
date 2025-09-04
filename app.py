from flask import Flask, request, render_template
from flask_cors import cross_origin
import sklearn
import pickle
import pandas as pd
import numpy as np 
import os

app = Flask(__name__)

# Déplace cette ligne DANS la condition __main__
# port = int(os.environ.get('PORT', 5000))

# Fonction pour charger le modèle avec gestion d'erreur
def load_model():
    try:
        model_path = "flight_xgb.pkl"
        if not os.path.exists(model_path):
            print(f"❌ ERREUR: Fichier {model_path} non trouvé!")
            print(f"Contenu du répertoire courant: {os.listdir('.')}")
            return None
        return pickle.load(open(model_path, "rb"))
    except Exception as e:
        print(f"❌ ERREUR de chargement du modèle: {str(e)}")
        return None

# Fonction pour charger les colonnes avec gestion d'erreur
def load_feature_names():
    try:
        feature_path = "model_columns.pkl"
        if not os.path.exists(feature_path):
            print(f"❌ ERREUR: Fichier {feature_path} non trouvé!")
            print(f"Contenu du répertoire courant: {os.listdir('.')}")
            return []
        with open(feature_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"❌ ERREUR de chargement des colonnes: {str(e)}")
        return []

# Charger le modèle et les colonnes
model = load_model()
feature_names = load_feature_names()

@app.route("/")
@cross_origin()
def home():
    # Vérifie si le modèle est chargé
    if model is None or not feature_names:
        return "Erreur: Modèle non chargé. Vérifie les logs Render.", 500
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
@cross_origin()
def predict():
    # Vérifie si le modèle est chargé
    if model is None or not feature_names:
        return "Erreur: Modèle non chargé. Vérifie les logs Render.", 500
        
    if request.method == "POST":
        try:
            # --- Extraction des données ---
            date_dep = request.form["Dep_Time"]
            Journey_day = int(pd.to_datetime(date_dep).day)
            Journey_month = int(pd.to_datetime(date_dep).month)
            Dep_hour = int(pd.to_datetime(date_dep).hour)
            Dep_min = int(pd.to_datetime(date_dep).minute)

            date_arr = request.form["Arrival_Time"]
            Arrival_hour = int(pd.to_datetime(date_arr).hour)
            Arrival_min = int(pd.to_datetime(date_arr).minute)

            dur_hour = abs(Arrival_hour - Dep_hour)
            dur_min = abs(Arrival_min - Dep_min)

            Total_stops = int(request.form["stops"])

            # --- One-Hot Encoding ---
            airlines = ['Air India', 'GoAir', 'IndiGo', 'Jet Airways', 'Jet Airways Business',
                        'Multiple carriers', 'Multiple carriers Premium economy', 'SpiceJet',
                        'Trujet', 'Vistara', 'Vistara Premium economy']
            sources = ['Chennai', 'Delhi', 'Kolkata', 'Mumbai']
            destinations = ['Cochin', 'Delhi', 'Hyderabad', 'Kolkata', 'New Delhi']

            input_dict = {
                'Total_Stops': Total_stops,
                'Journey_day': Journey_day,
                'Journey_month': Journey_month,
                'Dep_hour': Dep_hour,
                'Dep_min': Dep_min,
                'Arrival_hour': Arrival_hour,
                'Arrival_min': Arrival_min,
                'Duration_hours': dur_hour,
                'Duration_mins': dur_min,
            }

            # Ajouter les one-hot encodings
            for a in airlines:
                input_dict[f"Airline_{a.replace(' ', '_')}"] = 1 if request.form['airline'] == a else 0

            for s in sources:
                input_dict[f"Source_{s}"] = 1 if request.form['Source'] == s else 0

            for d in destinations:
                input_dict[f"Destination_{d.replace(' ', '_')}"] = 1 if request.form['Destination'] == d else 0

            # Remplir les colonnes manquantes avec 0
            for col in feature_names:
                if col not in input_dict:
                    input_dict[col] = 0

            # Créer le DataFrame dans le bon ordre
            input_df = pd.DataFrame([input_dict], columns=feature_names)

            # --- Prédiction ---
            prediction = model.predict(input_df)
            output = round(np.exp(prediction[0]), 2)  # dé-logarithme

            return render_template('home.html', prediction_text=f"Le prix de votre vol est de {output} roupies.")
            
        except Exception as e:
            print(f"❌ ERREUR de prédiction: {str(e)}")
            return f"Erreur de prédiction: {str(e)}", 500

    return render_template("home.html")

if __name__ == "__main__":
    # Déplace la définition du port ici
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, threaded=True)