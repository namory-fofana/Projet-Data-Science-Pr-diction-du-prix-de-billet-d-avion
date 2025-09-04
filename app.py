from flask import Flask, request, render_template
from flask_cors import cross_origin
import sklearn
import pickle
import pandas as pd
import numpy as np 
import os

# Pour Render.com, utiliser le port fourni par l'environnement
port = int(os.environ.get('PORT', 5000))

app = Flask(__name__)

# Charger le modèle
model = pickle.load(open("flight_xgb.pkl", "rb"))

# Charger les colonnes du modèle (avec pickle pour éviter les problèmes)
with open('model_columns.pkl', 'rb') as f:
    feature_names = pickle.load(f)


@app.route("/")
@cross_origin()
def home():
    return render_template("home.html")


@app.route("/predict", methods=["POST"])
@cross_origin()
def predict():
    if request.method == "POST":

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

    return render_template("home.html")


if __name__ == "__main__":
     app.run(host='0.0.0.0', port=port, threaded=True)