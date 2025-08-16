
from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
import os, requests

app = Flask(__name__)

# Optional ML model (if present)
MODEL_PATH = "lstm_weather_model.h5"
SCALER_PATH = "scaler.save"
model = None
scaler = None
try:
    from keras.models import load_model
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("Loaded ML model and scaler.")
    else:
        print("Model/scaler not found. Using simple moving-average fallback.")
except Exception as e:
    print("Keras not available or failed to load model. Using fallback.", e)

def simple_predict(seq):
    seq = np.array(seq, dtype=float)
    weights = np.linspace(1, 2, num=len(seq))
    return float(np.average(seq, weights=weights))

# City coordinates for Open-Meteo
CITY_COORDS = {
    "hyd": {"name": "Hyderabad", "country": "IN", "lat": 17.3850, "lon": 78.4867},
    "vij": {"name": "Vijayawada", "country": "IN", "lat": 16.5062, "lon": 80.6480},
    "ndl": {"name": "Nellore", "country": "IN", "lat": 14.4426, "lon": 79.9865},
    "vskp": {"name": "Visakhapatnam", "country": "IN", "lat": 17.6868, "lon": 83.2185},
    "blr": {"name": "Bengaluru", "country": "IN", "lat": 12.9716, "lon": 77.5946},
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/cities")
def api_cities():
    cities = [{"id": cid, "name": c["name"], "country": c["country"]} for cid, c in CITY_COORDS.items()]
    return jsonify({"cities": cities})

@app.route("/api/weather/<city_id>")
def api_weather(city_id):
    if city_id not in CITY_COORDS:
        return jsonify({"error": "city not found"}), 404
    city = CITY_COORDS[city_id]
    url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={city['lat']}&longitude={city['lon']}"
        f"&current=temperature_2m&daily=temperature_2m_max,temperature_2m_min&timezone=auto"
    )
    r = requests.get(url)
    data = r.json()

    current = data["current"]["temperature_2m"]
    five_day_avg = float(np.mean(data["daily"]["temperature_2m_max"][:5]))

    city_info = {"id": city_id, "name": city["name"], "country": city["country"], "condition": "sunny"}

    return jsonify({"city": city_info, "current": current, "five_day_avg": five_day_avg})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    seq = data.get("temp_sequence", [])
    if not isinstance(seq, list) or len(seq) != 10:
        return jsonify({"error":"Provide exactly 10 numeric values in 'temp_sequence'."}), 400
    try:
        seq = [float(x) for x in seq]
    except:
        return jsonify({"error":"Non-numeric value in sequence."}), 400

    if model is not None and scaler is not None:
        arr = np.array(seq).reshape(-1,1)
        scaled = scaler.transform(arr)
        X_input = scaled.reshape(1, 10, 1)
        pred_scaled = model.predict(X_input, verbose=0)
        pred = scaler.inverse_transform(pred_scaled)
        value = float(pred[0][0])
    else:
        value = simple_predict(seq)
    return jsonify({"predicted_temperature": value})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
