
# AI Weather • Simple Forecast (No API key needed)

A ready-to-run Flask app with a clean UI, local icons/images, city selector, refresh button, and optional ML prediction.

## Features
- ✨ Simple, responsive UI served by Flask
- 📍 Working city dropdown + Refresh button
- 🌡️ Shows current temp & 5‑day average (mocked locally so it works offline)
- 🤖 Optional ML prediction endpoint that uses your LSTM model **if present**, otherwise falls back to a smart moving‑average
- 🖼️ Local SVG icons (publicly visible, no external CDNs)

## Project Structure
```
weather-app/
├─ app.py
├─ templates/
│  └─ index.html
├─ static/
│  ├─ css/styles.css
│  └─ icons/{sun.svg, cloud.svg, rain.svg}
└─ data/
   └─ cities.json
```
(Optional) If you also place `lstm_weather_model.h5` and `scaler.save` next to `app.py`, the `/predict` endpoint will use your trained model.

## How to Run
1) (Recommended) Create a virtual environment and install minimal deps:
```bash
pip install flask numpy joblib keras tensorflow scikit-learn
```
> If you don't plan to use the LSTM model, you only need:
```bash
pip install flask numpy joblib
```

2) Start the server:
```bash
python app.py
```
The app runs at: http://127.0.0.1:5000/

3) Use the UI
- Pick a city ➜ see temperature tiles
- Click **Refresh** to simulate an update
- Enter 10 comma‑separated temps and click **Predict**

## Using Your Existing Model
- Train/save with your scripts to produce:
  - `lstm_weather_model.h5`
  - `scaler.save`
- Place both files in the project root (same folder as `app.py`), then restart the server.

## Notes
- Current/5‑day temps are generated locally for demo purposes so the app works without any API keys or internet access.
- You can later wire `/api/weather/<city_id>` to a real weather API.
