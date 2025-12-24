from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    return send_from_directory(os.path.dirname(__file__), "index.html")

MODEL_PATH = "car_price_model.joblib"
CSV_PATH = "cars.csv"

# Load model and encoders
model_data = joblib.load(MODEL_PATH)
model = model_data['model']
le_brand = model_data['le_brand']
le_model = model_data['le_model']
le_gear = model_data['le_gear']

cars_df = pd.read_csv(CSV_PATH)

REQUIRED_FIELDS = ["year", "km", "brand", "model", "engine_size", "gear", "owners"]

@app.get("/health")
def health():
    return {"status": "ok"}

def normalize_payload(data: dict):
    missing = [f for f in REQUIRED_FIELDS if f not in data]
    if missing:
        return None, {"error": "Missing fields", "missing": missing}

    try:
        row = {
            "year": int(data["year"]),
            "km": int(data["km"]),
            "brand": str(data["brand"]),
            "model": str(data["model"]),
            "engine_size": int(data["engine_size"]),
            "gear": str(data["gear"]),
            "owners": int(data["owners"]),
        }
        return row, None
    except Exception as e:
        return None, {"error": "Bad field types", "details": str(e)}

@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    row, err = normalize_payload(data)
    if err:
        return jsonify(err), 400

    try:
        # Encode categorical variables
        brand_enc = le_brand.transform([row["brand"]])[0] if row["brand"] in le_brand.classes_ else 0
        model_enc = le_model.transform([row["model"]])[0] if row["model"] in le_model.classes_ else 0
        gear_enc = le_gear.transform([row["gear"]])[0] if row["gear"] in le_gear.classes_ else 0

        # Create feature array
        features = [[
            row["year"],
            row["km"],
            brand_enc,
            model_enc,
            row["engine_size"],
            gear_enc,
            row["owners"]
        ]]

        pred = float(model.predict(features)[0])
        return jsonify({"predicted_price": round(pred)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.post("/similar")
def similar():
    data = request.get_json(silent=True) or {}
    row, err = normalize_payload(data)
    if err:
        return jsonify(err), 400

    df = cars_df.copy()

    # Category matching
    same_brand = (df["brand"].astype(str) == row["brand"]).astype(int)
    same_model = (df["model"].astype(str) == row["model"]).astype(int)
    same_gear  = (df["gear"].astype(str) == row["gear"]).astype(int)

    # Numeric distance
    def safe_num(col):
        return pd.to_numeric(df[col], errors="coerce").fillna(df[col].median())

    year = safe_num("year")
    km = safe_num("km")
    engine = safe_num("engine_size")
    owners = safe_num("owners")

    dy = (year - row["year"]).abs() / 10.0
    dkm = (km - row["km"]).abs() / 100000.0
    deng = (engine - row["engine_size"]).abs() / 2000.0
    down = (owners - row["owners"]).abs() / 6.0

    score = (dy + dkm + deng + down) - (same_brand * 0.6) - (same_model * 0.9) - (same_gear * 0.3)
    df["_score"] = score

    out = df.sort_values("_score").head(5).drop(columns=["_score"])
    cols = [c for c in ["year","km","brand","model","engine_size","gear","owners","price"] if c in out.columns]
    return jsonify({"items": out[cols].to_dict(orient="records")})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
