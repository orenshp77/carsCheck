import joblib
import pandas as pd

# טען את המודל
model = joblib.load("car_price_model.joblib")

# רכב לדוגמה (חייבים להיות אותם שדות כמו ב-CSV)
car = {
    "year": 2019,
    "km": 60000,
    "brand": "Mazda",
    "model": "3",
    "engine_size": 2000,
    "gear": "Automatic",
    "owners": 2
}

df = pd.DataFrame([car])

price = model.predict(df)[0]
print(f"Predicted price: {int(price):,} ₪")
