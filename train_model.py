import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

# Load data
df = pd.read_csv('cars.csv')

print(f"Training with {len(df)} records")
print(f"Brands: {df['brand'].nunique()}")
print(f"Models: {df['model'].nunique()}")

# Encode categorical variables
le_brand = LabelEncoder()
le_model = LabelEncoder()
le_gear = LabelEncoder()

df['brand_enc'] = le_brand.fit_transform(df['brand'])
df['model_enc'] = le_model.fit_transform(df['model'])
df['gear_enc'] = le_gear.fit_transform(df['gear'])

# Features
X = df[['year', 'km', 'brand_enc', 'model_enc', 'engine_size', 'gear_enc', 'owners']]
y = df['price']

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
model.fit(X, y)

# Save everything
joblib.dump({
    'model': model,
    'le_brand': le_brand,
    'le_model': le_model,
    'le_gear': le_gear
}, 'car_price_model.joblib')

print("Model trained and saved!")
print(f"Feature importances:")
for name, imp in zip(['year', 'km', 'brand', 'model', 'engine_size', 'gear', 'owners'], model.feature_importances_):
    print(f"  {name}: {imp:.3f}")
