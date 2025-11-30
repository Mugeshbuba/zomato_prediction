import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import os

# -------- 1. Load data --------
# Change this path to your CSV location
csv_path = r"C:\Users\denesh babu\OneDrive - Kumaraguru College of Technology\Desktop\Restaurant-Rating-Prediction\data\zomato.csv"
df = pd.read_csv(csv_path)

# -------- 2. Basic cleaning --------
# Keep only needed columns
cols = [
    'location',
    'cuisines',
    'rest_type',
    'approx_cost(for two people)',
    'votes',
    'online_order',
    'book_table',
    'rate'
]
df = df[cols].copy()

# Clean 'rate' column like "3.8/5" -> 3.8
df['rate'] = df['rate'].astype(str).str.replace('/5', '', regex=False)
df['rate'] = df['rate'].replace('nan', np.nan)
df['rate'] = pd.to_numeric(df['rate'], errors='coerce')

# Drop rows with missing target or key features
df.dropna(subset=['rate', 'location', 'cuisines', 'rest_type',
                  'approx_cost(for two people)', 'votes',
                  'online_order', 'book_table'], inplace=True)

# Clean approx_cost (remove commas)
df['approx_cost(for two people)'] = (
    df['approx_cost(for two people)']
    .astype(str).str.replace(',', '', regex=False)
)
df['approx_cost(for two people)'] = pd.to_numeric(
    df['approx_cost(for two people)'], errors='coerce'
)

# Convert online_order, book_table to 0/1
df['online_order'] = df['online_order'].map({'Yes': 1, 'No': 0})
df['book_table'] = df['book_table'].map({'Yes': 1, 'No': 0})

# Drop any remaining NaN
df.dropna(inplace=True)

# -------- 3. Encode categorical columns --------
le_location = LabelEncoder()
le_cuisines = LabelEncoder()
le_rest_type = LabelEncoder()

df['location_enc'] = le_location.fit_transform(df['location'])
df['cuisines_enc'] = le_cuisines.fit_transform(df['cuisines'])
df['rest_type_enc'] = le_rest_type.fit_transform(df['rest_type'])

# Features and target
X = df[[
    'location_enc',
    'cuisines_enc',
    'rest_type_enc',
    'approx_cost(for two people)',
    'votes',
    'online_order',
    'book_table'
]]
y = df['rate']

# -------- 4. Train / test split --------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------- 5. Train model (Random Forest) --------
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# -------- 6. Evaluate model --------
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("R2:", r2)
print("MAE:", mae)
print("RMSE:", rmse)

# -------- 7. Save model and encoders --------
base_dir = os.path.dirname(os.path.abspath(__file__))

joblib.dump(model, os.path.join(base_dir, 'model.pkl'))
joblib.dump(le_location, os.path.join(base_dir, 'le_location.pkl'))
joblib.dump(le_cuisines, os.path.join(base_dir, 'le_cuisines.pkl'))
joblib.dump(le_rest_type, os.path.join(base_dir, 'le_rest_type.pkl'))

# -------- 8. Save average rating per location for heatmap --------
location_avg = (
    df.groupby('location')['rate']
      .mean()
      .reset_index()
      .rename(columns={'rate': 'avg_rating'})
)
heatmap_path = os.path.join(base_dir, 'location_avg.csv')
location_avg.to_csv(heatmap_path, index=False)
print("Saved location_avg.csv to:", heatmap_path)

print("Saved model and encoders to:", base_dir)
