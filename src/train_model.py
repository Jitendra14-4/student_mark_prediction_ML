
import os
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load dataset
df = pd.read_csv("data/student-mat.csv")

# 2. Select features and target
features = ["studytime", "absences", "G1", "G2"]
target = "G3"

X = df[features]
y = df[target]

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Create model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

# 5. Train model
model.fit(X_train, y_train)

# 6. Evaluate model
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Model Training Completed!")
print("MAE:", round(mae, 2))
print("R2 Score:", round(r2, 2))

# 7. Save model
os.makedirs("models", exist_ok=True)

joblib.dump(
    {"model": model, "features": features},
    "models/student_marks_model.pkl"
)

print("Model saved successfully!")
