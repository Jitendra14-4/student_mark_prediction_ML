
from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model
saved_data = joblib.load("models/student_marks_model.pkl")
model = saved_data["model"]
features = saved_data["features"]

# Detect weak areas
def detect_weak_areas(row):
    weak_areas = []

    if row["studytime"] <= 1:
        weak_areas.append("Low study time")

    if row["absences"] > 5:
        weak_areas.append("High absences")

    if row["G1"] < 10:
        weak_areas.append("Low first-period grade")

    if row["G2"] < 10:
        weak_areas.append("Low second-period grade")

    if not weak_areas:
        weak_areas.append("No major risk detected")

    return ", ".join(weak_areas)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return "Please upload a file."

    file = request.files["file"]

    if file.filename == "":
        return "No file selected."

    try:
        filename = file.filename.lower()

        # Read CSV or Excel
        if filename.endswith(".csv"):
            df = pd.read_csv(file)

        elif filename.endswith(".xlsx"):
            df = pd.read_excel(file)

        else:
            return "Upload only CSV or XLSX files."

        # Validate columns
        missing = [col for col in features if col not in df.columns]

        if missing:
            return f"Missing columns: {', '.join(missing)}"

        # Convert input values
        for col in features:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        if df[features].isnull().any().any():
            return "Missing or invalid student data."

        # Predict final marks (0-20)
        predictions = model.predict(df[features])

        df["Predicted_Marks"] = predictions.clip(0, 20).round(2)

        # Weak area detection
        df["Weak_Areas"] = df.apply(
            detect_weak_areas, axis=1
        )

        # Performance
        df["Performance"] = df["Predicted_Marks"].apply(
            lambda marks:
                "Excellent" if marks >= 16
                else "Good" if marks >= 12
                else "Needs Improvement"
        )

        results = df.to_dict(orient="records")

        return render_template(
            "result.html",
            results=results
        )

    except Exception as e:
        return f"Error processing file: {str(e)}"


if __name__ == "__main__":
    app.run(debug=True)
