from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("../models/random_forest_model.pkl")

# Load training columns used during model training
X_train = pd.read_csv("../data/processed/X_train.csv")
training_columns = X_train.columns.tolist()

# Load source dataset for dropdown values
df_source = pd.read_csv("../data/processed/mta_subway_understood.csv")

# Station dropdown data
station_df = (
    df_source[["station_complex", "station_complex_id", "borough"]]
    .dropna()
    .drop_duplicates()
    .sort_values("station_complex")
)

station_names = station_df["station_complex"].tolist()

# Other dropdown values
payment_methods = sorted(df_source["payment_method"].dropna().unique().tolist())
fare_categories = sorted(df_source["fare_class_category"].dropna().unique().tolist())
boroughs = sorted(df_source["borough"].dropna().unique().tolist())

# Friendly mapping dictionaries
hour_map = {
    "12:00 AM": 0, "1:00 AM": 1, "2:00 AM": 2, "3:00 AM": 3,
    "4:00 AM": 4, "5:00 AM": 5, "6:00 AM": 6, "7:00 AM": 7,
    "8:00 AM": 8, "9:00 AM": 9, "10:00 AM": 10, "11:00 AM": 11,
    "12:00 PM": 12, "1:00 PM": 13, "2:00 PM": 14, "3:00 PM": 15,
    "4:00 PM": 16, "5:00 PM": 17, "6:00 PM": 18, "7:00 PM": 19,
    "8:00 PM": 20, "9:00 PM": 21, "10:00 PM": 22, "11:00 PM": 23
}

day_map = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6
}

month_map = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12
}


@app.route("/")
def home():
    return render_template(
        "index.html",
        station_names=station_names,
        payment_methods=payment_methods,
        fare_categories=fare_categories,
        boroughs=boroughs,
        prediction_text=None,
        demand_level=None,
        selected_station=None
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get station name selected by user
        station_complex = request.form["station_complex"]

        # Find station row
        station_row = station_df[station_df["station_complex"] == station_complex].iloc[0]
        station_complex_id = int(station_row["station_complex_id"])

        # Get other form values
        borough = request.form["borough"]
        payment_method = request.form["payment_method"]
        fare_class_category = request.form["fare_class_category"]

        # Friendly user inputs
        time_str = request.form["hour"]
        day_name = request.form["day_of_week"]
        month_name = request.form["month"]

        # Convert friendly values into numeric model values
        hour = hour_map[time_str]
        day_of_week = day_map[day_name]
        month = month_map[month_name]

        # Automatically determine weekend from selected day
        is_weekend = 1 if day_of_week in [5, 6] else 0

        # Fixed day of month for cleaner demo
        day = 15

        # Hidden/default values kept for model compatibility
        transfers = 0.0
        latitude = 0.0
        longitude = 0.0

        # Create input dictionary
        input_data = {
            "station_complex_id": station_complex_id,
            "transfers": transfers,
            "latitude": latitude,
            "longitude": longitude,
            "hour": hour,
            "day": day,
            "month": month,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "borough": borough,
            "payment_method": payment_method,
            "fare_class_category": fare_class_category
        }

        # Convert into DataFrame
        input_df = pd.DataFrame([input_data])

        # One-hot encode categorical values
        input_df = pd.get_dummies(input_df)

        # Match the exact training columns
        input_df = input_df.reindex(columns=training_columns, fill_value=0)

        # Make prediction
        prediction = model.predict(input_df)[0]

        # Prevent negative passenger count
        prediction = max(0, prediction)

        # Create demand level for display
        if prediction < 100:
            demand_level = "Low"
        elif prediction < 300:
            demand_level = "Medium"
        else:
            demand_level = "High"

        return render_template(
            "index.html",
            station_names=station_names,
            payment_methods=payment_methods,
            fare_categories=fare_categories,
            boroughs=boroughs,
            prediction_text=f"Predicted Passenger Count: {int(round(prediction))}",
            demand_level=f"Demand Level: {demand_level}",
            selected_station=station_complex
        )

    except Exception as e:
        return render_template(
            "index.html",
            station_names=station_names,
            payment_methods=payment_methods,
            fare_categories=fare_categories,
            boroughs=boroughs,
            prediction_text=f"Error: {str(e)}",
            demand_level=None,
            selected_station=None
        )


if __name__ == "__main__":
    app.run(debug=True)