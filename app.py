import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------
# Load dataset
# ---------------------
df = pd.read_csv("final_dataset.csv")

# Features and target
features = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'Ozone']
target = 'AQI'

X = df[features]
y = df[target]

# ---------------------
# Split dataset
# ---------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------
# Train Regression Models
# ---------------------
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

# ---------------------
# Logistic Regression Category Model
# ---------------------
def categorize_aqi(aqi):
    if aqi <= 50:
        return 0  # Good
    elif aqi <= 100:
        return 1  # Moderate
    else:
        return 2  # Bad

y_class = df[target].apply(categorize_aqi)

X_train_cl, X_test_cl, y_train_cl, y_test_cl = train_test_split(
    X, y_class, test_size=0.2, random_state=42
)

X_train_cl_scaled = scaler.fit_transform(X_train_cl)
X_test_cl_scaled = scaler.transform(X_test_cl)

log_reg = LogisticRegression(max_iter=500)
log_reg.fit(X_train_cl_scaled, y_train_cl)

# ---------------------
# Streamlit UI
# ---------------------
st.title("Air Quality Prediction App")

# Prediction type selection
prediction_type = st.radio(
    "Choose Prediction Type",
    ("Exact AQI Value", "Air Quality Category")
)

# ---------------------
# Optional Inputs (NOT used for prediction)
# ---------------------
st.sidebar.header("Optional Information")
date_input = st.sidebar.date_input("Date (Optional)")
year_input = st.sidebar.number_input(
    "Year (Optional)", min_value=1900, max_value=2100, value=2025
)
holiday_input = st.sidebar.checkbox("Is it a Holiday? (Optional)", value=False)

optional_info = {
    "Date": str(date_input),
    "Year": year_input,
    "Holiday": "Yes" if holiday_input else "No"
}

# ---------------------
# Pollutant Inputs
# ---------------------
st.sidebar.header("Input Air Pollutant Levels")

pm25 = st.sidebar.number_input("PM2.5", 0.0, 500.0, 50.0)
pm10 = st.sidebar.number_input("PM10", 0.0, 500.0, 50.0)
no2 = st.sidebar.number_input("NO2", 0.0, 200.0, 20.0)
so2 = st.sidebar.number_input("SO2", 0.0, 200.0, 10.0)
co = st.sidebar.number_input("CO", 0.0, 50.0, 1.0)
ozone = st.sidebar.number_input("Ozone", 0.0, 300.0, 50.0)

input_data = pd.DataFrame(
    [[pm25, pm10, no2, so2, co, ozone]],
    columns=features
)


# ---------------------
# Predictions
# ---------------------
if prediction_type == "Exact AQI Value":
    lin_pred = lin_reg.predict(input_data)[0]
    lasso_pred = lasso.predict(input_data)[0]

    st.subheader("Predicted AQI Values")
    st.write(f"**Linear Regression Prediction:** {lin_pred:.2f}")

else:
    log_pred_class = log_reg.predict(
        scaler.transform(input_data)
    )[0]

    aqi_class_dict = {
        0: "Good",
        1: "Moderate",
        2: "Bad"
    }

    st.subheader("Predicted Air Quality Category")
    st.write(f"**Air Quality:** {aqi_class_dict[log_pred_class]}")
    
    # ---------------------
# Save Predictions to CSV
# ---------------------

save_data = st.button("Save Prediction")

if save_data:
    # Create a record dictionary
    record = {
        "PM2.5": pm25,
        "PM10": pm10,
        "NO2": no2,
        "SO2": so2,
        "CO": co,
        "Ozone": ozone,
        "Date": optional_info["Date"],
        "Year": optional_info["Year"],
        "Holiday": optional_info["Holiday"],
    }

    # Add prediction output
    if prediction_type == "Exact AQI Value":
        record["Prediction_Type"] = "Exact AQI"
        record["Linear_Regression_Pred"] = lin_pred
    else:
        record["Prediction_Type"] = "Category"
        record["Category_Pred"] = aqi_class_dict[log_pred_class]

    # Convert to DataFrame
    record_df = pd.DataFrame([record])

    # Save to CSV (append mode)
    try:
        record_df.to_csv("prediction_history.csv", mode='a', header=not pd.io.common.file_exists("prediction_history.csv"), index=False)
        st.success("✅ Prediction saved to prediction_history.csv")
    except Exception as e:
        st.error(f"Error saving prediction: {e}")

