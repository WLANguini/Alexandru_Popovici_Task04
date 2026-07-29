import pandas as pd
import joblib
from data_preprocessing import split_features_and_target
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = "data/features_data.csv"
MODEL_PATH = "models/car_price_LinearRegression.joblib"

df = pd.read_csv(DATA_PATH)

X, Y = split_features_and_target(df)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = joblib.load(MODEL_PATH)

Y_pred = model.predict(X_test)

mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
rmse = mse ** 0.5
r2 = r2_score(Y_test, Y_pred)

print(Y_pred[:10])
print(f"Mean absolute error: {mae}\n Mean squared error: {mse}\n RMSE: {rmse}\n R2 Score: {r2}")

metrics = pd.DataFrame({
    "metric": ["MAE", "MSE", "RMSE", "R2"],
    "value": [mae, mse, rmse, r2],
})
 
print("\nRegression metrics:")
print(metrics.round(2))

print("\nCreating prediction analysis table...")
 
prediction_analysis = pd.DataFrame({
    "Actual Price": Y_test.values,
    "Predicted Price": Y_pred,
})

prediction_analysis["Error Price"] = (
    prediction_analysis["Actual Price"]
    - prediction_analysis["Predicted Price"]
)
prediction_analysis["Absolute error price"] = (
    prediction_analysis["Error Price"].abs()
)
print(
    prediction_analysis
    .sample(10, random_state=42)
)

print("\nLargest prediction errors:")
 
print(
    prediction_analysis
    .sort_values("Absolute error price", ascending=False)
    .head(10)
)