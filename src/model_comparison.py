import pandas as pd
import os
from data_preprocessing import split_features_and_target
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

os.makedirs("model", exist_ok=True)

DATA_PATH = "data/features_data.csv"
MODEL_PATH = "models/"
CHOSEN_MODEL_PATH = "model/car_price_model.joblib"

df = pd.read_csv(DATA_PATH)

X, Y = split_features_and_target(df)
X_train, X_test, Y_train, Y_test =train_test_split(X, Y, test_size = 0.2, random_state=42)

rezultate = []

for file in os.listdir(MODEL_PATH):
    CURRENT_MODEL_PATH = f"models/{file}"
    
    model = joblib.load(CURRENT_MODEL_PATH)
    
    Y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(Y_test, Y_pred)
    mse = mean_squared_error(Y_test, Y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(Y_test, Y_pred)
    
    rezultate.append([
        file,
        mae,
        mse,
        rmse,
        r2
    ])

    df_results = pd.DataFrame(
        rezultate,
        columns=[
            "Model",
            "MAE",
            "MSE",
            "RMSE",
            "R2"
        ]
    )
    
    print(f"Modelul {file}, a obtinut urmatoarele scoruri:\n Mean absolute error: {mae}\n Mean squared error: {mse}\n RMSE: {rmse}\n R2 Score: {r2}")


df_results["Score"] = (
    df_results["R2"] 
    - df_results["RMSE"] / df_results["RMSE"].max()
)

df_results["Placement"] = (
    df_results["Score"]
    .rank(method="dense", ascending=False)
    .astype(int)
)

# Deoarece git push nu functioneaza cu modelele din cauza faptului ca acestea au peste 300MB am hotarat sa fac un gitignore pentru a putea continua

best_model_name = df_results.loc[df_results["Placement"] == 1, "Model"].iloc[0]

best_model_path = f"{MODEL_PATH}/{best_model_name}"

best_model = joblib.load(best_model_path)

joblib.dump(
    best_model,
    f"{CHOSEN_MODEL_PATH}"
)


print(
    f"\nCel mai bun model este: {best_model_name}"
)
print(
    "Model salvat în best_model/best_car_price_model.joblib"
)


