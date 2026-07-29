import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from data_preprocessing import build_preprocessor, split_features_and_target

DATA_PATH = "data/features_data.csv"
MODEL_PATH = "models/car_price_model.joblib"

lista_modele = [LinearRegression(), DecisionTreeRegressor(), RandomForestRegressor(), GradientBoostingRegressor()]

df = pd.read_csv(DATA_PATH)

X, Y = split_features_and_target(df)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print("Creare Pipeline")
for i in lista_modele:
    model_name = i.__class__.__name__
    
    model = Pipeline(
        steps=[
        ("processors", build_preprocessor()),
        ("regressors", i)
        ]
    )
    
    print("Training Model")
    model.fit(X_train, Y_train)
    print("Saving Model")
    joblib.dump(model, f"models/car_price_{model_name}.joblib")
    print("Model Saved")

