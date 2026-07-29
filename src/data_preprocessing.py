import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

FEATURE_PATH = "features_data.csv"
df = pd.read_csv(FEATURE_PATH)

TARGET_COLUMN = "priceusd"

NUMERIC_FEATURES = [
    "year",
    "mileage_kilometers",
    "volume_cm3",
    "car_age",
    "mileage_per_year",
    "engine_volume_liters"
]

CATEGORICAL_FEATURES = [
    "make",
    "model",
    "condition",
    "fuel_type",
    "transmission",
    "drive_unit",
    "segment"
]

def get_all_feature_columns() -> list[str]:
    return NUMERIC_FEATURES + CATEGORICAL_FEATURES

def split_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
 
    X = df[get_all_feature_columns()].copy()
    Y = df[TARGET_COLUMN].copy()
 
    return X, Y

X, Y = split_features_and_target(df)

def build_numeric_transformer() -> Pipeline:
 
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(missing_values=pd.NA, strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
 
    return numeric_transformer
    
def build_categorical_transformer() -> Pipeline:
 
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(missing_values=pd.NA, strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
 
    return categorical_transformer

def build_preprocessor() -> ColumnTransformer:
 
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", build_numeric_transformer(), NUMERIC_FEATURES),
            ("cat", build_categorical_transformer(), CATEGORICAL_FEATURES)
        ],
        remainder="drop"
    )
 
    return preprocessor
    