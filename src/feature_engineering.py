import pandas as pd
import numpy as np

CLEANED_DATA_PATH = "data/cleaned_cars.csv"
FEATURES_DATA_PATH = "data/features_data.csv"



def add_car_age (df: pd.DataFrame, max_year: int = 2019) -> pd.DataFrame:
    
    df = df.copy()
    df["car_age"] = max_year - df["year"]
    
    return df

def add_mileage_per_year (df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.copy()
    df["mileage_per_year"] = (df["mileage_kilometers"] / df["car_age"]).round(2)
    df["mileage_per_year"] = (df["mileage_per_year"].replace([np.inf, -np.inf], 0))
    
    return df

def add_engine_volume_liters (df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.copy()
    df["engine_volume_liters"] = (df["volume_cm3"] / 1000).round(2)
    
    return df

def build_features (df: pd.DataFrame) -> pd.DataFrame:
    
    df_features = (
        df
        .pipe(add_car_age)
        .pipe(add_mileage_per_year)
        .pipe(add_engine_volume_liters)
        .reset_index(drop=True)
        )

    return df_features

def main() -> None:
    """Load cleaned data, build features, and save the feature-engineered dataset."""
    print("Loading cleaned dataset...")
 
    df_cleaned = pd.read_csv(CLEANED_DATA_PATH)
 
    print("Building features...")
 
    df_features = build_features(df_cleaned)
 
    print("Saving feature-engineered dataset...")
 
    df_features.to_csv(FEATURES_DATA_PATH, index=False)
 
    print(f"Feature-engineered dataset saved to: {FEATURES_DATA_PATH}")
    
if __name__ == "__main__":
    main()



