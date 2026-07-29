import pandas as pd
import re

RAW_PATH = "data/cars.csv"
CLEANED_DATA_PATH = "data/cleaned_cars.csv"

def drop_unwanted_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop(columns=["color"], errors="ignore")
    return df

def filter_mileage_outliers(df: pd.DataFrame, max_year: int = 2019) -> pd.DataFrame:
    df = df.copy()
    df["car_age"] = max_year - df["year"]
    df["km_per_year"] = df["mileage_kilometers"] / df["car_age"].clip(lower=1)
    df = df[(df["km_per_year"] >= 100) & (df["km_per_year"] <= 100000)]
    df = df.drop(columns=["car_age", "km_per_year"])
    return df

def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
 
    df = df.copy()
 
    new_columns = []
 
    for col in df.columns:
        clean_col = col.strip().lower()
 
        clean_col = clean_col.replace("(", "_")
        clean_col = clean_col.replace(")", "")
        clean_col = clean_col.replace("-", "_")
        clean_col = clean_col.replace("/", "_")
 
        clean_col = re.sub(r"\s+", "_", clean_col)
        clean_col = re.sub(r"[^a-z0-9_]", "", clean_col)
        clean_col = re.sub(r"_+", "_", clean_col)
        clean_col = clean_col.strip("_")
 
        new_columns.append(clean_col)
 
    df.columns = new_columns

 
    return df



def strip_string_values(df: pd.DataFrame) -> pd.DataFrame:
 
    df = df.copy()
 
    text_columns = df.select_dtypes(include=["object"]).columns
 
    for col in text_columns:
        df[col] = df[col].str.strip()
 
    return df



MISSING_LIKE_VALUES = {
    "",
    " ",
    "nan",
    "NaN",
    "NAN",
    "null",
    "Null",
    "NULL",
    "none",
    "None",
    "NONE",
}
def replace_missing_like_values(df: pd.DataFrame) -> pd.DataFrame:
 
    df = df.copy()
 
    df = df.replace(list(MISSING_LIKE_VALUES), pd.NA)
 
    return df



def clean_categorical_values(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.copy()

    categorical_columns = [
        "make",
        "model",
        "condition",
        "fuel_type",
        "transmission",
        "drive_unit",
        "segment"
    ]
    
    for col in categorical_columns:
        df[col] = (df[col].astype("string").str.strip().str.lower())
    
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
 
    df_clean = (
        df
        .drop_duplicates()
        .pipe(standardize_column_names)
        .pipe(strip_string_values)
        .pipe(replace_missing_like_values)
        .pipe(clean_categorical_values)
        .pipe(drop_unwanted_columns)
        .pipe(filter_mileage_outliers)
        .reset_index(drop=True)
        .dropna(subset="")
    )
 
    return df_clean 

def main() -> None:
    """Load raw data, clean it, and save the cleaned dataset."""
    print("Loading raw dataset...")
 
    df_raw = pd.read_csv(RAW_PATH)
 
    print("Cleaning dataset...")
 
    df_cleaned = clean(df_raw)
 
    print("Saving cleaned dataset...")
 
    df_cleaned.to_csv(CLEANED_DATA_PATH, index=False)
 
    print(f"Cleaned dataset saved to: {CLEANED_DATA_PATH}")
    print(df_cleaned.isna().sum())


if __name__ == "__main__":
    main()