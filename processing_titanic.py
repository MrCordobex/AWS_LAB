import os

import pandas as pd
from sklearn.model_selection import train_test_split


BASE_DIR = "/opt/ml/processing"
INPUT_PATH = os.path.join(BASE_DIR, "input", "titanic-dataset.csv")

TRAIN_PATH = os.path.join(BASE_DIR, "train", "train.csv")
VALIDATION_PATH = os.path.join(BASE_DIR, "validation", "validation.csv")
TEST_PATH = os.path.join(BASE_DIR, "test", "test.csv")


def clean_and_engineer_features(df):
    if "Cabin" in df.columns:
        df = df.drop(columns=["Cabin"])

    df["Embarked"] = df["Embarked"].fillna("S")
    df["Fare"] = df["Fare"].fillna(df["Fare"].mean())

    df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)

    rare_titles = [
        "Lady",
        "Countess",
        "Capt",
        "Col",
        "Don",
        "Dr",
        "Major",
        "Rev",
        "Sir",
        "Jonkheer",
        "Dona",
    ]

    df["Title"] = df["Title"].replace(rare_titles, "Rare")
    df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})

    df["Age"] = df.groupby(["Sex", "Pclass"])["Age"].transform(
        lambda values: values.fillna(values.median())
    )
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Age"] = df["Age"].astype("int64")

    df["Family_size"] = df["SibSp"] + df["Parch"] + 1

    def family_category(size):
        if size == 1:
            return "Alone"
        if size <= 4:
            return "Small"
        return "Large"

    df["Family_size"] = df["Family_size"].apply(family_category)

    columns_to_drop = ["PassengerId", "Name", "Ticket", "SibSp", "Parch"]
    df = df.drop(columns=columns_to_drop)

    categorical_columns = ["Sex", "Embarked", "Title", "Family_size"]
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=False, dtype=int)

    target = df.pop("Survived")
    df.insert(0, "Survived", target)

    return df


if __name__ == "__main__":
    print("Reading raw data from:", INPUT_PATH)
    df = pd.read_csv(INPUT_PATH)

    print("Original shape:", df.shape)

    df = clean_and_engineer_features(df)

    print("Processed shape:", df.shape)
    print("Processed columns:", df.columns.tolist())

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=42,
        stratify=df["Survived"],
    )

    validation_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df["Survived"],
    )

    os.makedirs(os.path.dirname(TRAIN_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(VALIDATION_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(TEST_PATH), exist_ok=True)

    train_df.to_csv(TRAIN_PATH, header=False, index=False)
    validation_df.to_csv(VALIDATION_PATH, header=False, index=False)
    test_df.to_csv(TEST_PATH, header=False, index=False)

    print("Train shape:", train_df.shape)
    print("Validation shape:", validation_df.shape)
    print("Test shape:", test_df.shape)
    print("Processing completed successfully")
