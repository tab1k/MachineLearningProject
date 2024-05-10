from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd


def preprocess_data(df):
    scaler = StandardScaler()
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
    df[numeric_features] = scaler.fit_transform(df[numeric_features])

    categorical_features = df.select_dtypes(include=['object']).columns
    for feature in categorical_features:
        le = LabelEncoder()
        df[feature] = le.fit_transform(df[feature])

    selected_columns = ["Indicate your gender", "Who are you", "Please indicate your age"]

    # Проверяем, есть ли столбец "Which form of domestic violence is most prevalent?"
    if "Which form of domestic violence is most prevalent?" in df.columns:
        selected_columns.append("Which form of domestic violence is most prevalent?")

    df = df[selected_columns]

    return df


