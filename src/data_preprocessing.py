import pandas as pd

def load_data():
    df = pd.read_csv('../data/mental_health_risk_dataset.csv')
    return df

def preprocess_data(df):
    df_encoded = pd.get_dummies(
        df,
        columns=['gender','marital_status','education_level','employment_status']
    )
    return df_encoded