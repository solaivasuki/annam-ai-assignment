import pandas as pd
import re

def load_kcc_data(file_path):
    df = pd.read_csv(file_path,low_memory=False)
    print(f"Loaded {len(df)} rows")
    return df

def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    return text.strip()

def preprocess_data(df):
    df['question'] = df['question'].apply(clean_text)
    df['answer'] = df['answer'].apply(clean_text)
    df = df.dropna(subset=['question', 'answer'])
    df = df.reset_index(drop=True)
    return df[['question', 'answer']]

if __name__ == "__main__":
    kcc_raw_path = "kcc_dataset.csv"  # your KCC CSV path
    df = load_kcc_data(kcc_raw_path)
    df_clean = preprocess_data(df)
    df_clean.to_csv("kcc_preprocessed.csv", index=False)
    print("Preprocessing complete, saved to kcc_preprocessed.csv")