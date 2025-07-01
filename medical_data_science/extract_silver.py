# filename: extract_silver.py

import pandas as pd


bronze_path = "burn_parameters.csv"
df = pd.read_csv(bronze_path)


df_clean = df.dropna(subset=["value", "charttime", "parameter_name"])


df_clean["charttime"] = pd.to_datetime(df_clean["charttime"], errors="coerce")


df_clean = df_clean.dropna(subset=["charttime"])


columns = ["subject_id", "hadm_id", "stay_id", "charttime", "parameter_name", "value", "unit"]
df_clean = df_clean[columns].sort_values(by=["subject_id", "charttime"])


silver_path = "burn_parameters_silver.csv"
df_clean.to_csv(silver_path, index=False)

print(f"✅ Silber-Tabelle gespeichert: {silver_path}")
