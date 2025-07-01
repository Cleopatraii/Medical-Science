import pandas as pd

# 1. Load the ML-ready dataset and the original gold table for admission times
ml_df = pd.read_csv("gold_table_ml.csv")
gold = pd.read_csv("gold_table.csv", parse_dates=["window_start"])

# 2. Compute ICU admission time per stay (earliest window_start)
admission_times = (
    gold
    .groupby("stay_id")["window_start"]
    .min()
    .reset_index()
    .rename(columns={"window_start": "admission_time"})
)

# 3. Merge admission_time back into the ML dataframe
ml_df = ml_df.merge(admission_times, on="stay_id", how="left")

# 4. Sort stays by admission_time and determine the 80/20 split
unique_stays = admission_times.sort_values("admission_time")["stay_id"].tolist()
n_stays = len(unique_stays)
n_train = int(n_stays * 0.8)

train_stays = unique_stays[:n_train]
test_stays  = unique_stays[n_train:]

# 5. Split into train and test DataFrames
train_df = ml_df[ml_df["stay_id"].isin(train_stays)].copy()
test_df  = ml_df[ml_df["stay_id"].isin(test_stays)].copy()

# 6. Save to separate CSV files
train_df.to_csv("train_ml.csv", index=False)
test_df.to_csv("test_ml.csv",  index=False)

print(f"Saved train_ml.csv: {len(train_df)} rows ({len(train_stays)} stays)")
print(f"Saved test_ml.csv:  {len(test_df)} rows ({len(test_stays)} stays)")
