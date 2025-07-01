import pandas as pd
import numpy as np

# Load gold-level data and sort by patient and time
gold = pd.read_csv("gold_table.csv")
gold['window_start'] = pd.to_datetime(gold['window_start'])
gold.sort_values(['stay_id','window_start'], inplace=True)

# Define Shock Index and create 12h-ahead target
gold['shock_index_orig'] = gold['Heart Rate'] / gold['Systolic BP']
gold['shock_index_future'] = gold.groupby('stay_id')['shock_index_orig'].shift(-12)
gold['shock_index_critical_12h'] = (gold['shock_index_future'] > 0.9).astype(int)
# Drop rows without a future value (cannot determine label)
gold_ml = gold[~gold['shock_index_future'].isna()].copy()

# Forward-fill missing feature values within each stay
features = ['GCS','Heart Rate','Systolic BP','Body Temp','WBC','CRP','Albumin']
gold_ml[features] = gold_ml.groupby('stay_id')[features].ffill()
# Impute any remaining NaNs with typical normal values (since some labs were never measured)
defaults = {'Body Temp': 37.0, 'WBC': 7.0, 'CRP': 0.0, 'Albumin': 4.0}
for col in features:
    gold_ml[col].fillna(defaults.get(col, gold_ml[col].median()), inplace=True)

# Recompute Shock Index with filled values (for use as current feature)
gold_ml['ShockIndex_current'] = gold_ml['Heart Rate'] / gold_ml['Systolic BP']

# Calculate hours since ICU admission for each row
gold_ml['hours_since_admission'] = gold_ml.groupby('stay_id').cumcount()

# Engineer trend feature: change in Shock Index over last 4 hours
gold_ml['ShockIndex_trend_4h'] = gold_ml['ShockIndex_current'] - gold_ml.groupby('stay_id')['ShockIndex_current'].shift(4)
gold_ml['ShockIndex_trend_4h'].fillna(0, inplace=True)  # no trend for first 4 hours

# Engineer time since last abnormal WBC (>12 or <4 K/µL)
gold_ml['time_since_abnormal_WBC'] = 0
for stay_id, group in gold_ml.groupby('stay_id'):
    last_abnormal = None
    for idx, row in group.iterrows():
        wbc = row['WBC']
        hrs = row['hours_since_admission']
        if not np.isnan(wbc):
            if wbc > 12 or wbc < 4:
                last_abnormal = hrs
                gold_ml.at[idx, 'time_since_abnormal_WBC'] = 0
            else:
                gold_ml.at[idx, 'time_since_abnormal_WBC'] = hrs if last_abnormal is None else hrs - last_abnormal
        else:
            gold_ml.at[idx, 'time_since_abnormal_WBC'] = hrs if last_abnormal is None else hrs - last_abnormal

# Select final columns and save to CSV
final_cols = ['subject_id','hadm_id','stay_id','hours_since_admission',
              'GCS','Heart Rate','Systolic BP','ShockIndex_current',
              'Albumin','WBC','CRP','Body Temp',
              'ShockIndex_trend_4h','time_since_abnormal_WBC',
              'shock_index_critical_12h']
gold_ml[final_cols].to_csv("gold_table_ml.csv", index=False)
