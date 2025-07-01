import pandas as pd
import numpy as np

# Load the Silver-level data (standardized measurements)
silver_df = pd.read_csv('burn_parameters_silver.csv')
silver_df['charttime'] = pd.to_datetime(silver_df['charttime'])

# --- Unit Normalization and Data Cleaning ---
# Convert CRP from mg/L to mg/dL (divide values by 10):contentReference[oaicite:11]{index=11}
mask_crp = (silver_df['parameter_name'] == 'CRP') & (silver_df['unit'] == 'mg/L')
silver_df.loc[mask_crp, 'value'] = silver_df.loc[mask_crp, 'value'] / 10.0
silver_df.loc[mask_crp, 'unit'] = 'mg/dL'

# Convert body temperature from °F to °C if needed (T(°C) = (T(°F) - 32) * 5/9):contentReference[oaicite:12]{index=12}
mask_temp_f = (silver_df['parameter_name'] == 'Body Temp') & (silver_df['unit'] == '°F')
silver_df.loc[mask_temp_f, 'value'] = (silver_df.loc[mask_temp_f, 'value'] - 32.0) * (5.0 / 9.0)
silver_df.loc[mask_temp_f, 'unit'] = '°C'

# (Additional unit conversions can be handled similarly if other units are present)

# Drop measurements not associated with an ICU stay (ensure each row has a stay_id)
silver_df = silver_df.dropna(subset=['stay_id']).copy()
# Convert IDs to integers (after dropping NaN)
silver_df['stay_id'] = silver_df['stay_id'].astype(int)
silver_df['hadm_id'] = silver_df['hadm_id'].astype(int)

# Sort by patient and time for chronological consistency
silver_df.sort_values(by=['subject_id', 'stay_id', 'charttime'], inplace=True)

# --- Initialize data structure for Gold table ---
gold_rows = []  # will collect rows for each patient stay

# Identify the set of features of interest (OMOP concepts)
features = ['Albumin', 'Body Temp', 'CRP', 'GCS', 'Heart Rate', 'Systolic BP', 'WBC']

# Process each ICU stay independently
for (subj, hadm, stay), grp in silver_df.groupby(['subject_id', 'hadm_id', 'stay_id']):
    # Sort records by time within the stay
    grp = grp.sort_values('charttime')
    # Determine ICU admission time (t0) as the first charttime in this stay
    t0 = grp['charttime'].iloc[0]
    # Compute offset hours from ICU admission for each record
    # (floor division by 3600s to get the hour index since t0)
    offsets = ((grp['charttime'] - t0).dt.total_seconds() // 3600).astype(int)
    grp = grp.assign(offset_hours=offsets)

    # Aggregate values within each hour window for each parameter (mean aggregation)
    agg = grp.groupby(['offset_hours', 'parameter_name'])['value'].mean().unstack('parameter_name')
    # Ensure all feature columns are present in the aggregation (even if no data in some hours)
    for param in features:
        if param not in agg.columns:
            agg[param] = np.nan
    agg = agg[features]  # reorder columns to the defined feature list

    # Create a continuous hourly index from 0 to max offset to fill missing hours
    max_hour = int(agg.index.max())
    agg = agg.reindex(range(0, max_hour + 1), fill_value=np.nan)

    # Compute Shock Index for each hour (Heart Rate / Systolic BP):contentReference[oaicite:13]{index=13}
    agg['Shock Index'] = agg['Heart Rate'] / agg['Systolic BP']

    # Prepare window start and end times for each hour
    hours_index = agg.index.to_numpy()
    window_start_times = t0 + pd.to_timedelta(hours_index, unit='h')
    window_end_times = t0 + pd.to_timedelta(hours_index + 1, unit='h')

    # Combine identifier columns with time windows and feature values
    out_df = pd.DataFrame({
        'subject_id': subj,
        'hadm_id': hadm,
        'stay_id': stay,
        'window_start': window_start_times,
        'window_end': window_end_times
    })
    # Merge with the aggregated values (align by index which represents the hour offset)
    out_df = pd.concat([out_df.reset_index(drop=True), agg.reset_index(drop=True)], axis=1)

    # Append this stay's data to the list
    gold_rows.append(out_df)

# Concatenate all patient stays into the final Gold DataFrame
gold_df = pd.concat(gold_rows, ignore_index=True)

# Save the Gold table to CSV (each row = one patient-hour with features)
gold_df.to_csv('gold_table.csv', index=False)

print(f"Gold table constructed with shape {gold_df.shape}, saved to gold_table.csv")
