# extract_bronze.py
import pandas as pd
from sqlalchemy import create_engine


engine = create_engine("postgresql://postgres:postgres@localhost:5432/mimic4")


parameter_map = [
    (220050, "Systolic BP", "chartevents", "mimiciv_icu"),
    (220045, "Heart Rate", "chartevents", "mimiciv_icu"),
    (223901, "GCS", "chartevents", "mimiciv_icu"),
    (50825, "Body Temp", "labevents", "mimiciv_hosp"),
    (51300, "WBC", "labevents", "mimiciv_hosp"),
    (50889, "CRP", "labevents", "mimiciv_hosp"),
    (50862, "Albumin", "labevents", "mimiciv_hosp"),
]


def extract_parameter(param):
    itemid, name, table, schema = param
    if table == "chartevents":
        sql = f"""
            SELECT 
                subject_id,
                hadm_id,
                stay_id,
                charttime,
                '{name}' AS parameter_name,
                valuenum AS value,
                valueuom AS unit,
                '{table}' AS source_table,
                itemid
            FROM {schema}.{table}
            WHERE itemid = {itemid}
        """
    else:
        sql = f"""
            SELECT 
                subject_id,
                hadm_id,
                charttime,
                '{name}' AS parameter_name,
                valuenum AS value,
                valueuom AS unit,
                '{table}' AS source_table,
                itemid
            FROM {schema}.{table}
            WHERE itemid = {itemid}
        """
    return pd.read_sql(sql, engine)


frames = [extract_parameter(p) for p in parameter_map]
bronze_df = pd.concat(frames, ignore_index=True)

bronze_df.to_csv("burn_parameters.csv", index=False)
print("save burn_parameters.csv")
