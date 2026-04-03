import pandas as pd
import numpy as np

def build_features(df):

    # -------------------------------
    # 1. HANDLE ANOMALIES
    # -------------------------------
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)

    # -------------------------------
    # 2. TIME TRANSFORMATIONS
    # -------------------------------
    df["AGE_YEARS"] = abs(df["DAYS_BIRTH"]) / 365
    df["YEARS_EMPLOYED"] = abs(df["DAYS_EMPLOYED"]) / 365

    # -------------------------------
    # 3. FINANCIAL RATIOS
    # -------------------------------
    df["CREDIT_TO_INCOME_RATIO"] = df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + 1)
    df["ANNUITY_TO_INCOME_RATIO"] = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] + 1)

    if "CNT_FAM_MEMBERS" in df.columns:
        df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / (df["CNT_FAM_MEMBERS"] + 1)
    else:
        df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"]

    # -------------------------------
    # 4. EMPLOYMENT STABILITY
    # -------------------------------
    df["EMPLOYED_TO_AGE_RATIO"] = df["YEARS_EMPLOYED"] / (df["AGE_YEARS"] + 1)

    # -------------------------------
    # 5. CREDIT STRUCTURE
    # -------------------------------
    if "AMT_GOODS_PRICE" in df.columns:
        df["CREDIT_TO_GOODS_RATIO"] = df["AMT_CREDIT"] / (df["AMT_GOODS_PRICE"] + 1)
    else:
        df["CREDIT_TO_GOODS_RATIO"] = df["AMT_CREDIT"]

    # -------------------------------
    # 6. EXTERNAL SCORES
    # -------------------------------
    ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    available_ext = [c for c in ext_cols if c in df.columns]

    if len(available_ext) > 0:
        df["EXT_SOURCE_MEAN"] = df[available_ext].mean(axis=1)
        df["EXT_SOURCE_STD"] = df[available_ext].std(axis=1)
    else:
        df["EXT_SOURCE_MEAN"] = 0
        df["EXT_SOURCE_STD"] = 0

    # -------------------------------
    # 7. SAFE FILL
    # -------------------------------
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    return df