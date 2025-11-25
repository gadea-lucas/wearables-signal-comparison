import numpy as np
import pandas as pd
import neurokit2 as nk


def hampel_IQR_GSR_BVP(df, column, sampling_rate=64):
    df[column] = nk.rsp_clean(
        df[column].values, sampling_rate=sampling_rate, method="hampel"
    )

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)

    df_clean = df.copy()
    df_clean.loc[outliers, column] = np.nan

    df_clean[column] = pd.Series(df_clean[column]).interpolate(method="linear")

    df_clean[column] = df_clean[column].fillna(df_clean[column].mean())

    return df_clean


def IQR(df, column):

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)

    df_clean = df.copy()
    df_clean.loc[outliers, column] = np.nan

    df_clean[column] = pd.Series(df_clean[column]).interpolate(method="linear")

    df_clean[column] = df_clean[column].fillna(df_clean[column].mean())

    return df_clean
