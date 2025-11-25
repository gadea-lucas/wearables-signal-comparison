import numpy as np
import pandas as pd
from scipy.signal import cheby2, sosfiltfilt, resample, butter
from scipy.ndimage import gaussian_filter1d
import neurokit2 as nk


def five_cheby2_gsr(df, column, sampling_rate=64):
    sos = cheby2(
        N=5,
        rs=20,
        Wn=0.05,
        btype="lowpass",
        fs=sampling_rate,
        output="sos"
    )
    df[column] = sosfiltfilt(sos, df[column].values)
    return df


def butterworth_gsr(df, column, sampling_rate=64):
    column_values = df[column]

    column_values = nk.signal.signal_filter(
        column_values, sampling_rate=sampling_rate, highcut=1
    )  # En el paper no indica order, por defecto es 2

    df[column] = column_values

    return df


def gaussian_gsr(df, column, sampling_rate=4):
    sigma = 100  # suavizado gaussiano
    df_copy = df.copy()
    df_copy[column] = gaussian_filter1d(
        df_copy[column].values, sigma=sigma)
    return df_copy


def four_cheby2_bvp(df, column, sampling_rate=64):
    sos = cheby2(
        N=4, rs=20, Wn=[0.5, 5], btype="bandpass", fs=sampling_rate, output="sos"
    )
    df[column] = sosfiltfilt(sos, df[column].values)
    return df


def butterworth_bvp(df, column, sampling_rate=64):
    column_values = df[column]

    lowcut = 1
    highcut = 15

    column_values = nk.signal.signal_filter(
        column_values,
        sampling_rate=sampling_rate,
        lowcut=lowcut,
        highcut=highcut,
        order=5,
    )

    df[column] = column_values

    return df

# 1. Langevin et al. (2021): Band-pass filter 0.7 - 3.5 Hz


def langevin_bandpass(df, column, sampling_rate=64):
    sos = butter(
        N=4, Wn=[0.7, 3.5], btype="bandpass", fs=sampling_rate, output="sos"
    )
    df[column] = sosfiltfilt(sos, df[column].values)
    return df
