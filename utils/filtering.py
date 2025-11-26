import numpy as np
import pandas as pd
from scipy.signal import cheby2, sosfiltfilt, resample, butter
from scipy.ndimage import gaussian_filter1d
import neurokit2 as nk


def five_cheby2_gsr(df, column, sampling_rate=15):
    """
    Apply a 5th-order Chebyshev Type II low-pass filter to a GSR signal.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing the GSR signal.
    column : str
        Name of the column to filter.
    sampling_rate : int, optional
        Sampling rate of the signal (Hz), default is 15.

    Returns
    -------
    pandas.DataFrame
        A copy of the DataFrame with the smoothed signal.
    """
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


def butterworth_gsr(df, column, sampling_rate=15):
    """
    Apply a Butterworth low-pass filter to GSR using NeuroKit2.

    Notes
    -----
    Paper does not specify filter order; Neurokit's default used.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing the GSR signal.
    column : str
        Name of the column to filter.
    sampling_rate : int, optional
        Sampling rate of the signal (Hz), default is 15.

    Returns
    -------
    pandas.DataFrame
        A copy of the DataFrame with the smoothed signal.
    """
    column_values = df[column]

    # In the referenced paper, the filter order is not specified
    column_values = nk.signal.signal_filter(
        column_values, sampling_rate=sampling_rate, highcut=1
    )

    df[column] = column_values
    return df


def gaussian_gsr(df, column, sampling_rate=64):
    """
    Apply Gaussian smoothing to a GSR signal.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing the GSR signal.
    column : str
        Name of the column to filter.
    sampling_rate : int, optional
        Sampling rate of the signal (Hz), default is 15. Not used.

    Returns
    -------
    pandas.DataFrame
        A copy of the DataFrame with the smoothed signal.
    """
    sigma = 100  # Strong Gaussian smoothing
    df_copy = df.copy()
    df_copy[column] = gaussian_filter1d(df_copy[column].values, sigma=sigma)
    return df_copy


def four_cheby2_bvp(df, column, sampling_rate=64):
    """
    Apply a 4th-order Chebyshev Type II band-pass filter to BVP/PPG.

    Band-pass range: 0.5-5 Hz (typical for heart-rate-related components).

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing the GSR signal.
    column : str
        Name of the column to filter.
    sampling_rate : int, optional
        Sampling rate of the signal (Hz), default is 64.
    Returns
    -------
    pandas.DataFrame
    """
    sos = cheby2(
        N=4, rs=20, Wn=[0.5, 5], btype="bandpass",
        fs=sampling_rate, output="sos"
    )
    df[column] = sosfiltfilt(sos, df[column].values)
    return df


def butterworth_bvp(df, column, sampling_rate=64):
    """
    Apply a Butterworth band-pass filter to BVP/PPG using NeuroKit2.

    Frequency band: 1-15 Hz

   Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing the GSR signal.
    column : str
        Name of the column to filter.
    sampling_rate : int, optional
        Sampling rate of the signal (Hz), default is 64.

    Returns
    -------
    pandas.DataFrame
    """
    lowcut = 1
    highcut = 15

    column_values = nk.signal.signal_filter(
        df[column],
        sampling_rate=sampling_rate,
        lowcut=lowcut,
        highcut=highcut,
        order=5,
    )

    df[column] = column_values
    return df


def langevin_bandpass(df, column, sampling_rate=64):
    """
    Apply a Butterworth band-pass filter based on Langevin et al. (2021).

    Frequency band: 0.7-3.5 Hz
    Order: 4

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing the GSR signal.
    column : str
        Name of the column to filter.
    sampling_rate : int, optional
        Sampling rate of the signal (Hz), default is 64.

    Returns
    -------
    pandas.DataFrame
    """
    sos = butter(
        N=4, Wn=[0.7, 3.5], btype="bandpass",
        fs=sampling_rate, output="sos"
    )
    df[column] = sosfiltfilt(sos, df[column].values)
    return df


class DigitalFilter:
    """
    Simple IIR filter implementation based on EmotiBit firmware.
    https://github.com/EmotiBit/EmotiBit_XPlat_Utils/blob/master/src/DigitalFilter.cpp
    https://github.com/EmotiBit/EmotiBit_XPlat_Utils/blob/master/src/DigitalFilter.h

    Supports:
    - IIR_LOWPASS
    - IIR_HIGHPASS

    Parameters
    ----------
    filtertype : str
        Type of filter ("IIR_LOWPASS" or "IIR_HIGHPASS").
    samplingFreq : float
        Sampling frequency (Hz).
    filterFreq1 : float
        Cutoff frequency (Hz).
    """

    DIGITAL_FILTER_PI = 3.1415926535897932384626433832795
    DIGITAL_FILTER_E = 2.7182818284590452353602874713526

    def __init__(self, filtertype, samplingFreq, filterFreq1):
        self._type = filtertype
        self._alpha = pow(
            self.DIGITAL_FILTER_E,
            -2.0 * self.DIGITAL_FILTER_PI * filterFreq1 / samplingFreq
        )
        self._nInitSamples = 0
        self._nPoles = 1

    def filter(self, inputSample):
        """
        Filter a single sample and return the filtered output.
        """
        # Initialization: first sample passes directly
        if self._nInitSamples < self._nPoles:
            self._filteredValue = inputSample
            self._nInitSamples += 1

        if self._type == "IIR_LOWPASS":
            self._filteredValue = (
                inputSample * (1. - self._alpha)
                + self._filteredValue * self._alpha
            )
            return self._filteredValue

        elif self._type == "IIR_HIGHPASS":
            self._filteredValue = (
                inputSample * (1. - self._alpha)
                + self._filteredValue * self._alpha
            )
            return inputSample - self._filteredValue

        else:
            return 0.0


def emotibit_filter(df, column, freq):
    """
    Apply the EmotiBit-style IIR high-pass filter to PPG/BVP.
    https://github.com/EmotiBit/EmotiBit_FeatherWing/blob/master/EmotiBit.cpp#L3289C16-L3289C32.

    Parameters
    ----------
    df : pandas.DataFrame
    column : str
        Column containing the PPG signal.
    freq : float
        Sampling frequency.

    Returns
    -------
    pandas.DataFrame
    """
    ppgSensorHighpass = DigitalFilter("IIR_HIGHPASS", freq, 1)
    vec_func = np.vectorize(ppgSensorHighpass.filter)
    ppg_filtered = vec_func(df[column].values)
    df[column] = ppg_filtered
    return df
