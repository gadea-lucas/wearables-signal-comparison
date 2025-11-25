import numpy as np
import neurokit2 as nk
from scipy.stats import skew
from scipy.signal import find_peaks, filtfilt


def skewness(signal, fs=None):
    """
    Elgendi, M. (2016). Optimal signal quality index for photoplethysmogram signals. Bioengineering, 3(4), 21
    """

    return np.abs(skew(signal))


def skewness_window(signal, fs=64, W=2):
    """
    Elgendi, M. (2016). Optimal signal quality index for photoplethysmogram signals. Bioengineering, 3(4), 21
    """

    window_size = fs * W

    windows = np.lib.stride_tricks.sliding_window_view(signal, window_size)

    skewness_vals = skew(windows, axis=1, bias=False)

    return np.mean(np.abs(skewness_vals))


def maki_quality(data, fs=64, MinPeakDistance=60/240, MinPeakHeight=0):

    def normalize_mean_peak_height(data):
        data = data - np.mean(data)
        idx_pks, _ = find_peaks(data, height=MinPeakHeight,
                                distance=int(round(MinPeakDistance * fs))+1)
        pks = data[idx_pks]
        dataout = data / np.mean(pks)
        return dataout

    signal = normalize_mean_peak_height(data)

    idx_pks, _ = find_peaks(signal, height=MinPeakHeight,
                            distance=int(round(MinPeakDistance * fs))+1)
    pks = signal[idx_pks]
    Q_PHV = np.var(pks)

    return Q_PHV


def neurokit(bvp, fs=1000):
    # print(bvp)
    # fs = 1000
    # print(nk.ppg.ppg_peaks(bvp, fs)[1])
    # ppg_quality = nk.ppg.ppg_quality(bvp)
    # result = np.mean(ppg_quality)
    # print(f"Longitud: {len(bvp)}")
    result = 0
    try:
        result = np.mean(nk.ppg.ppg_quality(bvp), sampling_rate=fs)
    except Warning:
        pass
    except Exception:
        pass
    return result


def bottcher_quality(eda, stamps=None, fs=4):
    """
    Evaluates the quality of an EDA (Electrodermal Activity) signal.

    Reference
    ---------
    Böttcher, S., Vieluf, S., Bruno, E., Joseph, B.,
    Epitashvili, N., Biondi, A., ... & Loddenkemper, T. (2022).
    Data quality evaluation in wearable monitoring. Scientific Reports, 12(1), 21412.

    Source code
    -----------
    https://github.com/WEAR-ISG/WEAR-DataQuality/blob/main/dependencies/WEAR_EvalSignalQuality.m

    Parameters
    ----------
    eda : array-like
        EDA signal data.
    stamps: array-like
        EDA timestamps in secods. This is not used in this implementation, but it could be easily adapted.
    fs : int, optional
        Sampling frequency in Hz, by default 4.

    Returns
    -------
    float
        Overall quality score of the EDA signal.
    """

    quality = {}

    def getRAC(signal, T=2):
        """
        Computes the Rate of Amplitude Change (RAC) over a given time window.

        Parameters
        ----------
        signal : array-like
            Input EDA signal.
        T : int, optional
            Time window in seconds, by default 2.

        Returns
        -------
        np.ndarray
            RAC values for the given signal.
        """
        intervals = range(0, len(signal), T * fs)

        rac = np.full(len(signal), np.nan)

        for ix in intervals:
            if (ix + T * fs) >= len(signal):
                continue

            windowdata = signal[ix: ix + T * fs]
            imin = np.argmin(windowdata)
            vmin = windowdata[imin]

            imax = np.argmax(windowdata)
            vmax = windowdata[imax]

            if imin < imax:
                rac[ix] = (vmax - vmin) / (abs(vmin) +
                                           1e-20)  # Avoid zero division
            elif imin > imax:
                rac[ix] = (vmin - vmax) / (abs(vmax) + 1e-20)

        # Fill missing values with the last available value
        last = np.nan
        for i in range(len(rac)):
            if not np.isnan(rac[i]):
                last = rac[i]
            else:
                rac[i] = last

        return rac

    def getWindowedMMScore(score, T=60, fs=4):
        """
        Computes the Moving Mean quality score over a specified time window.

        Parameters
        ----------
        score : array-like
            Quality values of the signal.
        stamps : array-like, optional
            Timestamps associated with the scores, by default None.
        T : int, optional
            Time window in seconds, by default 60.
        fs : int, optional
            Sampling frequency in Hz, by default 4 (Empatica E4).

        Returns
        -------
        np.ndarray
            Windowed mean quality scores.
        """
        moving_window = T * fs  # (T*fs)

        # movmean (MATLAB behavior)
        cumsum = np.cumsum(np.insert(score, 0, 0))
        head_mean = (cumsum[moving_window:] -
                     cumsum[:-moving_window]) / moving_window
        tail_means = np.array([np.mean(score[-i:])
                              for i in range(moving_window - 1, 0, -1)])
        scoremm = np.concatenate((head_mean, tail_means))

        max_length = min(len(score), len(stamps)) if stamps else len(score)
        return scoremm[:max_length:moving_window]  # Slice with sample_ix

    # Metric 1: Zero-line
    quality["metric1"] = eda
    quality["values1"] = eda >= 0.05

    # Metric 2: Rate of Amplitude Change
    quality["metric2"] = getRAC(eda)
    quality["values2"] = abs(quality["metric2"]) < 0.2

    # Scoring
    quality["values"] = quality["values1"] & quality["values2"]

    quality["score_windowed"] = getWindowedMMScore(
        quality["values"], T=60, fs=fs)  # Fix parameters if needed
    quality["score"] = np.mean(quality["score_windowed"])

    return quality["score"]


def kleckner_quality(data_EDA_uS,
                     fs=4,
                     data_time_sec=[],
                     data_temperature_C=[],
                     QA_filter_window_EDA_sec=None,
                     QA_EDA_floor=0.05,
                     QA_EDA_ceiling=60,
                     QA_EDA_max_slope_uS_per_sec=10,
                     QA_temperature_C_min=30,
                     QA_temperature_C_max=40,
                     QA_radius_to_spread_invalid_datum_sec=5):
    """
    Evaluates the quality of an EDA (Electrodermal Activity) signal based on the method described by Kleckner et al. (2017).

    Reference
    ---------
    Kleckner, I. R., et al. (2017). Methodological considerations for measuring electrodermal activity in 
    psychophysiological research. Psychophysiology, 54(6), 848-859.
    https://pubmed.ncbi.nlm.nih.gov/28976309/

    Source code
    -----------
    https://github.com/iankleckner/EDAQA/blob/master/run_automated_EDAQA.m


    Quality assessment rules applied
    -------------------------------
    Rule 1: EDA is out of range (not within 0.05-60 μS)  
        To prevent “floor” artifacts (e.g., electrode loses contact with skin) and “ceiling” artifacts (circuit overload).  
        The floor (0.05 μS) is chosen as the accepted minimum for skin conductance response amplitude.  
        The ceiling (60 μS) approximates the maximum measurable by typical devices (e.g., Q Sensor).

    Rule 2: EDA changes too quickly (faster than ±10 μS/sec)  
        To prevent high-frequency or “jump” artifacts indicating sudden, implausible changes in signal.

    Rule 3: Temperature is out of range (not within 30-40°C)  
        To account for times when the sensor is not worn or not properly contacting the skin, as skin temperature stabilizes around 32-36°C in worn devices.

    Rule 4: EDA data surrounding (within 5 sec of) invalid portions identified by Rules 1-3  
        To account for transitional artifacts close in time to detected invalid data.

    Parameters
    ----------
    data_EDA_uS : array-like
        Raw EDA signal data in microsiemens.
    fs : int, optional
        Sampling frequency in Hz, by default 4.
    data_time_sec : array-like, optional
        Timestamps of the EDA data in seconds, by default [] (ignored if empty).
    data_temperature_C : array-like, optional
        Corresponding temperature data in Celsius for each EDA sample, by default [].
    QA_filter_window_EDA_sec : float, optional
        Window size in seconds for smoothing/filtering the EDA and temperature signals, by default None (no filtering).
    QA_EDA_floor : float, optional
        Minimum acceptable EDA value in microsiemens, by default 0.05.
    QA_EDA_ceiling : float, optional
        Maximum acceptable EDA value in microsiemens, by default 60.
    QA_EDA_max_slope_uS_per_sec : float, optional
        Maximum allowed instantaneous slope of EDA signal in microsiemens per second, by default 10.
    QA_temperature_C_min : float, optional
        Minimum acceptable skin temperature in Celsius, by default 30.
    QA_temperature_C_max : float, optional
        Maximum acceptable skin temperature in Celsius, by default 40.
    QA_radius_to_spread_invalid_datum_sec : int, optional
        Number of seconds around an invalid sample to also mark as invalid, by default 5.

    Returns
    -------
    float
        Overall quality score as the proportion of valid EDA data points (between 0 and 1).
    """

    if not data_temperature_C:
        QA_temperature_C_min = 0
        QA_temperature_C_max = 1
        data_temperature_C = 0.5 * np.ones(len(data_EDA_uS))
    else:
        if QA_temperature_C_min >= QA_temperature_C_max:
            raise ValueError(
                'Temperature min must be less than temperature max')

    if (len(data_EDA_uS) != len(data_time_sec) and len(data_time_sec) != 0) or \
        (len(data_EDA_uS) != len(data_temperature_C) and len(data_temperature_C) != 0) or \
            (len(data_time_sec) != len(data_temperature_C) and len(data_time_sec) != 0 and len(data_temperature_C) != 0):
        raise ValueError(
            "Input data must all be the same length. If you do not have temperature or time data, use []")

    if QA_EDA_floor >= QA_EDA_ceiling:
        raise ValueError("EDA floor must be less than EDA ceiling")

    if data_time_sec:
        sampling_period_EDA = data_time_sec[1] - data_time_sec[0]
    else:
        sampling_period_EDA = 1 / fs  # None

    if QA_filter_window_EDA_sec:
        # Filter with small window for quality assessment
        windowSize = QA_filter_window_EDA_sec / sampling_period_EDA
        b = (1 / windowSize) * np.ones(int(windowSize))
        a = 1
        data_EDA_uS_filtered = filtfilt(b, a, data_EDA_uS)

        # Repeat filter for QA of temperature
        windowSize = QA_filter_window_EDA_sec / sampling_period_EDA
        b = (1 / windowSize) * np.ones(int(windowSize))
        a = 1
        data_temperature_C_filtered = filtfilt(b, a, data_temperature_C)

        # If temperature data are all NaN
        if np.sum(~np.isnan(data_temperature_C_filtered)) == 0:
            print("Warning: Temperature data could not be filtered for some reason. Using unfiltered temperature data for QA.")
            data_temperature_C_filtered = data_temperature_C

            print(
                f"\n\tNumber of NaNs in raw temperature data: {np.sum(np.isnan(data_temperature_C))} ({100 * np.sum(np.isnan(data_temperature_C)) / len(data_temperature_C):.0f}%)")

    else:
        # Use UNFILTERED data
        data_EDA_uS_filtered = data_EDA_uS
        data_temperature_C_filtered = data_temperature_C

    # Calculate instantaneous slope for Rule 2
    data_Q_EDA_uS_per_sec_filtered_QA = np.insert(
        (np.diff(data_EDA_uS_filtered) / sampling_period_EDA), 0, 0
    )

    # Implementation of EDA rules 1, 2, 3
    EDA_datum_invalid_123 = (
        (data_EDA_uS_filtered < QA_EDA_floor) |
        (data_EDA_uS_filtered > QA_EDA_ceiling) |
        (np.abs(data_Q_EDA_uS_per_sec_filtered_QA) > QA_EDA_max_slope_uS_per_sec) |
        (data_temperature_C_filtered < QA_temperature_C_min) |
        (data_temperature_C_filtered > QA_temperature_C_max)
    )

    # Determine number of data points to spread for Rule 4
    QA_radius_to_spread_invalid_datum_Ndata = int(
        QA_radius_to_spread_invalid_datum_sec / sampling_period_EDA
    )

    EDA_datum_invalid = EDA_datum_invalid_123.copy()
    for d in range(len(EDA_datum_invalid_123)):
        if EDA_datum_invalid_123[d]:

            # Propagate this to the end of the array of the point radius, whatever is smaller

            # Spread to right
            EDA_datum_invalid[d: d +
                              QA_radius_to_spread_invalid_datum_Ndata] = 1

            # Spread to left
            EDA_datum_invalid[
                max(0, d - QA_radius_to_spread_invalid_datum_Ndata + 1): d
            ] = 1

    return np.mean(~EDA_datum_invalid)


def kleckner_quality_filter(data_EDA_uS, fs=4):
    """
    Wrapper for `kleckner_quality` that applies a 2-second moving average filter window to the EDA data before quality assessment.

    Parameters
    ----------
    data_EDA_uS : array-like
        Raw EDA signal data in microsiemens.

    Returns
    -------
    float
        Overall quality score as the proportion of valid EDA data points after filtering.
    """

    return kleckner_quality(data_EDA_uS, fs=fs, QA_filter_window_EDA_sec=2)
