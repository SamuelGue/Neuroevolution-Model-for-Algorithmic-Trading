import pandas as pd
import numpy as np

def ADX(df, period=14):
    """
    Computes the 'ADX' (Average Directional Movement Index) for the given dataframe.

    Parameters
    ----------
    df : pd.Dataframe
        Must contain columns 'open', 'close', 'low', 'high', 'volume', 'date'
    period : int
        Lookback period for smoothing (default 14, the standard Wilder period)

    Returns
    -------
    pd.DataFrame
        Copy of df with '+di', '-di', and 'adx' columns added.
    """
    df = df.copy()

    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    # TRUE RANGE
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # DIRECTIONAL MOVEMENT
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)

    # WILDER SMOOTHING
    def wilder_smooth(series, period):
        return series.ewm(alpha = 1 / period, adjust = False, min_periods = period).mean()

    atr = wilder_smooth(tr, period)
    smoothed_plus_dm = wilder_smooth(plus_dm, period)
    smoothed_minus_dm = wilder_smooth(minus_dm, period)

    plus_di = 100 * (smoothed_plus_dm / atr)
    minus_di = 100 * (smoothed_minus_dm / atr)

    # DX & ADX
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = wilder_smooth(dx, period)

    df['+di'] = plus_di
    df['-di'] = minus_di
    df['ADX'] = adx

    return df

def CCI(df, period=14):
    """
    Computes the 'CCI' (Commodity Channel Index) for the given dataframe.
    
    Parameters
    ----------
    df : pd.Dataframe
        Must contain columns 'open', 'close', 'low', 'high', 'volume', 'date'
    period : int
        Lookback period for smoothing (default 14, the standard Wilder period)
    
    Returns
    -------
    pd.DataFrame
        Copy of df with 'CCI' column added.
    """
    df = df.copy()

    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    tp = (high + low + close) / 3

    sma = pd.Series(tp).rolling(period).mean()

    def MAD(x):
        return np.abs(x - x.mean()).mean()
    
    md = tp.rolling(period).apply(MAD, raw=True)

    df["CCI"] = (tp - sma) / (0.015 * md)

    return df

def VI(df, period=14):
    """
    Computes the 'VI' (Vortex Indicators) for the given dataframe.
        
    Parameters
    ----------
    df : pd.Dataframe
        Must contain columns 'open', 'close', 'low', 'high', 'volume', 'date'
    period : int
        Lookback period for smoothing (default 14, the standard Wilder period)
        
    Returns
    -------
    pd.DataFrame
        Copy of df with 'VI+' and 'VI-' columns added.
    """
    df = df.copy()

    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    # TRUE RANGE
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # VORTEX MOVEMENTS
    prev_low = low.shift(1)
    prev_high = high.shift(1)

    vm_up = (high - prev_low).abs()
    vm_down = (low - prev_high).abs()

    sum_tr = tr
    sum_vm_up = vm_up
    sum_vm_down = vm_down

    for i in range(1, period):
        sum_tr += tr.shift(i)
        sum_vm_up += vm_up.shift(i)
        sum_vm_down += vm_down.shift(i)

    # VI+ and VI-

    df["VI+"] = sum_vm_up / sum_tr
    df["VI-"] = sum_vm_down / sum_tr

    return df

def RSI(df, period=14):
    """
    Computes the 'RSI' (Relative Strength Index) for the given dataframe.
    
    Parameters
    ----------
    df : pd.Dataframe
        Must contain columns 'open', 'close', 'low', 'high', 'volume', 'date'
    period : int
        Lookback period for smoothing (default 14, the standard Wilder period)
    
    Returns
    -------
    pd.DataFrame
        Copy of df with 'RSI' columns added.
    """
    df = df.copy()

    # UP & DOWN PERIODS

    close = df["Close"]
    diffs = close - close.shift(1)

    U = np.where(diffs > 0, diffs, 0)
    D = np.where(diffs < 0, -diffs, 0)

    # EXPONENTIAL SMOOTHING

    def wilders_smma(series, period):
        values = series
        smma = np.full(len(values), np.nan)

        if len(values) >= period:
            smma[period - 1] = values[:period].mean()
            for i in range(period, len(values)):
                smma[i] = (smma[i-1] * (period - 1) + values[i]) / period

        return smma

    # RELATIVE STRENGTH

    RS = wilders_smma(U, period) / wilders_smma(D, period)

    # RELATIVE STRENGTH INDEX

    df["RSI"] = 100 - 100 / (1 + RS)

    return df
        
def Williams_Percent_R(df, period=14):
    """
    Computes the Williams %R for the given dataframe.
    
    Parameters
    ----------
    df : pd.Dataframe
        Must contain columns 'open', 'close', 'low', 'high', 'volume', 'date'
    period : int
        Lookback period for smoothing (default 14, the standard Wilder period)
    
    Returns
    -------
    pd.DataFrame
        Copy of df with '%R' column added.
    """    
    df = df.copy()

    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    # PERIOD HIGHS & LOWS

    high_N_days = high.rolling(period).apply(max, raw=True)
    low_N_days = low.rolling(period).apply(min, raw=True)

    # Williams %R

    df["Percent_R"] = -100 * (high_N_days - close) / (high_N_days - low_N_days)

    return df

def FI(df):
    """
    Computes the Force Index (FI) for the given dataframe.
    
    Parameters
    ----------
    df : pd.Dataframe
        Must contain columns 'open', 'close', 'low', 'high', 'volume', 'date'
    
    Returns
    -------
    pd.DataFrame
        Copy of df with 'FI' column added.
    """    
    df = df.copy()

    close = df["Close"]

    volume = df["Volume"]

    df["FI"] = (close - close.shift(1)) * volume

    return df

def OBV(df):
    """
    Computes the On-Balance Volume (OBV) for the given dataframe.
    
    Parameters
    ----------
    df : pd.Dataframe
        Must contain columns 'open', 'close', 'low', 'high', 'volume', 'date'
    
    Returns
    -------
    pd.DataFrame
        Copy of df with 'OBV' column added.
    """
    df = df.copy()

    close = df["Close"]
    volume = df["Volume"]

    OBV_arr = np.full(len(close), 0)
    for i in range(1,len(close)):
        if close[i] > close[i-1]:
            OBV_arr[i] = OBV_arr[i-1] + volume[i]
        elif close[i] == close[i-1]:
            OBV_arr[i] = OBV_arr[i-1]
        else:
            OBV_arr[i] = OBV_arr[i-1] - volume[i]

    df["OBV"] = OBV_arr
        
    return df

def VPT(df):
    """
    Computes the Volume-Price Trend (VPT) for the given dataframe.
    
    Parameters
    ----------
    df : pd.Dataframe
        Must contain columns 'open', 'close', 'low', 'high', 'volume', 'date'
    
    Returns
    -------
    pd.DataFrame
        Copy of df with 'VPT' column added.
    """
    df = df.copy()

    close = df["Close"]
    volume = df["Volume"]

    VPT_arr = np.full(len(close), 0)
    for i in range(1, len(close)):
        VPT_arr[i] = VPT_arr[i-1] + volume[i] * (close[i] - close[i-1]) / close[i-1]

    df["VPT"] = VPT_arr

    return df

def ADI(df):
    """
    Computes the Accumulation/Distribution Index (ADI) for the given dataframe.
    
    Parameters
    ----------
    df : pd.Dataframe
        Must contain columns 'open', 'close', 'low', 'high', 'volume', 'date'
    
    Returns
    -------
    pd.DataFrame
        Copy of df with 'ADI' column added.
    """
    df = df.copy()
    eps = 1e-10
    close = df["Close"]
    low = df["Low"]
    high = df["High"]
    volume = df["Volume"]

    CLV = (close - low) - (high - close) / (high - low + eps)

    accdist = np.full(len(low),0)
    for i in range(1, len(low)):
        accdist[i] = accdist[i-1] + volume[i] * CLV[i]

    df["ADI"] = accdist

    return df

def normalize_prices(df):
    """
    Computes the price of a stock normalized by its initial value for the given dataframe.
        
    Parameters
    ----------
    df : pd.Dataframe
        Must contain columns 'open', 'close', 'low', 'high', 'volume', 'date'
        
    Returns
    -------
    pd.DataFrame
        Copy of df with 'Normalized_Open_Price' column added.
    """
    df = df.copy()

    opens = df["Open"]
    close = df["Close"]
    opens_0 = opens.loc[0]
    close_0 = close.loc[0]

    df["Normalized_Open_Price"] = opens / opens_0
    df["Normalized_Close_Price"] = close / close_0

    return df

def setup_dataframe(df, period = 14):
    df = df.copy()
    df = ADX(df=df,period=period)
    df = CCI(df=df,period=period)
    df = VI(df=df,period=period)
    df = RSI(df=df,period=period)
    df = Williams_Percent_R(df=df,period=period)
    df = FI(df=df)
    df = OBV(df=df)
    df = VPT(df=df)
    df = ADI(df=df)
    df = normalize_prices(df=df)

    return df