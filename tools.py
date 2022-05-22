def pre_btc():
    import numpy as np
    from pathlib import Path
    import pandas as pd
    p = Path('data/btc-indicators.csv')
    df = pd.read_csv(p, parse_dates=["Time"])
    df['date'] = pd.to_datetime(df["Time"]).dt.date
    df['Log(Price)'] = np.log(df['BTC / Closing Price'])
    df.to_csv(p)

def raw_fin_d():
    import ccxt
    from pathlib import Path
    import pandas as pd
    from finta import TA
    # bn = ccxt.binance()
    # bn.fetch_ohlcv('BTC/USDT', timeframe='1d', limit=800)
    # prices = bn.fetch_ohlcv('BTC/USDT', timeframe='1d', limit=800)
    # df = pd.DataFrame(prices)
    # df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    # df['date'] = pd.to_datetime(df['date'], unit='ms').dt.date
    # df.to_csv('data/btc-raw.csv')
    ohlc = pd.read_csv('data/btc-raw.csv')
    for indicator in [
        'MACD', # Moving Average Convergence Divergence
        'EMA', # Exponential Moving Average
        'RSI', # Relative Strenght Index
    ]:
        newdf = getattr(TA, indicator)(ohlc)
        if isinstance(newdf, pd.DataFrame):
            for col in newdf.columns:
                ohlc[col] = newdf[col]
        else:
            ohlc[indicator] = newdf
    ohlc['date'] = pd.to_datetime(ohlc['date']).dt.date

    # p = Path('data/btc-indicators.csv')
    # df = pd.read_csv(p, parse_dates=["date"])
    # df['date'] = pd.to_datetime(df["date"]).dt.date
    # df = df.merge(ohlc, how="left", on="date")
    # df.to_csv(p)
    # dependency
    data_path = Path('data')
    for which in ['reddit', 'telegram']:
        p = data_path / 'agg_sia_ind_{}.csv'.format(which)
        df_features = pd.read_csv(p, parse_dates=["date"])
        df_features['date'] = pd.to_datetime(df_features['date'], unit='ns').dt.date
        df_X = df_features
        df_m = df_X.merge(ohlc, how="left", on="date")
        df_m.to_csv(p)
raw_fin_d()