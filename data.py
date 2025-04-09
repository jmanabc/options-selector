import pandas as pd


def get_options_data(ticker):
    df = pd.read_csv('strikes_202402091600.csv')
    df_ticker = df[df['ticker'] == f'{ticker}']
    df_ticker.loc[:, 'tradeDate'] = pd.to_datetime(df_ticker['tradeDate'])

    return df_ticker


def get_hist_equity_data(ticker):
    df = pd.read_csv(f'{ticker.lower()}_hist_price.csv')
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
    df = df[['Date', 'Close/Last']].iloc[::-1]
    df['Close/Last'] = (
        df['Close/Last'].str.replace(r'\$', '', regex=True).astype(float)
    )

    return df
