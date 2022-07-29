# This is the master bactest file for AB type signals | 2022-07-28 15:29

import pandas as pd
import numpy as np
from cindicatorAB_custom_bt import trade, step, bt_helper, backtest, bt_top_N
from tqdm import tqdm

# of 750 signals and 135 tickers, the selections of and total corresponding signals are given below
# 10 - 170
# 20 - 291
# 25 - 342
# 30 - 388
N = 20

tqdm.pandas(desc="my bar!")
signal_file = "/home/u237/projects/parsing_cindicator/data/CND_AB_parsed_fix1.json"
signals_df   = pd.read_json(signal_file)

# getting top N tickers as a series object to iterate in the loop
ticker_counts = signals_df.value_counts("ticker", dropna=False)
N_tickers = ticker_counts.head(N)

# initalize the empty df trades that each loop is going to append rows to
col_names = [
    # !! Have decided that "ticker" wont be a column but a second level index instead"
    'indicator', 'is_win', '%profit', 'M_dt', 'candle1_dt', 
    'candle2_dt', 'low1', 'high1', 'entry_price', 'base', 'low2', 'high2', 
    'exit_price', 'above', 'below', 'elapsed_hours', 'elapsed_days'
]
df_all_tickers = pd.DataFrame(columns=col_names, ignore_index=True) 

my_params = {
    "trade_buffer" : 0.99,
    "min_indicator": 80, # for long signals
    "max_indicator": 20, # for short signals
    "ticker"       : "ZEC_USDT",
    "signal_file"  : "/home/u237/projects/parsing_cindicator/data/CND_AB_parsed_fix1.json",
    "ticker_file"  : "/home/u237/projects/backtests/cindicator-bt1/ft_userdata/user_data/data/binance/ZEC_USDT-1h.json",
    "N_tickers"    : N_tickers,
    "N"            : len(N_tickers)
    }

def download_command(ticker):
    # return "wget -O /home/u237/projects/backtests/cindicator-bt1/ft_userdata/user_data/data/binance_old/" + ticker + "-1h.json https://api.binance.com/api/v1/klines?symbol=" + ticker + "&interval=1h"
    # c_string = f"dcr freqtrade download-data -t 15m --timerange 20180401- --pairs {ticker} --exchange binance"
    c_string = f"dcr freqtrade download-data -t 30m --timerange 20180401- --exchange binance --pairs-file /home/u237/projects/backtests/cindicator-bt1/ft_userdata/user_data/data/binance/pairs.json"

def tickers_to_config(N_tickers):
    """
    To be used before the backtest, externally
    The output is added to the appropriate place in config.json
    Print all tickers in N_tickers as json list of string  | 2022-07-29 01:00
    """
    tickers = N_tickers.index.tolist()

    # append T to the end of any item in tickers that ends with "USD"
    for i, ticker in enumerate(tickers):
        if ticker.endswith("USD"):
            tickers[i] = ticker + "T"
        # also add double quotes to the tickers
        tickers[i] = '"' + tickers[i] + '"'

    tickers_json = "[\n\t" + ",\n\t".join(tickers) + "\n]"
    print(tickers_json)    

# the bactest loop
for i, ticker in enumerate(N_tickers):
    my_params["ticker"] = ticker
    bt = backtest(my_params)
    df_ticker = bt.bt_helper.df_closed
    # df_ticker["ticker"] = ticker

    # add the ticker as a second level index
    pd.concat([df_ticker], keys=[ticker], names=['ticker'])
    # append df_ticker to df_tickers
    df_all_tickers = pd.concat([df_all_tickers, df_ticker])

    print(f"{i} - {ticker}")


