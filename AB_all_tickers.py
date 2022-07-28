# This is the master bactest file for AB type signals | 2022-07-28 15:29

import pandas as pd
import numpy as np
from cindicatorAB_custom_bt import trade, step, bt_helper, backtest
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
    "ticker_file"  : "/home/u237/projects/backtests/cindicator-bt1/ft_userdata/user_data/data/binance_old/ZEC_USDT-1h.json"
    }

# dowload data loop
for i, j in enumerate(N_tickers):
    print(f"{i} - {j}")


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


