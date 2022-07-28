import pandas as pd
import numpy as np
from cindicatorAB_custom_bt import trade, step, bt_helper, backtest
from tqdm import tqdm

tqdm.pandas(desc="my bar!")
signal_file = "/home/u237/projects/parsing_cindicator/data/CND_AB_parsed_fix1.json"
signals_df   = pd.read_json(signal_file)


ticker_list = []

for i, t in enumerate(ticker_list):
    pass

