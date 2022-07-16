# Decided to add a short signal, since freqtrade doesn't do that,
# I'll be overriding the default one with this one 
# This file is the copy of cindicatorAB_method_2.py at commit 400642f91ab043db383b9807cfa5745e713aa2f8
# 2022-07-07 17:38

"""
Some text
"""


## --- !! UNCOMMENT ONLY IN CASE OF ERROR ---
# from ctypes.wintypes import BOOL
# from freqtrade.strategy import IStrategy, merge_informative_pair
# from functools import reduce

# my imports / un-imports here
from typing import Dict, List
import pandas as pd
import arrow
from tqdm import tqdm
# launching tdqm pandas methods
tqdm.pandas(desc="my bar!")

# from pandas import pd.DataFrame
## --------------------------------

# parameters that will go to hyperopt
my_params = {
    "trade_buffer" : 0.99,
    "min_indicator": 80,   # for long signals
    "max_indicator": 20  # for short signals
    # these two will be directly multiplied with the raw signal price
    }

def effective_price(current_price):
    """
    returns the effective buy price, based on the current price
    """
    return current_price * my_params["trade_buffer"]

def first_is_recent_or_eq(ts1, ts2, use_arrow=True):
        """
        compares two unix timestamps, if the timetamps might be of different units, 
        keep use_arrow unchanged
        """
        if use_arrow:
            return arrow.get(ts1) >= arrow.get(ts2)
        return ts1 >= ts2

def return_current_signal_as_dict(signals_df_C, candle_T, high, low):
        """
        signals_df_C  - dataframe that contains parsed signal values indexed by datetime 
        candle_T    - unix timestamp of the current candle start time 
        !!WARNING   - assumes candle_T is the STARTING timestamp of a given candle  
        """
        # METHOD 2 | 2022-07-01 16:07 -
        # !! Unlike the last one, this requires iteration, so will be instersting to see which one is
        # !! faster
        # 1- start from first candle and first signal, go over the signals to find the latest one
        #   that is earlier than the candle itself. 
        # 2- If no such signal is found, assign empty dict, otherwise assign signals' values to that of the candle 
        # 3- If that one is not the earliest signal, delete from df.  
        # This way we'll always be assigning the data from the last signal, and avoid unnecessary comparisons 

        # start with empty dict, as we progress, we'll check if the length is 1 and drop other entries other than the latest
        index_earlier = []
        # goes from earliest to latest, 
        for index, row in signals_df_C.iterrows():
            # if the signal is earlier or eq to the candle timestamp
            if first_is_recent_or_eq(candle_T, row["M_dt"], use_arrow=False):
                # drop signal from df if high and low is not between last_dict["above"] and last_dict["below"]
                # todo (?) - this block is faulty, add a new column to df that check for this condition where there is a new signal
                # if (high >= row["above"]) or (low <= row["below"]):  # if not in range
                #     signals_df_C.drop(row.name, inplace=True) 
                #     continue
                index_earlier.append(index)
            # if the signal is later than the candle timestamp, we've reached the future, so stop the loop
            else: 
                break

        len_earlier = len(index_earlier)

        # no valid signals, return empty dict
        if len_earlier == 0:
            return {}
        # more than 1 signals that is earlier, drop the rest from df 
        elif len_earlier > 1:
            signals_df_C.drop(index_earlier[:-1], inplace=True) 
        # proceed without dropping if exactly 1 signal
            
        last_dict = signals_df_C.loc[index_earlier[-1]].to_dict()

        return last_dict

class trade():
    """
    Keeps the trade object that will be stored by class bt()
    Initialized with the signal object of the signal the trade was opened upon
    Trade status will be kept by class bt()
    """
    def __init__(self, signal, candle_index, my_params):
        self.my_params = my_params
        self.entry_index     = candle_index
        self.signal          = signal
        self.signal_id       = signal["Mid"]
        self.indicator       = signal["indicator"]
        self.effective_above = effective_price(signal["above"])
        self.effective_below = effective_price(signal["below"])
        self.effective_entry = effective_price(signal["base"])

        self.type        = None
        if   self.indicator  >= my_params["min_indicator"]:
             self.type        = "L"
        elif self.indicator  <= my_params["max_indicator"]:
             self.type        = "S"
        assert self.type is not None, "long / short type is None"


        # these will be filled upon exit
        self.exit_index     = None
        self.is_win         = None
        # self.is_above       = None
        # self.is_win         = (True if ((self.is_above and self.type == "L") or (not self.is_above and self.type == "S")) else False)
        self.exit_price     = None
        self.profit_percent = None
        
    def set_exit_attributes(self, exit_index, is_above: bool) -> None:
        """
        Sets the exit attributes of the trade   
        """
        self.exit_index    = exit_index

        if (is_above and self.type == "L"):
            self.is_win = True
            self.exit_price = self.effective_above
            self.profit_percent = (self.exit_price - self.effective_entry) / self.effective_entry
        elif (not is_above and self.type == "S"):
            self.is_win = True
            self.exit_price = self.effective_below
            self.profit_percent = (self.effective_entry - self.exit_price) / self.effective_entry
        elif (is_above and self.type == "S"):
            self.is_win = False
            self.exit_price = self.effective_above
            self.profit_percent = (self.effective_entry - self.exit_price) / self.effective_entry
        elif (not is_above and self.type == "L"):
            self.is_win = False
            self.exit_price = self.effective_below
            self.profit_percent = (self.exit_price - self.effective_entry) / self.effective_entry

        # make sure if one of the conditions have been met 
        assert self.is_win is not None, "is_win is None"

class bt():
    """
    This is our custom backtest class, this needs to be initialized inside the strategy class | 2022-07-06 20:49
    """

    def __init__(self) -> None:
        self.last_entry_signal_id = None
        self.last_exit_signal_id  = None
        self.buys_set             = set()   # these will be deprecated
        self.sells_set            = set()  # these will be deprecated
        self.open_trades          = [] # at the end of bt, we'll check this to see if any trades were left open
        self.closed_trades        = [] # at the end of bt, will be turned into a dataframe to give us the bt results

    def is_actual_buy(self, row : Dict) -> bool:
        """
        Called by populate_buy_trend whenever the nominal buying condition holds. 
        If the nominal buy condition is produced by a  new signal (and not for another that has already been bought)
        returns true after updating last_entry_signal_id to current signal, otherwise returns false. 
        """
        current_signal_dict = row.signal
        entry_index         = row.name
        current_signal_id   = current_signal_dict["Mid"]

        is_new_signal = (self.last_entry_signal_id != current_signal_id)

        if is_new_signal:

            self.last_entry_signal_id = current_signal_id
            # add the current signal to the set of bought signals
            self.buys_set.add(current_signal_id)

            # create the trade object with the current signal
            trade_obj = trade(current_signal_dict, entry_index)
            self.open_trades.append(trade_obj)

            return 1
        # pass since others will be pass as well
        else:
            return 0

    # def is_actual_sell(self, current_signal_dict, sell_index, is_above_exit: bool):
    def is_actual_sell(self, row: Dict) -> bool:
        """
        Called by populate_sell_trend whenever the nominal selling condition holds. 
        If the nominal sell condition is produced by a  new signal (and not for another that has already been sold)
        returns true after updating last_exit_signal_id to current signal, otherwise returns false. 
        """
        # check if the signal has been bought before
        # we check the set of bought signals, if the current signal is in the set, then it has been bought
        # if not, it has not been bought

        # todo - make sure these are accesed correctly
        sell_index = row.name
        current_signal_dict = row.signal

        # 2 is for high and 3 is for low
        is_above_exit = None
        if row.above >= row.loc[:,2] and row.above <= row.loc[:,3]:
            is_above_exit = True
        elif row.below >= row.loc[:,2] and row.below <= row.loc[:,3]:
            is_above_exit = False
        else:
            assert is_above_exit is not None, "is_above_exit is None"

        current_signal_id = current_signal_dict["Mid"]
        has_been_bought = self.last_entry_signal_id == current_signal_id

        # check if the signal has not been sold before
        new_sell_signal = (self.last_exit_signal_id != current_signal_id)

        if (new_sell_signal and has_been_bought):
            self.last_exit_signal_id = current_signal_id

            # remove the last item from open_trades and append to closed_trades
            last_trade = self.open_trades.pop()
            last_trade.set_exit_attributes(sell_index, sell_index, is_above_exit)
            self.closed_trades.append(last_trade)

            return 1
        # pass since others will be pass as well
        else:
            return 0

# class cindicatorAB_longshort(IStrategy):
class backtest():
    # So all the methods below are called only once per backtest and each time during the bot loop for 
    # live trading and dry run modes
    # keep this in mind..

    def __init__(self, signal_file, ticker_file, my_params: Dict) -> None:
        self.signal_file = signal_file #"/home/u237/projects/parsing_cindicator/data/CND_AB_parsed_fix1.json" 
        self.ticker_file = ticker_file #"/home/u237/projects/backtests/cindicator-bt1/ft_userdata/user_data/data/binance_old/ZEC_USDT-1h.json"
        self.my_params = my_params
        self.bt = bt()
        # we import our cindictor AB sorted df here for use in populate_indicators, also the ticker df
        self.signals = pd.read_json(signal_file)
        self.ticker_df = pd.read_json(ticker_file)

        # filter signals df for only our TICKER and those with indicator greater than min_indicator 
        self.signal_df_indicator_filtered = self.signals.loc[((self.signals.ticker == "ZEC/USD") & ~(self.signals.indicator.between(my_params["max_indicator"], my_params["min_indicator"], inclusive='neither'))), ['base', 'above', 'below',"indicator", "M_dt", "Mid"]]
        self.signal_df = self.signal_df_indicator_filtered.copy()
    
    def run_bactest(self):
        """
        TODO - 
        Will be populated with sequence of correct method calls
        after testing the class from a main function
        """
        # todo - apply effective_price() to necessary column in the signal df, remove applications of effective_price() in the rest of the code

    # An informative dictionary mapping ticker df column to their meanings - 2022-06-30 21:07
    ticker_df_column_ref = {
        "timestamp": 0,
        "open"     : 1,
        "high"     : 2,
        "low"      : 3,
        "close"    : 4,
        "volume"   : 5
    }

    def populate_indicators(self) -> pd.DataFrame:
        """
        !! row.loc[0] - the candle timestamp
        """
        # ticker_df["signal"]       = ticker_df.progress_apply(lambda row: return_current_signal_as_dict(self.signal_df, row.loc[0]), axis='columns')
        self.ticker_df["signal"]       = self.ticker_df.progress_apply(lambda row: return_current_signal_as_dict(self.signal_df, row.loc[0], row.loc[2], row.loc[3]), axis='columns')
        self.ticker_df["base"]         = self.ticker_df.apply(lambda row: row["signal"].get("base"), axis="columns")
        # self.ticker_df["base"]         = self.ticker_df.apply(lambda row: effective_price(row["signal"].get("base")), axis="columns")
        self.ticker_df["above"]        = self.ticker_df.apply(lambda row: effective_price(row["signal"].get("above")), axis="columns")
        self.ticker_df["below"]        = self.ticker_df.apply(lambda row: effective_price(row["signal"].get("below")), axis="columns")
        self.ticker_df["indicator"]    = self.ticker_df.apply(lambda row: row["signal"].get("indicator"), axis="columns")
        self.ticker_df["M_dt"]         = self.ticker_df.apply(lambda row: row["signal"].get("M_dt"), axis="columns")
        self.ticker_df["Mid"]          = self.ticker_df.apply(lambda row: row["signal"].get("Mid"), axis="columns")

    # |*|
    def populate_buy_trend(self) -> pd.DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: pd.DataFrame
        :return: pd.DataFrame with buy column
        """

        # returns the signals that might be a buy, we will then iterate over this subset 
        # to see which ones are actual buys based on "is_actual_buy" 
        df_copy = self.ticker_df
        df_copy.assign(buy = 0)

        # FILTER FOR THE NOMINAL BUY CONDITION
        # we esentially buy whenever the (base price + trade_buffer) is within low price and high price 
        # (this will go to hyperopt later) 
        df_nominal_buy = df_copy.loc[
            (df_copy['base']).between(df_copy.loc[:,3], df_copy.loc[:,2], inclusive='both')
        ]

        df_copy["buy"] = df_nominal_buy.progress_apply(lambda row: self.bt.is_actual_buy(row.signal), axis='columns')

    # |*|
    def populate_sell_trend_orig(self) -> pd.DataFrame:
        """
        !! Deprecated, use populate_sell_trend instead | 2022-07-15 14:58
        Based on TA indicators, populates the sell signal for the given ticker_df
        !! Assumes both above and below price cannot be wihtin the same candle
        :param ticker_df: pd.DataFrame
        :return: pd.DataFrame with buy column
        """

        # add a new column "sell" to ticker_df, initialize it to 0
        ticker_df = ticker_df.assign(sell = 0)

        # filter for the nominal sell condition
        # we sell whenever the either above or below price is within the low and high of the candle
        df_nominal_sell = ticker_df.loc[
            (ticker_df['above'] * my_params["trade_buffer"]).between(ticker_df.loc[:,3], ticker_df.loc[:,2], inclusive='neither') |
            (ticker_df['below'] * my_params["trade_buffer"]).between(ticker_df.loc[:,3], ticker_df.loc[:,2], inclusive='neither')
        ]
        
        ticker_df["sell"] = df_nominal_sell.progress_apply(lambda row: self.bt.is_actual_sell(row.Mid), axis='columns')

        return ticker_df

    def populate_sell_trend(self) -> pd.DataFrame:
        """
        Based on TA indicators, populates the sell signal on the self.ticker_df 
        !! Assumes both above and below price cannot be wihtin the same candle
        :return: pd.DataFrame with buy column
        """
        # !! this line is important
        dataframe = self.ticker_df

        # add a new column "sell" to dataframe, initialize it to 0
        # dataframe = dataframe.assign(sell = 0)
        dataframe.assign(sell = 0)

        df_nominal_above = dataframe.loc[
            (dataframe['above'] * my_params["trade_buffer"]).between(dataframe.loc[:,3], dataframe.loc[:,2], inclusive='neither')
        ]
        df_nominal_below = dataframe.loc[
            (dataframe['below'] * my_params["trade_buffer"]).between(dataframe.loc[:,3], dataframe.loc[:,2], inclusive='neither')
        ]
        #     def is_actual_sell(self, current_signal_dict, sell_index, is_above_exit: bool):

        dataframe["sell"] = df_nominal_above.progress_apply(lambda row: self.bt.is_actual_sell(current_signal_dict=row.signal, sell_index=row.name, is_above_exit=1), axis='columns')
        dataframe["sell"] = df_nominal_below.progress_apply(lambda row: self.bt.is_actual_sell(current_signal_dict=row.signal, sell_index=row.name, is_above_exit=0), axis='columns')

    def display_results(self):
        min_indicator = my_params["min_indicator"]
        max_indicator = my_params["max_indicator"]
        tot_filtered = len(self.signal_df_indicator_filtered)
        tot_sells = len(self.bt.sells_set)
        tot_bought_never_sold = len(self.bt.sells_set)
        tot_never_bought = tot_filtered - (tot_sells + tot_bought_never_sold)

        print("======THE BACKTEST OF IS OVER======")
        print(f"The number of signals outside range ({min_indicator}, {max_indicator}) is {tot_filtered}")
        print(f"The number of total sells is {tot_sells}")
        print(f"The number of signals never sold is {tot_bought_never_sold}") # all signals that were sold were popped from here
        print(f"The number of signals never bought is {tot_never_bought}")
        print("=====================================")


