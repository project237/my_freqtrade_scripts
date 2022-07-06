# testing new repo | 2022-06-28 16:37

"""
ok, so this will be the strategy file where we are gonna be running for each single iteration of 
the backtest loop. 

1- Inside populate_indicators, we simply use M_dt column to the find the correct 15 min candle and insert the columns "buy", 
price1-3 and indicator
as new column to the backtest dataframe. All but first columns will have the same value for each row (TODO - workaround) and 
the column "buy" will insert a 1 for an otherwise 0 on whichever row matches the the M_dt   

2- Inside populate_buy_trend we just instruct it to buy when the column "buy" is 1. 

3- Inside populate_sell_trend we instruct it to sell whenever either price2 or price3 is within the candle range. 

# todo - check the script that calls these methods to see if you need to create a copy of df within the methods. 
"""


## --- Do not remove these libs ---
from ctypes.wintypes import BOOL
from freqtrade.strategy import IStrategy, merge_informative_pair
from typing import Dict, List
from functools import reduce

# my imports / un-imports here
import pandas as pd
import arrow
# from pandas import pd.DataFrame
## --------------------------------

def first_is_recent_or_eq(ts1, ts2, use_arrow=True):
        """
        compares two unix timestamps, if the timetamps might be of different units, 
        keep use_arrow unchanged
        """
        if use_arrow:
            return arrow.get(ts1) >= arrow.get(ts2)
        return ts1 >= ts2

def return_current_signal_as_dict(signals_df_C, candle_T):
        """
        signals_df_C  - dataframe that contains parsed signal values indexed by datetime 
        candle_T    - unix timestamp of the current candle start time 
        !!WARNING   - assumes candle_T is the STARTING timestamp of a given candle  
        TODO - CHECK ASSUMPTION ABOVE
        """
        # METHOD 2 | 2022-07-01 16:07 -
        # !! Unlike the last one, this requires iteration, so will be instersting to see which one is
        # !! faster
        # 1- start from first candle and first signal, go over the signals to find the latest one
        #   that is earlier than the candle itself. 
        # 2- If no such signal is found, assign empty dict, otherwise assign signals' values to that of the candle 
        # 3- If that one is not the earliest signal, delete from df.  
        # This way we'll always be assigning the data from the last signal, and avoid unnecessary comparisons 


        index_earlier = []
        # goes from eraliest to latest, 
        for index, row in signals_df_C.iterrows():
            # if the signal is later or eq to the candle timestamp
            if first_is_recent_or_eq(candle_T, row["M_dt"], use_arrow=False):
                index_earlier.append(index)
            # if the signal is later than the candle timestamp
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

class bt():
    """
    This is our custom bactest class, this needs to be initialized inside the strategy class | 2022-07-06 20:49
    """

    def __init__(self) -> None:
        self.last_buy_signal_id = None
        self.last_sell_signal_id = None
        self.buys_set = set()
        self.sells_set = set()

    def is_actual_buy(self, current_signal_id):
        """
        Called by populate_buy_trend whenever the nominal buying condition holds. 
        If the nominal buy condition is produced by a  new signal (and not for another that has already been bought)
        returns true after updating last_buy_signal_id to current signal, otherwise returns false. 
        """

        new_signal = (self.last_buy_signal_id != current_signal_id)

        if new_signal:
            self.last_buy_signal_id = current_signal_id
            # add the current signal to the set of bought signals
            self.buys_set.add(current_signal_id)
            return 1
        # pass since others will be pass as well
        else:
            return 0

    def is_actual_sell(self, current_signal_id):
        """
        Called by populate_sell_trend whenever the nominal selling condition holds. 
        If the nominal sell condition is produced by a  new signal (and not for another that has already been sold)
        returns true after updating last_sell_signal_id to current signal, otherwise returns false. 
        """
        # check if the signal has been bought before
        # we check the set of bought signals, if the current signal is in the set, then it has been bought
        # if not, it has not been bought
        has_been_bought = (current_signal_id in self.buys_set)

        # check if the signal has not been sold before
        new_sell_signal = (self.last_sell_signal_id != current_signal_id)

        if (new_sell_signal and has_been_bought):
            self.last_sell_signal_id = current_signal_id
            # add the current signal to the set of sold signals
            self.sells_set.add(current_signal_id)
            return 1
        # pass since others will be pass as well
        else:
            return 0

class InformativeSample(IStrategy):
    # So all the methods below are called only once per backtest and each time during the bot loop for 
    # live trading and dry run modes
    # keep this in mind..

    # initialize the custom bt class
    bt_obj = bt()

    # An informative dictionary mapping ticker df column to their meanings - 2022-06-30 21:07
    ticker_df_column_ref = {
        "timestamp": 0,
        "open"     : 1,
        "high"     : 2,
        "low"      : 3,
        "close"    : 4,
        "volume"   : 5
    }
    # parameters that will go to hyperopt
    my_params = {
        "buy_buffer" : 1.01,
        "sell_buffer": 1.01,
        "min_indicator": 85
        # these two will be directly multiplied with the raw signal price
     }
    
    signal_file = "/home/u237/projects/parsing_cindicator/data/CND_AB_parsed_fix1.json"

    # we import our cindictor AB sorted df here for use in populate_indicators
    signals = pd.read_json(signal_file)

    # filter signals df for only our TICKER and those with indicator greater than min_indicator 
    signal_df_indicator_filtered = signals.loc[((signals["ticker"] == "ZEC/USD") & (signals["indicator"] > my_params["min_indicator"])), ['base', 'above', 'below',"indicator", "M_dt", "Mid"]]
    signal_df = signal_df_indicator_filtered.copy()


    ## Minimal ROI designed for the strategy.
    ## This attribute will be overridden if the config file contains "minimal_roi"
    minimal_roi = {
        "60":  0.01,
        "30":  0.03,
        "20":  0.04,
         "0":  0.05
    }

    ## Optimal stoploss designed for the strategy
    ## This attribute will be overridden if the config file contains "stoploss"
    # !! leaving this ratio well below what the exit signal should be doing,
    # !! in effect, we're not using this
    stoploss = -0.25

    ## Optimal timeframe for the strategy
    timeframe = '1h'

    ## trailing stoploss
    trailing_stop                 = False
    trailing_stop_positive        = 0.02
    trailing_stop_positive_offset = 0.04

    ## run "populate_indicators" only for new candle
    ta_on_candle = False

    ## Experimental settings (configuration will overide these if set)
    use_exit_signal            = True
    ignore_roi_if_entry_signal = True


    ## Optional order type mapping
    # !! leave these as is since these don't affect the bt
    order_types = {
        'buy'                 : 'limit',
        'sell'                : 'limit',
        'stoploss'            : 'market',
        'stoploss_on_exchange': False
    }

    # |*|
    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        return []

    def populate_indicators(self, dataframe: pd.DataFrame ) -> pd.DataFrame:
        """
        !! row.loc[0] - the candle timestamp
        """
        dataframe["signal"]       = dataframe.progress_apply(lambda row: return_current_signal_as_dict(self.signal_df, row.loc[0]), axis='columns')
        dataframe["base"]         = dataframe.apply(lambda row: row["signal"].get("base"), axis="columns")
        dataframe["above"]        = dataframe.apply(lambda row: row["signal"].get("above"), axis="columns")
        dataframe["below"]        = dataframe.apply(lambda row: row["signal"].get("below"), axis="columns")
        dataframe["indicator"]    = dataframe.apply(lambda row: row["signal"].get("indicator"), axis="columns")
        dataframe["M_dt"]         = dataframe.apply(lambda row: row["signal"].get("M_dt"), axis="columns")
        dataframe["Mid"]          = dataframe.apply(lambda row: row["signal"].get("Mid"), axis="columns")

        return dataframe

    # |*|
    def populate_buy_trend(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: pd.DataFrame
        :return: pd.DataFrame with buy column
        """

        # returns the signals that might be a buy, we will then iterate over this subset 
        # to see which ones are actual buys based on "is_actual_buy" 
        df_copy = dataframe.assign(buy = 0)

        # FILTER FOR THE NOMINAL BUY CONDITION
        # we esentially buy whenever the (base price + buy_buffer) is within low price and high price 
        # (this will go to hyperopt later) 
        # todo - remove unnecessary computation here
        df_nominal_buy = df_copy.loc[
            (df_copy.loc[:,3] <= (df_copy['base'] * self.my_params["buy_buffer"])) & \
            ( (df_copy['base'] * self.my_params["buy_buffer"]) <= df_copy.loc[:,2])
        ]

        # df_copy.assign(buy = lambda row: bt1.is_actual_buy(row.Mid))
        df_copy["buy"] = df_nominal_buy.progress_apply(lambda row: self.bt.is_actual_buy(row.Mid), axis='columns')

        return df_copy


    # |*|
    def populate_sell_trend(self, dataframe: pd.DataFrame ) -> pd.DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: pd.DataFrame
        :return: pd.DataFrame with buy column
        """

        # add a new column "sell" to dataframe, initialize it to 0
        dataframe = dataframe.assign(sell = 0)

        # filter for the nominal sell condition
        # we sell whenever the either above or below price is within the low and high of the candle
        df_nominal_sell = dataframe.loc[
            (dataframe['above'] * self.my_params["sell_buffer"]).between(dataframe.loc[:,3], dataframe.loc[:,2]) |
            (dataframe['below'] * self.my_params["sell_buffer"]).between(dataframe.loc[:,3], dataframe.loc[:,2])
        ]
        
        dataframe["sell"] = df_nominal_sell.progress_apply(lambda row: self.bt.is_actual_sell(row.Mid), axis='columns')
        return dataframe

    def number_of_signal_never_bought(self) -> int:
        """
        Returns the number of signals that have never been bought
        :return: int
        """
        # todo - test this function
        # get the set of all signals
        all_signals = set(self.signal_df.Mid)
        # get the set of all bought signals
        bought_signals = self.bt.buys_set
        # get the set of all signals that have never been bought
        never_bought_signals = all_signals - bought_signals
        # return the number of signals that have never been bought
        return len(never_bought_signals)

    def number_of_signal_never_sold(self) -> int:
        """
        gets the length of difference between self.bt.sells_set and self.bt.buys_set
        :return: int
        """
        # todo - test this function
        return len(self.bt.buys_set - self.bt.sells_set)

    def display_results(self):
        min_indicator = self.my_params["min_indicator"]

        print("======THE BACKTEST OF IS OVER======")
        print(f"The number of signals above {min_indicator} is {len(self.signal_df_indicator_filtered)}")
        # todo - sort the rest of these out 
        print(f"The number of total buys and sells is {len(df_sells)}")
        print(f"The number of signals never bought is {number_of_signal_never_bought(signal_df_indicator_filtered, bt1)}")
        print(f"The number of signals never sold is {number_of_signal_never_sold(bt1)}")
        print("=====================================")


