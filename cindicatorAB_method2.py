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

class InformativeSample(IStrategy):
    # So all the methods below are called only once per backtest and each time during the bot loop for 
    # live trading and dry run modes
    # keep this in mind..
    """
    Sample strategy implementing Informative Pairs - compares stake_currency with USDT.
    Not performing very well - but should serve as an example how to use a referential pair against USDT.
    author@: xmatthias
    github@: https://github.com/freqtrade/freqtrade-strategies

    How to use it?
    > python3 freqtrade -s InformativeSample
    """

    # An informative dictionary mapping ticker df column to their meanings - 2022-06-30 21:07
    ticker_df_column_ref = {
        "timestamp": 0,
        "open"     : 1,
        "high"     : 2,
        "low"      : 3,
        "close"    : 4,
        "volume"   : 5
    }

    # we import our cindictor AB sorted df here for use in populate_indicators
    signals = pd.read_json("CND_AB_parsed_fix1.json")

    # create a reversed copy that will be modified later
    signals_df_C = signals[::-1].reset_index(drop=True)

    # parameters that will go to hyperopt
    my_params = {
        "buy_buffer" : 1.01,
        "sell_buffer": 1.01
        # these two will be directly multiplied with the raw signal price
     }

    def first_is_recent_or_eq(self, ts1, ts2, use_arrow=True):
        """
        compares two unix timestamps, if the timetamps might be of different units, 
        keep use_arrow unchanged
        """
        if use_arrow:
            return arrow.get(ts1) >= arrow.get(ts2)
        return ts1 >= ts2

   
    def return_current_signal_as_dict(self, candle_T):
        """
         self.signals_df_C  - dataframe that contains parsed signal values indexed by datetime 
        candle_T    - unix timestamp of the current candle start time 
        !!WARNING   - assumes candle_T is the STARTING timestamp of a given candle  
        TODO - CHECK ASSUMPTION ABOVE
        """
        # METHOD 2 | 2022-07-01 16:07 -
        # !! Unlike the last one, this requires iteration, so will be instersting to see which one is
        # !! faster
        # 1- start from last candle last signal, go over the signals to find the earliest one
        #   that is later than the candle itself. 
        # 2- assign signals' values to that of the candle 
        # 3- If that is not the last signal, delete from df.  
        # This way we'll always be assigning the data from the last signal, and avoid unnecessary comparisons 


        index_later = []
        # goes from latest to earliest, 
        for index, row in self.signals_df_C.iterrows():
            # if the signal is later or eq to the candle timestamp
            if self.first_is_recent_or_eq(row["M_dt"], candle_T, use_arrow=False):
                index_later.append(index)
            # if the signal is earlier than the candle timestamp
            else: 
                break
        
        # check index_later to pick the last index, and delete signal from all other indices
        cols = ['base', 'above', 'below',"indicator", "M_dt"]
        last_dict = self.signals_df_C.loc(index_later[-1], cols).to_dict()

        # delete the rest if any 
        self.signals_df_C.drop(index_later[:-1])

        return last_dict


    
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

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Adds several different TA indicators to the given pd.DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        !! - when binance data is converted from json to df, row.loc[0] inside lambda expression should
        !! refer to the candle timestamp
        """

        # We have two apporaches here so comment them as you need
        # !! signals exists as a global variable here CONSIDER PASSING IT AS ARGUMENT IF THIS VERSION FAILS

        ## dataframe['ema20'] = ta.EMA(dataframe, timeperiod=20)
        ## dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        ## dataframe['ema100'] = ta.EMA(dataframe, timeperiod=100)

        # APPROACH 1 new columns at once
        # new_cols = ["base","above","below","indicator", "M_dt"]

        # # inserts new empty columns to the price dataframe with NaN values
        # dataframe = dataframe.reindex(columns = dataframe.columns.tolist() + new_cols)

        # # populate empty columns from signals dataframe using insert_signal_cols()
        # dataframe = dataframe.apply(lambda row: return_current_signal_as_dict(row.loc[0]), axis='columns', result_type='expand')

        # approach 2 - get the dict and populate / expand one by one 
        dataframe["signal"]    = ""
        for index, row in dataframe.iterrows():
            row["signal"] = self.return_current_signal_as_dict(row.loc[0])
        # dataframe["signal"]    = dataframe.apply(lambda row: self.return_current_signal_as_dict(row.loc[0]), axis='columns')
        dataframe["base"]      = dataframe.apply(lambda row: row.signal["base"], axis="columns")
        dataframe["above"]     = dataframe.apply(lambda row: row.signal["above"], axis="columns")
        dataframe["below"]     = dataframe.apply(lambda row: row.signal["below"], axis="columns")
        dataframe["indicator"] = dataframe.apply(lambda row: row.signal["indicator"], axis="columns")
        dataframe["M_dt"]      = dataframe.apply(lambda row: row.signal["M_dt"], axis="columns")

        return dataframe

    # |*|
    def populate_buy_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: pd.DataFrame
        :return: pd.DataFrame with buy column
        """
        
        # we esentially buy whenever the (base price + buy_buffer) is within low price and high price 
        # (this will go to hyperopt later) 
        dataframe.loc[
            (
                # (dataframe['low_15m'] <= (dataframe['base'] * self.my_params["buy_buffer"])  <= dataframe['high_15m'])

                # in case reference needed, check deprecated line above. 
                (dataframe.loc[:,3] <= (dataframe['base'] * self.my_params["buy_buffer"])  <= dataframe.loc[:,2])
            ),
            'buy'] = 1

        return dataframe

    # |*|
    def populate_sell_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: pd.DataFrame
        :return: pd.DataFrame with buy column
        """

        # we sell whenever the either above or below price is within the low and high of the candle
        dataframe.loc[
            (
                # (dataframe['low_15m'] <= (dataframe['above'] * self.my_params["sell_buffer"])  <= dataframe['high_15m']) |
                # (dataframe['low_15m'] <= (dataframe['below'] * self.my_params["sell_buffer"])  <= dataframe['high_15m'])

                # in case reference needed, check deprecated line above. 
                (dataframe.loc[:,3] <= (dataframe['above'] * self.my_params["sell_buffer"])  <= dataframe.loc[:,2]) |
                (dataframe.loc[:,3] <= (dataframe['below'] * self.my_params["sell_buffer"])  <= dataframe.loc[:,2])
            ),
            'sell'] = 1
        return dataframe
