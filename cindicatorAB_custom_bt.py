# Decided to add a short signal, since freqtrade doesn't do that,
# I'll be overriding the default one with this one 
# This file is the copy of cindicatorAB_method_2.py at commit 400642f91ab043db383b9807cfa5745e713aa2f8
# 2022-07-07 17:38

## --- !! UNCOMMENT ONLY IN CASE OF ERROR ---
# from ctypes.wintypes import BOOL
# from freqtrade.strategy import IStrategy, merge_informative_pair
# from functools import reduce

# my imports / un-imports here
from typing import Dict, List
import pandas as pd
import arrow
import numpy as np
import pickle
import os
import re
from contextlib import redirect_stdout
from tqdm import tqdm
from pprint import pprint
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
    Keeps the trade object that will be stored by class bt_helper()
    Initialized with the signal object of the signal the trade was opened upon
    Trade status will be kept by class bt_helper()
    """
    # ticker will inherit from backtest.my_params["ticker"]

    def __init__(self, signal, candle, my_params):
        # todo - atts to add
        # - entry candle as dictionary with keys "high", "low", "open", "close", "timestamp"
        # - exit candle as dictionary with keys "high", "low", "open", "close", "timestamp"

        # self.my_params       = my_params # todo - set this as class variable
        self.entry_index     = candle.name # todo - will be deprecated 
        # self.entry_candle will contain the first 6 attributes of the candle with keys 0-5
        self.entry_candle    = {
            "TS": candle[0],
            "O" : candle[1],
            "H" : candle[2],
            "L" : candle[3],
            "C" : candle[4],
            "V" : candle[5]
        }
        self.exit_candle     = None
        self.signal          = signal
        self.signal_id       = signal["Mid"]
        self.indicator       = signal["indicator"]
        # todo - avoid 2nd application of effective_price() by just calling row.above etc. 
        self.effective_above = effective_price(signal["above"])
        self.effective_below = effective_price(signal["below"])
        self.effective_entry = effective_price(signal["base"])

        self.type             = None
        if   self.indicator  >= my_params["min_indicator"]:
             self.type        = "L"
        elif self.indicator  <= my_params["max_indicator"]:
             self.type        = "S"
        assert self.type is not None, "long / short type is None"

        # these will be filled upon exit
        # self.is_above       = None
        # self.is_win         = (True if ((self.is_above and self.type == "L") or (not self.is_above and self.type == "S")) else False)
        self.exit_index         = None
        self.exit_candle        = None
        self.is_win             = None
        self.exit_price         = None
        self.profit_percent     = None
        self.elapsed_hours      = None
        
    def set_exit_attributes(self, exit_candle, is_above: bool) -> None:
        """
        Sets the exit attributes of the trade   
        """
        self.exit_index         = exit_candle.name
        self.exit_candle        = {
            "TS": exit_candle[0],
            "O" : exit_candle[1],
            "H" : exit_candle[2],
            "L" : exit_candle[3],
            "C" : exit_candle[4],
            "V" : exit_candle[5]
        }
        # note that this is in ms hence 3600000 = 1 hour
        self.elapsed_hours = (self.exit_candle["TS"] - self.entry_candle["TS"]) / 3600000

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

class step():
    """
    Stores step objects the will be used by class bt_helper for creating final bactest results 
    Could have been a dict but you know, whatever. | 2022-07-24 19:31
    """
    def __init__(self, cutoff, trades, wins, Rwin, cum_ret) -> None:
        self.cutoff  = cutoff
        self.trades  = trades
        self.wins    = wins
        self.Rwin    = Rwin
        self.cum_ret = cum_ret

class bt_helper():
    """
    This is our custom backtest class, this needs to be initialized inside the strategy class | 2022-07-06 20:49
    """

    def __init__(self, my_params, local_mode=True) -> None:
        # self.local_mode             = local_mode
        self.my_params              = my_params
        self.last_entry_signal_id   = None
        self.last_exit_signal_id    = None
        self.buys_set               = set()   # todo - these will be deprecated
        self.sells_set              = set()   # todo - these will be deprecated
        self.open_trades            = [] # at the end of bt_helper, we'll check this to see if any trades were left open
        self.closed_trades          = [] # at the end of bt_helper, will be turned into a dataframe to give us the bt_helper results
        self.df_closed              = None # will store the df that is turned from self.closed_trades where we set the bactest is over
        # "steps" is an alias for indicator values
        self.df_long_steps          = None
        self.df_short_steps         = None
        self.best_indicators        = None
        self.best_indicators_df     = None
        self.best_L_ind_totwin      = None
        self.best_L_ind_Rwin        = None
        self.best_S_ind_totwin      = None
        self.best_S_ind_Rwin        = None
        self.cumret_totwin_best_ind = None
        self.cumret_Rwin_best_ind   = None

    def is_actual_buy_sell(self, row : Dict) -> bool:
        """
        Combined version of is_actual_buy and is_actual_sell | 2022-07-19 18:59
        Returns 1 if signal is a buy, 2 if a sell, 0 if neither
        """
        current_signal_dict = row.signal
        ticker_index        = row.name
        current_signal_id   = current_signal_dict["Mid"]

        # FOR BUYS - check if the signal is a new one
        # in case new signal, process for buy conditions
        is_new_signal = (self.last_entry_signal_id != current_signal_id)
        if is_new_signal:

            # check for prices before doing the entry
            # 2 is for high and 3 is for low
            if row.base < row.loc[2] and row.base > row.loc[3]:
            
                self.last_entry_signal_id = current_signal_id
                # add the current signal to the set of bought signals
                self.buys_set.add(current_signal_id)

                # create the trade object with the current signal
                # trade_obj = trade(current_signal_dict, ticker_index, self.my_params)
                trade_obj = trade(current_signal_dict, row, self.my_params)
                self.open_trades.append(trade_obj)

                return 1
        # pass since others will be pass as well

        # might be a sell signal
        else:            
            # check if it is the last bought signal
            has_been_bought = (self.last_entry_signal_id == current_signal_id)

            # check if the signal has not been sold before
            new_sell_signal = (self.last_exit_signal_id != current_signal_id)

            if (new_sell_signal and has_been_bought): 

                # 2 is for high and 3 is for low
                is_above_exit = None
                if row.above < row.loc[2] and row.above > row.loc[3]: 
                    is_above_exit = True
                elif row.below < row.loc[2] and row.below > row.loc[3]: 
                    is_above_exit = False
                # the candle doesn't touch exit prices as well, terminate 
                else: 
                    return 0

                # if not returned 0 yet means one of the sell conditions has been met
                self.last_exit_signal_id = current_signal_id

                # remove the last item from open_trades and append to closed_trades
                last_trade = self.open_trades.pop()
                # last_trade.set_exit_attributes(ticker_index, is_above_exit)
                last_trade.set_exit_attributes(row, is_above_exit)
                self.closed_trades.append(last_trade)

                return 2
            # in case it hasn't yet been bought or already sold, terminate
            else:
                return 0

    def get_best_indicator(self, df_results, max_colname: str):
        """
        Returns column of the max_colname that needs to be maximized as a dictionary, if 
        there are multiple such rows, returns the range of the max_colname
        !! the reason we include this inside the class is for the sake of readability
        """
        # df_results = df_results.loc[df_results[max_colname] == df_results[max_colname].max()]
        max_df     = df_results.loc[df_results[max_colname] == df_results[max_colname].max()]

        # this will give us the values of the loosest indicator
        ref_row = max_df.iloc[-1]
        range = ref_row.cutoff
        # range = None
        # if max_df.shape[0] == 1:
        #     # get value of the indicator column of df_results that has the maximum value of max_colname
        #     range = ref_row.cutoff
        # else:
        #     # return a tuple of the min and max values of the indicator column of df_results
        #     range = (max_df.loc[max_df.cutoff.idxmin()].cutoff, max_df.loc[max_df.cutoff.idxmax()].cutoff)
        dict = {
            "best_indicator": range,
            "trades"        : ref_row.trades,
            "wins"          : ref_row.wins,
            "Rwin"          : ref_row.Rwin,
            "cum_ret"       : ref_row.cum_ret
        }
        return dict

    def get_best_indicators(self):
        """
        Returns a dict of 4 dictionaries for indicator rows that maximize Rwin and cumret in both df_long_steps and df_short_steps
        Called by set_full_exit_df()
        """
        dict_long_rwin         = self.get_best_indicator(self.df_long_steps, "Rwin")
        self.best_L_ind_Rwin   = dict_long_rwin["best_indicator"]
        dict_long_cumret       = self.get_best_indicator(self.df_long_steps, "cum_ret")
        self.best_L_ind_totwin = dict_long_cumret["best_indicator"]
        dict_short_rwin        = self.get_best_indicator(self.df_short_steps, "Rwin")
        self.best_S_ind_Rwin   = dict_short_rwin["best_indicator"]
        dict_short_cumret      = self.get_best_indicator(self.df_short_steps, "cum_ret")
        self.best_S_ind_totwin = dict_short_cumret["best_indicator"]

        self.cumret_totwin_best_ind = round((1 + dict_long_cumret["cum_ret"]) * (1 + dict_short_cumret["cum_ret"]) - 1, 4)
        self.cumret_Rwin_best_ind   = round(( 1+ dict_long_rwin["cum_ret"]) * (1 + dict_short_rwin["cum_ret"]) - 1, 4)

        dict = {
            "long_rwin"   : dict_long_rwin,
            "long_cumret" : dict_long_cumret,
            "short_rwin"  : dict_short_rwin,
            "short_cumret": dict_short_cumret
        }
        self.best_indicators = dict
        self.best_indicators_df = pd.DataFrame(dict).T

    def construct_trade_df(self):
        """
        # construct a df that will contain rows as trades inside bt_helper.closed_trades with attributes of 
        # signal.M_dt, signal.above, signal.below, signal.indicator, effective_entry, exit_price,
        # entry_index, exit_index
        # also, from ticker_df, finds rows with index entry_index, exit_index and adds column number 0, 2 and 3 for both
        """
        col_list = [
            "above", "below", "indicator", "entry_price", "exit_price", "M_dt", "base",
            "candle1_dt", "candle2_dt", "high1", "low1", "high2", "low2", "is_win", "elapsed_hours", "elapsed_days", "%profit"
            ] 
        trade_df = pd.DataFrame(columns=col_list)
        for trade in self.closed_trades:
            row_dict = {
                "base"          : trade.signal["base"],
                "above"         : trade.signal["above"],
                "below"         : trade.signal["below"],
                "indicator"     : trade.signal["indicator"],
                "entry_price"   : trade.effective_entry,
                "exit_price"    : trade.exit_price,
                "is_win"        : int(trade.is_win),
                "%profit"       : round(trade.profit_percent, 3),
                "candle1_dt"    : arrow.get(int(trade.entry_candle["TS"])).format("YYYY-MM-DD HH:mm:ss"),
                "candle2_dt"    : arrow.get(int(trade.exit_candle["TS"])).format("YYYY-MM-DD HH:mm:ss"),
                "high1"         : trade.entry_candle["H"],
                "low1"          : trade.entry_candle["L"],
                "high2"         : trade.exit_candle["H"],
                "low2"          : trade.exit_candle["L"],
                "elapsed_hours" : trade.elapsed_hours,
                "elapsed_days"  : round(trade.elapsed_hours / 24, 2)
                }
            
            row_dict["M_dt"] = arrow.get(int(trade.signal["M_dt"])).format("YYYY-MM-DD HH:mm:ss")
            trade_df = pd.concat([trade_df, pd.DataFrame(row_dict, index=[0])], ignore_index=True)
        # change thge order of columns as "indicator", "M_dt",	"candle1_dt", "candle2_dt", "high1", "low1", "entry_price", "high2", "low2", "exit_price", "above", "below"
        trade_df = trade_df[["indicator", "is_win", "%profit", "M_dt", "candle1_dt", "candle2_dt", "low1", "high1", "entry_price", "base", "low2", "high2", "exit_price", "above", "below", "elapsed_hours", "elapsed_days"]]
        # reorder with column M_dt as increasing
        trade_df = trade_df.sort_values(by=["M_dt"])
        # turn column is_win into int
        # trade_df["elapsed_days"] = trade_df["elapsed_hours"] / 24
        trade_df["is_win"] = trade_df["is_win"].astype(int)
        self.df_closed = trade_df

    def set_full_exit_df(self):
        """
        Assings df_long_steps and df_short_steps which contain the sliced indicator step objects 
        for both long and short trades, so that the optimal cobination of long and short indicators can be analyzed | 2022-07-24 19:30
        The chosen columns for the df are given by cols_list
        """
        cols_list = ["type", "indicator", "is_win", "profit_percent", "signal_id"]

        # turns self.closed_trades (a list) into a pandas df
        # self.df_closed = pd.DataFrame([vars(s) for s in self.closed_trades], columns=cols_list)
        self.construct_trade_df()
        closed = self.df_closed

        # loop for longs
        long_steps = []
        long_steps_loop = np.arange(92, 79.5, -0.5)

        for i, s in enumerate(long_steps_loop):
            # select all trades in df_closed with indicator above i
            this_slice = closed[(closed.indicator >= s)]

            # calculate # of trades (number of rows)
            num_of_trades = this_slice.shape[0]

            wins   = None
            r_wins = None
            cumret = None
            # if above is 0 assign 0 to wins, r_wins, cumret
            if num_of_trades == 0:
                wins   = 0
                r_wins = 0
                cumret = 0
            else: 
                # calculate # of wins 
                wins = this_slice[this_slice.is_win == True].shape[0]

                # calculate wins / total
                r_wins = wins / num_of_trades

                # calculate cumulative profit
                # todo - make sure it is the correct form
                # cum_rets = np.cumprod(1 + this_slice['profit_percent'].values) - 1
                cum_rets = np.cumprod(1 + this_slice['%profit'].values) - 1
                cumret = cum_rets[-1]

            # add the step to the list of steps
            long_steps.append(step(s, num_of_trades, wins, r_wins, cumret))

        # create a df from steps and assign to self.df_long_steps
        self.df_long_steps = pd.DataFrame([vars(s) for s in long_steps])#, columns=["cutoff", "num_trades", "num_wins", "Rwin", "cum_ret"])

        # loop for shorts
        short_steps = []
        short_steps_loop = np.arange(8, 20.5, 0.5)

        for i, s in enumerate(short_steps_loop):
            # select all trades in df_closed with indicator below i
            this_slice = closed[(closed.indicator <= s)]

            # calculate # of trades (number of rows)
            num_of_trades = this_slice.shape[0]

            wins   = None
            r_wins = None
            cumret = None
            # if num_of_trades is 0 assign 0 to wins, r_wins, cumret
            if num_of_trades == 0:
                wins = 0
                r_wins = 0
                cumret = 0
            else:
                # calculate # of wins 
                wins = this_slice[this_slice.is_win == True].shape[0]

                # calculate wins / total
                r_wins = wins / num_of_trades

                # calculate cumulative profit
                # todo - make sure it is the correct form
                # cum_rets = np.cumprod(1 + this_slice['profit_percent'].values) - 1
                cum_rets = np.cumprod(1 + this_slice['%profit'].values) - 1
                cumret = cum_rets[-1]

            # add the step to the list of steps
            short_steps.append(step(s, num_of_trades, wins, r_wins, cumret))

        # create a df from steps and assign to self.df_long_steps
        self.df_short_steps = pd.DataFrame([vars(s) for s in short_steps])#, columns=["cutoff", "num_trades", "num_wins", "Rwin", "cum_ret"])

        self.get_best_indicators()

# class cindicatorAB_longshort(IStrategy):
class backtest():
    # So all the methods below are called only once per backtest and each time during the bot loop for 
    # live trading and dry run modes
    # keep this in mind..

    # def __init__(self, signal_file, ticker_file, my_params: Dict) -> None:
    def __init__(self, my_params: Dict, signal_df=None, local_mode=True) -> None:
        """
        Set local_mode to False if runnin the bactest over other tickers as well
        """
        self.local_mode  = local_mode 
        self.signal_file = my_params["signal_file"] #"/home/u237/projects/parsing_cindicator/data/CND_AB_parsed_fix1.json"
        self.ticker_file = my_params["ticker_file"] #"/home/u237/projects/backtests/cindicator-bt_helper1/ft_userdata/user_data/data/binance_old/ZEC_USDT-1h.json"
        self.my_params   = my_params
        self.bt_helper   = bt_helper(my_params)

        # we import our cindictor AB sorted df here for use in populate_indicators, also the ticker df
        self.signals   = signal_df
        if local_mode:
            self.signals   = pd.read_json(self.signal_file)
        else:
            self.signals   = signal_df
        self.ticker_df = pd.read_json(self.ticker_file)

        # filter signals df for only our TICKER and those with indicator greater than min_indicator 
        # todo - the indicator filtering will be done for a second time in case of global mode, insert an if statement for this
        self.signal_df_indicator_filtered = self.signals.loc[((self.signals.ticker == my_params["ticker"]) & ~(self.signals.indicator.between(my_params["max_indicator"], my_params["min_indicator"], inclusive='neither'))), ['base', 'above', 'below',"indicator", "M_dt", "Mid"]]
        self.signal_df                    = self.signal_df_indicator_filtered.copy()
        self.tot_filtered                 = self.signal_df_indicator_filtered.shape[0]

        # attributes to be filled at the end of the bactest
        self.tot_bought_never_sold  = None # will be len(self.bt_helper.open_trades)
        self.tot_never_bought       = None # will be self.tot_filtered - (self.tot_sells + self.tot_bought_never_sold)
        self.tot_sells              = None # will be self.bt_helper.df_closed.shape[0]


        # finally run the backtest
        self.run_backtest()
    
    def run_backtest(self):
        """
        Run the sequence of correct method calls as a final step in the constructor | 2022-07-27 17:32
        """
        self.populate_indicators()
        self.populate_buy_sell()
        self.bt_helper.set_full_exit_df()
        
        self.tot_bought_never_sold = len(self.bt_helper.open_trades)
        self.tot_sells             = self.bt_helper.df_closed.shape[0]
        self.tot_never_bought      = self.tot_filtered - (self.tot_sells + self.tot_bought_never_sold)

        if self.local_mode:
            self.display_results()


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
        # self.ticker_df["base"]         = self.ticker_df.apply(lambda row: row["signal"].get("base"), axis="columns")
        # Added this after applying effective_price to rows with empty signals has produced an error 
        # From self.ticker_df drop all rows where column signal is {}
        # !! make sure no problems with new index
        self.ticker_df = self.ticker_df.loc[self.ticker_df["signal"] != {}]
        self.ticker_df["base"]         = self.ticker_df.apply(lambda row: effective_price(row["signal"].get("base")), axis="columns")
        self.ticker_df["above"]        = self.ticker_df.apply(lambda row: effective_price(row["signal"].get("above")), axis="columns")
        self.ticker_df["below"]        = self.ticker_df.apply(lambda row: effective_price(row["signal"].get("below")), axis="columns")
        self.ticker_df["indicator"]    = self.ticker_df.apply(lambda row: row["signal"].get("indicator"), axis="columns")
        self.ticker_df["M_dt"]         = self.ticker_df.apply(lambda row: row["signal"].get("M_dt"), axis="columns")
        self.ticker_df["Mid"]          = self.ticker_df.apply(lambda row: row["signal"].get("Mid"), axis="columns")

    # |*|
    def populate_buy_sell(self) -> pd.DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: pd.DataFrame
        :return: pd.DataFrame with buy column
        """
        # !! buy_sell start
        # returns the signals that might be a buy, we will then iterate over this subset 
        # to see which ones are actual buys based on "is_actual_buy" 
        df_copy = self.ticker_df
        df_copy.assign(buy1_or_sell2 = 0)

        # FILTER FOR THE NOMINAL BUY CONDITION
        # we esentially buy whenever the (base price + trade_buffer) is within low price and high price 
        # (this will go to hyperopt later) 
        df_nominal_buy_sell = df_copy.loc[
            # TODO - PROBLEM - Entry and exit prices don't seem to be within high and low bounds, let's make that sure
            # filters rows that have base (entry) price within open and close
            (df_copy['base']).between(df_copy.loc[:,3], df_copy.loc[:,2], inclusive='neither') | 
            # filters rows that have above price within open and close
            (df_copy['above']).between(df_copy.loc[:,3], df_copy.loc[:,2], inclusive='neither') |
            # filters rows that have below price within open and close
            (df_copy['below']).between(df_copy.loc[:,3], df_copy.loc[:,2], inclusive='neither')
        ]

        df_copy["buy1_or_sell2"] = df_nominal_buy_sell.apply(lambda row: self.bt_helper.is_actual_buy_sell(row), axis='columns')
        # dataframe["sell"] = df_nominal_sell.progress_apply(lambda row: self.bt_helper.is_actual_sell(row), axis='columns')
    
    def display_results(self):
        """
        Displays the results of interest after setting the final attributes of self.bt_helper 
        """
        min_indicator         = self.my_params["min_indicator"]
        max_indicator         = self.my_params["max_indicator"]

        # call set_full_exit_df on the bt_helper
        df_closed = self.bt_helper.df_closed

        print("================================THE BACKTEST OF IS OVER================================")

        day0 = df_closed.M_dt.iloc[0]
        dayLast = df_closed.M_dt.iloc[-1]
        print(f"Showing results for backtest between times of: \n{day0}\n{dayLast}")
        arL = arrow.get(dayLast)
        ar0 = arrow.get(day0)
        print(f"{ar0.humanize(arL, granularity=['year', 'day'], only_distance=True)}\n")

        print("Displaying Parameters:")
        # pprint(self.my_params, indent=4)
        [print(f"   {x}: {self.my_params[x]}") for x in self.my_params.keys()]
        print()
        print(f"The number of signals outside range ({min_indicator}, {max_indicator}) - {self.tot_filtered}")
        print(f"The number of total sells                    - {self.tot_sells}")
        print(f"The number of signals never sold             - {self.tot_bought_never_sold}") # all signals that were sold were popped from here
        print(f"The number of signals never bought           - {self.tot_never_bought}")
        print( "\nThe df of top indicator values for shorts and longs:\n")
        print(self.bt_helper.best_indicators_df.to_markdown()) 
        print()
        print(f"Total return from best inficators that maximize total wins: \n=======>{self.bt_helper.cumret_totwin_best_ind}<=======")
        print(f"Total return from best inficators that maximize ratio of wins to losses: \n=======>{self.bt_helper.cumret_Rwin_best_ind}<=======")

        # print( "\nThe df of long trades above the indicator that maximizes cumulateive return:\n")
        # print( "\nThe df of short trades below the indicator that maximizes cumulateive return:\n")
        print("=======================================================================================")

class bt_top_N():
    """
    The class that will be running the backtest on all tickers given by the list top_N 
    Will be calling backtest() N times
    todo - try indicator-filtering and grouping all signals acc to ticker, and then passing signal_df to individual backtests accdingly
    """
    def __init__(self, my_params: dict) -> None:
        self.N_tickers = my_params["N_tickers"]
        self.my_params = my_params
        self.signal_df = pd.read_json(my_params["signal_file"])
        
        # getting top N tickers as a series object to iterate in the loop

        # ===================TO BE SET AFTER BACKTEST===================
        
        self.tot_signal_df_filtered = None # todo - will be set to the sum of backtest.self.tot_filtered
        self.tot_bought_never_sold  = None
        self.tot_never_bought       = None
        self.tot_sells              = None # will be the length of df_closed_topN
        self.df_long_steps          = None
        self.df_short_steps         = None
        self.best_indicators        = None
        self.best_indicators_df     = None
        self.best_L_ind_totwin      = None
        self.best_L_ind_Rwin        = None
        self.best_S_ind_totwin      = None
        self.best_S_ind_Rwin        = None
        self.cumret_totwin_best_ind = None
        self.cumret_Rwin_best_ind   = None

        # todo -turn last 6 attributes above into dictionary
        # self.backtest_results = {
        #     "best_L_ind_totwin"     : self.best_L_ind_totwin,
        #     "best_L_ind_Rwin"       : self.best_L_ind_Rwin,
        #     "best_S_ind_totwin"     : self.best_S_ind_totwin,
        #     "best_S_ind_Rwin"       : self.best_S_ind_Rwin,
        #     "cumret_totwin_best_ind": self.cumret_totwin_best_ind,
        #     "cumret_Rwin_best_ind"  : self.cumret_Rwin_best_ind,
        # }

        col_names                   = [
            # !! Have decided that "ticker" wont be a column but a second level index instead"
            'indicator', 'is_win', '%profit', 'M_dt', 'candle1_dt', 
            'candle2_dt', 'low1', 'high1', 'entry_price', 'base', 'low2', 'high2', 
            'exit_price', 'above', 'below', 'elapsed_hours', 'elapsed_days'
        ]

        # initalize the empty df trades that each loop is going to append rows to
        self.df_closed_topN = pd.DataFrame(columns=col_names)

        # =====================RUNNING THE BACKTEST=====================

        self.run_backtest()
        self.display_results_top_N()

        # open new file with path self.my_params["output_file_dir"] if it doesnt exist
        if not os.path.exists(self.my_params["output_file_dir"]):
            os.makedirs(self.my_params["output_file_dir"])

        # save self.df_closed_topN, self.df_long_steps, self.df_short_steps, self.best_indicators_df to self.my_params["output_file_dir"] as json
        self.df_closed_topN.to_csv(self.my_params["output_file_dir"] + "df_closed_topN.json")
        self.df_long_steps.to_json(self.my_params["output_file_dir"] + "df_long_steps.json")
        self.df_short_steps.to_json(self.my_params["output_file_dir"] + "df_short_steps.json")
        self.best_indicators_df.to_json(self.my_params["output_file_dir"] + "best_indicators_df.json")

    def get_ticker_file(self, ticker):
        # todo -use the glopbal version insteed
        path       = self.my_params["ticker_file_dir"]
        time_frame = self.my_params["time_frame"]
        ticker     = ticker.replace("/", "_")

        return f"{path}{ticker}-{time_frame}.json"

    def run_backtest(self):
        cum_signal_df_filtered    = 0
        cum_tot_bought_never_sold = 0
        cum_tot_never_bought      = 0
        for i, ticker in enumerate(self.N_tickers, start=1): 
            # correct all USD to USDT
            backtest_param_dict                     = self.my_params.copy()
            backtest_param_dict["ticker"]           = ticker
            if ticker.endswith("USD"):
                ticker = ticker + "T"
            ticker_file                             = self.get_ticker_file(ticker)

            # check if the ticker file exists
            if not os.path.exists(ticker_file):
                print(f"File for {ticker} does not exist. Skipping...")
                # self.my_params["N_tickers"].remove(ticker)
                # self.my_params["N"] -= 1
                continue
            backtest_param_dict["ticker_file"]      = ticker_file
            
            # !! everything happens here
            print(f"{i} - RUNNING BACKTEST FOR {ticker}")
            bt                         = backtest(backtest_param_dict, self.signal_df, local_mode=False)
            df_ticker                  = bt.bt_helper.df_closed

            cum_signal_df_filtered    += bt.tot_filtered
            cum_tot_bought_never_sold += bt.tot_bought_never_sold
            cum_tot_never_bought      += bt.tot_never_bought
            # df_ticker["ticker"]        = ticker

            # add the ticker as a second level index
            # todo - make sure this is the right place to do this
            pd.concat([df_ticker], keys=[ticker], names=['ticker'])
            # append df_ticker to df_tickers
            self.df_closed_topN = pd.concat([self.df_closed_topN, df_ticker])

        self.tot_signal_df_filtered = cum_signal_df_filtered
        self.tot_bought_never_sold  = cum_tot_bought_never_sold
        self.tot_never_bought       = cum_tot_never_bought
        self.tot_sells              = self.df_closed_topN.shape[0]

        # here we run the global version of set_full_exit_df
        self.set_full_exit_df()

    def p2f(func):
        def wrapper(self):
            file_dir, file_name = self.my_params["output_file_dir"], self.my_params["output_file"]
            # append "/" to the end of the file_dir if it doesn't have one
            if file_dir[-1] != "/":
                file_dir += "/"
            with open(file_dir + file_name, "w") as f:
                with redirect_stdout(f):
                    func(self)
        return wrapper

    # @p2f
    def display_results_top_N(self):
        """
        Displays the results of interest after setting the final attributes of backtest.bt_helper 
        """
        min_indicator         = self.my_params["min_indicator"]
        max_indicator         = self.my_params["max_indicator"]
        # tot_filtered          = len(backtest.signal_df_indicator_filtered)
        # tot_sells             = len(backtest.bt_helper.closed_trades)
        # tot_bought_never_sold = len(backtest.bt_helper.open_trades)
        # tot_never_bought      = tot_filtered - (tot_sells + tot_bought_never_sold)

        # call set_full_exit_df on the bt_helper
        # todo - will be set to the concatenated version
        df_closed = self.df_closed_topN

        print("================================THE BACKTEST OF IS OVER================================")
        print()

        # todo - sort df_closed acc to entry candles, should be from entry candle of 1st trade to exit candle of the last one
        df_closed_sorted = df_closed.sort_values(by=["M_dt"])
        day0             = df_closed_sorted.M_dt.iloc[0]
        dayLast          = df_closed_sorted.M_dt.iloc[-1]

        print(f"Showing results for backtest between times of: \n{day0}\n{dayLast}")
        arL = arrow.get(dayLast)
        ar0 = arrow.get(day0)
        print(f"{ar0.humanize(arL, granularity=['year', 'day'], only_distance=True)}\n")

        print("Displaying Parameters:")
        # pprint(backtest.my_params, indent=4)
        [print(f"   {x}: {self.my_params[x]}") for x in self.my_params.keys()]
        print()
        # backtest.tot_filtered will be the length of the signal_df filtered at the beginning
        print(f"The number of signals outside range ({min_indicator}, {max_indicator}) - {self.tot_signal_df_filtered}")
        # backtest.tot_sells will be the length of self.df_closed_topN
        print(f"The number of total sells                    - {self.tot_sells}")
        print(f"The number of signals never sold             - {self.tot_bought_never_sold}") # all signals that were sold were popped from here
        print(f"The number of signals never bought           - {self.tot_never_bought}")

        # todo - complete the rest after testing this part sofar 
        print( "\nThe df of top indicator values for shorts and longs:\n")
        print(self.best_indicators_df.to_markdown()) 
        print()
        print(f"Total return from best inficators that maximize total wins: \n=======> {self.cumret_totwin_best_ind} <=======")
        print(f"Total return from best inficators that maximize ratio of wins to losses: \n=======> {self.cumret_Rwin_best_ind} <=======")
        # todo - end
        print( "\nThe df of long indicator values and their performance metrics:\n")
        print(self.df_long_steps.to_markdown()) 
        print()
        print( "\nThe df of short indicator values and their performance metrics:\n")
        print(self.df_short_steps.to_markdown()) 
        print()
        print("=======================================================================================")


    def get_best_indicator(self, df_results, max_colname: str):
        """
        Returns column of the max_colname that needs to be maximized as a dictionary, if 
        there are multiple such rows, returns the range of the max_colname
        !! the reason we include this inside the class is for the sake of readability
        """
        # df_results = df_results.loc[df_results[max_colname] == df_results[max_colname].max()]
        max_df     = df_results.loc[df_results[max_colname] == df_results[max_colname].max()]

        # this will give us the values of the loosest indicator
        ref_row = max_df.iloc[-1]
        range = ref_row.cutoff
        # range = None
        # if max_df.shape[0] == 1:
        #     # get value of the indicator column of df_results that has the maximum value of max_colname
        #     range = ref_row.cutoff
        # else:
        #     # return a tuple of the min and max values of the indicator column of df_results
        #     range = (max_df.loc[max_df.cutoff.idxmin()].cutoff, max_df.loc[max_df.cutoff.idxmax()].cutoff)
        dict = {
            "best_indicator": range,
            "trades"        : ref_row.trades,
            "wins"          : ref_row.wins,
            "Rwin"          : ref_row.Rwin,
            "cum_ret"       : ref_row.cum_ret
        }
        return dict

    def get_best_indicators(self):
        """
        TODO - PARAMETERS NEED TO BE MOVED / DUPLICATED TO THE CLASS
        1- 

        Returns a dict of 4 dictionaries for indicator rows that maximize Rwin and cumret in both df_long_steps and df_short_steps
        Called by set_full_exit_df()
        """
        dict_long_rwin         = self.get_best_indicator(self.df_long_steps, "Rwin")
        self.best_L_ind_Rwin   = dict_long_rwin["best_indicator"]
        dict_long_cumret       = self.get_best_indicator(self.df_long_steps, "cum_ret")
        self.best_L_ind_totwin = dict_long_cumret["best_indicator"]
        dict_short_rwin        = self.get_best_indicator(self.df_short_steps, "Rwin")
        self.best_S_ind_Rwin   = dict_short_rwin["best_indicator"]
        dict_short_cumret      = self.get_best_indicator(self.df_short_steps, "cum_ret")
        self.best_S_ind_totwin = dict_short_cumret["best_indicator"]

        self.cumret_totwin_best_ind = round((1 + dict_long_cumret["cum_ret"]) * (1 + dict_short_cumret["cum_ret"]) - 1, 4)
        self.cumret_Rwin_best_ind   = round(( 1+ dict_long_rwin["cum_ret"]) * (1 + dict_short_rwin["cum_ret"]) - 1, 4)

        dict = {
            "long_rwin"   : dict_long_rwin,
            "long_cumret" : dict_long_cumret,
            "short_rwin"  : dict_short_rwin,
            "short_cumret": dict_short_cumret
        }
        self.best_indicators = dict
        self.best_indicators_df = pd.DataFrame(dict).T

    def set_full_exit_df(self):
        """
        PARAMETERS THAT WERE TO BE MOVED / DUPLICATED TO THE CLASS
        1- *df_closed
        2- *self.df_long_steps
        3- *self.df_short_steps
        Assings df_long_steps and df_short_steps which contain the sliced indicator step objects 
        for both long and short trades, so that the optimal cobination of long and short indicators can be analyzed | 2022-07-24 19:30
        The chosen columns for the df are given by cols_list
        """
        closed = self.df_closed_topN

        # ========================LOOP FOR LONGS========================
        long_steps = []
        long_steps_loop = np.arange(92, 79.5, -0.5)

        for i, s in enumerate(long_steps_loop):
            # select all trades in df_closed with indicator above i
            this_slice = closed[(closed.indicator >= s)]

            # calculate # of trades (number of rows)
            num_of_trades = this_slice.shape[0]

            wins   = None
            r_wins = None
            cumret = None
            # if above is 0 assign 0 to wins, r_wins, cumret
            if num_of_trades == 0:
                wins   = 0
                r_wins = 0
                cumret = 0
            else: 
                # calculate # of wins 
                wins = this_slice[this_slice.is_win == True].shape[0]

                # calculate wins / total
                r_wins = wins / num_of_trades

                # calculate cumulative profit
                # todo - make sure it is the correct form
                # cum_rets = np.cumprod(1 + this_slice['profit_percent'].values) - 1
                cum_rets = np.cumprod(1 + this_slice['%profit'].values) - 1
                cumret = cum_rets[-1]

            # add the step to the list of steps
            long_steps.append(step(s, num_of_trades, wins, r_wins, cumret))

        # create a df from steps and assign to self.df_long_steps
        self.df_long_steps = pd.DataFrame([vars(s) for s in long_steps])#, columns=["cutoff", "num_trades", "num_wins", "Rwin", "cum_ret"])

        # ========================LOOP FOR SHORTS========================
        short_steps = []
        short_steps_loop = np.arange(8, 20.5, 0.5)

        for i, s in enumerate(short_steps_loop):
            # select all trades in df_closed with indicator below i
            this_slice = closed[(closed.indicator <= s)]

            # calculate # of trades (number of rows)
            num_of_trades = this_slice.shape[0]

            wins   = None
            r_wins = None
            cumret = None
            # if num_of_trades is 0 assign 0 to wins, r_wins, cumret
            if num_of_trades == 0:
                wins = 0
                r_wins = 0
                cumret = 0
            else:
                # calculate # of wins 
                wins = this_slice[this_slice.is_win == True].shape[0]

                # calculate wins / total
                r_wins = wins / num_of_trades

                # calculate cumulative profit
                # todo - make sure it is the correct form
                # cum_rets = np.cumprod(1 + this_slice['profit_percent'].values) - 1
                cum_rets = np.cumprod(1 + this_slice['%profit'].values) - 1
                cumret = cum_rets[-1]

            # add the step to the list of steps
            short_steps.append(step(s, num_of_trades, wins, r_wins, cumret))

        # create a df from steps and assign to self.df_long_steps
        self.df_short_steps = pd.DataFrame([vars(s) for s in short_steps])#, columns=["cutoff", "num_trades", "num_wins", "Rwin", "cum_ret"])

        self.get_best_indicators()

class set_results():
    """
    This class will be used for constructing result object for saving and displaying 
    backtest results whichever class is being run (backtest or bt_topN) | 2022-07-30 23:55
    """
    def __init__(self, df_closed):
        pass

def get_top_N_tickers(N, time_frame, path, signal_file="/home/u237/projects/parsing_cindicator/data/CND_AB_parsed_fix1.json") -> list:
    """
    Returns a list of top N backtestable tickers (in the form that they appear inside the signal df) that exist inside 
    binance data directory, that match the pattern of ticker files acc to provided time frame 
    These are returned in their order of frequency from ticker with most signals, to least. 
    The list returned from this method will be fed to the class bt_top_N() as a parameter of 
    ticker to be bactested | 2022-08-03 00:14
    
    PARAMETERS:
    N          - number of tickers to be returned, if it is greater than the number of tickers available, it will return all tickers
    time_frame - the time frame of the tickers to be returned, can be "1h", "30m", "15m"
    path       - the path to the directory containing the ticker files
    """
    signals_df   = pd.read_json(signal_file)

    # getting top N tickers as a series object to iterate in the loop
    ticker_counts = signals_df.value_counts("ticker", dropna=False)
    N_tickers = ticker_counts.head(2* N)
    
    # turn N_ticker to a list
    N_tickers = N_tickers.index.tolist()

    pat_ticker_files = f"([A-Z]+\_[A-Z]+)\-{time_frame}\.json$"
    # pat_ticker_files = "([A-Z]+\_[A-Z]+)\-(1h|30m|15m)\.json$"
    
    # under directory path, search all files that contian the pattern pat_ticker_files, return a list of tickers by turning the
    # first capture group of all matches into a list
    tickers_in_path = [re.search(pat_ticker_files, f).group(1) for f in os.listdir(path) if re.search(pat_ticker_files, f)]

    top_N_ticker_in_path = []
    # for each element in tickers, replace "_" with "/" and remove the last T if ends with "USDT"

    for i, ticker in enumerate(tickers_in_path):
        ticker = ticker.replace("_", "/")
        if ticker.endswith("USDT"):
            # remove the last character
            ticker = ticker[:-1]
        tickers_in_path[i] = ticker

        # get the index of ticker on N_tickers
        ind = N_tickers.index(ticker)
        top_N_ticker_in_path.append((ticker, ind))
    
    df_top = pd.DataFrame(top_N_ticker_in_path, columns=["ticker","rank"])

    # sort acc to rank and turn top 20 into list
    df_sorted = df_top.sort_values(by=['rank'])
    top_N_ticker_in_path = df_sorted.head(N).ticker.tolist()

    return top_N_ticker_in_path








