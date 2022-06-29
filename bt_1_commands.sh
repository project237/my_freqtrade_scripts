# test download code
freqtrade download-data --exchange binance

# target download code
freqtrade download-data -t 30m --timerange 20180401- --include-inactive-pairs --exchange binance
freqtrade download-data -t 15m --timerange 20180401- --include-inactive-pairs --exchange binance