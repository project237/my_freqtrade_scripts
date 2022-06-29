# test download code
freqtrade download-data --exchange binance --pairs ETH/USDT

# target download code
freqtrade download-data -t 30m --timerange 20180401- --include-inactive-pairs --exchange binance --pairs ZEC/USDT
freqtrade download-data -t 15m --timerange 20180401- --include-inactive-pairs --exchange binance --pairs ZEC/USDT