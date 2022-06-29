# test download code
docker-compose run --rm freqtrade download-data --exchange binance --pairs ETH/USDT -t 30m 

# target download code
freqtrade download-data -t 30m --timerange 20180401- --include-inactive-pairs --exchange binance --pairs ZEC/USDT
freqtrade download-data -t 15m --timerange 20180401- --include-inactive-pairs --exchange binance --pairs ZEC/USDT

time docker-compose run --rm freqtrade download-data --exchange binance --pairs ZEC/USDT -t 15m --timerange 20180401- 


# backtest code

# !! doesnt work
dcr freqtrade backtesting --strategy SampleStrategy
# !! works
dcr freqtrade backtesting --strategy BBRSINaiveStrategy