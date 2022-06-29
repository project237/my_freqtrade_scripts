cd /home/u237/projects/backtests/cindicator-bt1/ft_userdata

format_time() {
  ((h=${1}/3600))
  ((m=(${1}%3600)/60))
  ((s=${1}%60))
  printf "%02d:%02d:%02d\n" $h $m $s
 }

t0=$SECONDS

printf "\nDOWLOADING 1H DATA..\n\n"
docker-compose run --rm freqtrade download-data --exchange binance --pairs ZEC/USDT -t 1h 
t1=$SECONDS

printf "\nDOWLOADING 30m DATA..\n\n"
docker-compose run --rm freqtrade download-data --exchange binance --pairs ZEC/USDT -t 30m 
t2=$SECONDS

printf "\nDOWLOADING 1H DATA..\n\n"
docker-compose run --rm freqtrade download-data --exchange binance --pairs ZEC/USDT -t 15m
t3=$SECONDS

printf "\n 1h COMPLETE IN $(format_time $(($t1 - $t0)))\n"
printf "\n30m COMPLETE IN $(format_time $(($t2 - $t1)))\n"
printf "\n15m COMPLETE IN $(format_time $(($t3 - $t2)))\n"

cd -

