## data source
### reddit
download data with reddit.py

### telegram
download data from desktop apps, pasre it with telegram.py

### discord
temporarly discarded

---
## data preprcessing & sentiment analysis
aggragate_sia_telegram.py

aggragate_sia_reddit.py


---
## fetch finantial data
data/coin_metrics.csv

tools.py (when new indicators requires)

(btc-indicators.csv)

---
## forecasting model
models.py 

---
## how it works
first run reddit/telegram.py to fetch data

then run aggragate_sia_{}.py to calculate SA scores and aggregate by day 

(produce data/{}/all-submissions-sia.csv # all single posts)

(produce data/{}/data/agg_sia_ind_{}.csv) # agg by date

finally run models.py (only need data/agg_sia_ind_{}.csv)

I seperates these steps due to large computing power required, but by saving csv I can run in the middle.

## report
[click me](report.pdf)