## Data Source
### Reddit
download data with reddit.py

### Telegram
download data from desktop apps, pasre it with telegram.py

### Discord
temporarly discarded

---
## Data Preprcessing & Sentiment Analysis
aggragate_sia_telegram.py

aggragate_sia_reddit.py


---
## Fetch Finantial Data
data/coin_metrics.csv

tools.py (when new indicators requires)

(btc-indicators.csv)

---
## Forecasting Model
models.py 

---
## How It Works
First run reddit/telegram.py to fetch data

Then run aggragate_sia_{}.py to calculate SA scores and aggregate by day 

(produce data/{}/all-submissions-sia.csv # all single posts)

(produce data/{}/data/agg_sia_ind_{}.csv) # agg by date

Finally run models.py (only need data/agg_sia_ind_{}.csv)

I seperates these steps due to large computing power required, but by saving csv I can run in the middle.

## Workshop Report
check ./report.pdf 
<!---
or [click me](report.pdf)
-->
