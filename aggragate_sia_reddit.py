from pathlib import Path
import math
import pandas as pd
from SentimentAnalysis.analyze import get_model, get_sia_prob
from SentimentAnalysis.senti_bignomics import senti_bignomics
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 

data_path = Path('data')
reddit_data_path = Path(data_path) / 'reddit'

invalid_reddit_kws = ['[removed]', '[deleted]']

def get_vader_sia():
    analyser = SentimentIntensityAnalyzer()
    analyser.lexicon.update(senti_bignomics)
    return analyser

vader_sia = get_vader_sia()
bert_models_tokenizers = [get_model(), ]

def preprocessing_reddit(df):
    # def get_text(x, limit=500):
    #     texts = []
    #     title, body = x['title'], x['selftext']
    #     len_t, len_b = len(title), len(body)
    #     texts.append(title)
    #     if len_t < limit:
    #         len_rest = limit - len_t
    #         if len_rest * 2 < len_b
    #             pass
    df[['title', 'selftext']] = df[['title', 'selftext']].fillna('')
    # here dealt with text
    df['text'] = df.apply(lambda x: (x['title'] + ' ' + x['selftext'])[: 500], axis=1)
    df[['num_comments']] = df[['num_comments']].astype('int32').fillna(0)
    df[['score']] = df[['score']].astype('int32').fillna(1)
    df[['upvote_ratio']] = df[['upvote_ratio']].astype('float32').fillna(1)
    df.dropna(subset=['created_utc'])
    df[['created_utc']] = df[['created_utc']].astype('int64').fillna(0)
    df['date'] = pd.to_datetime(df['created_utc'], unit='s').dt.date ## pd.Timestamp
    return df


def clean_reddit(df:pd.DataFrame):
    '''
    reddit size 718469  -  247530
    reddit after dropped 470939
    data/reddit/CryptoMarkets-submissions.csv
    reddit size 97171  -  36246
    reddit after dropped 60925
    data/reddit/bitcoin-submissions.csv
    reddit size 215855  -  33040
    reddit after dropped 182815
    data/reddit/btc-submissions.csv
    reddit size 65625  -  16037
    reddit after dropped 49588
    '''
    # 1. formatting preprocessing
    df = preprocessing_reddit(df)
    # 2. drop invalid rows
    dropped = []
    for idx, row in df.iterrows():
        flag = False
        title, text = row.get('title', ''), row.get('selftext', '')
        # 1. invalid post
        for kw in invalid_reddit_kws:
            if kw in title or kw in text:
                flag = True
        # 3. not about BTC (NER) # update: excluded in the end
        if flag:
            dropped.append(idx)
    print('reddit size', df.shape[0], ' - ', len(dropped))
    df.drop(index=dropped, inplace=True)
    print('reddit after dropped', df.shape[0])
    return df


def sia_score(row, keys=('text',), which='bert'):
    def to_n1_p1(x):
        return (-1) + (1 - (-1)) / (1 - 0) * (x - 0)
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    sentences = [row[key] for key in keys]
    if which == 'bert':
        scs = []
        for sentence in sentences:
            scs.extend([get_sia_prob(model, tokenizer, sentence) for model, tokenizer in bert_models_tokenizers])
        sc = sum(scs) / len(scs)
        sc = to_n1_p1(sc)
    elif which == 'vader':
        scs = []
        for sentence in sentences:
            scs.append(vader_sia.polarity_scores(sentence)['compound'])
        sc = sum(scs) / len(scs)
    else:
        raise
    return sc

def get_post_sia(row, sia_key=''):
    def to_n1_p1(x):
        return (-1) + (1 - (-1)) / (1 - 0) * (x - 0)
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    ## popular_score
    # .score: num of upvotes
    # .upvote_ratio:
    # .num_comments
    # score = popular_score[@#comments, #votes] * agree_score[@upvote_ratio] * text_sia[@sc]
    sc = row[sia_key]
    num_comments = row.get('num_comments', 0)
    num_upvotes = row.get('score', 1)
    upvote_ratio = row.get('upvote_ratio', 1)
    num_votes = num_upvotes / upvote_ratio
    ratio_nv_nc = 2.316
    popular_score = sigmoid((num_votes + num_comments * ratio_nv_nc) / 10)
    agree_score = to_n1_p1(upvote_ratio)
    return popular_score * agree_score * sc


def sia_reddit(bert_ratio=0.0):
    dfs = []
    path = reddit_data_path
    for p in path.glob('*submissions.csv'):
        print(p)
        df = pd.read_csv(p.absolute())
        # 0. data cleaning, delete irrelevant posts, delete bot posts
        df = clean_reddit(df)
        key_sia, key_sia_bert, key_sia_vader = 'sia', 'sia_bert', 'sia_vader'
        df[key_sia] = 0.0
        df[key_sia_bert] = 0.0
        df[key_sia_vader] = 0.0
        vader_ratio = 1 - bert_ratio
        # 1. bert
        if bert_ratio >= 0.1:
            # TODO define another albert model ('title', 'selftext')
            df[key_sia_bert] = df.apply(lambda x: sia_score(x, keys=('text',), which='bert'), axis=1)
        # 2. vadar
        if vader_ratio >= 0.1:
            df[key_sia_vader] = df.apply(lambda x: sia_score(x, keys=('text',), which='vader'), axis=1)
        # 3. sum of sia
        df[key_sia] = df.apply(lambda x: bert_ratio * get_post_sia(x, sia_key=key_sia_bert) + \
                                         vader_ratio * get_post_sia(x, sia_key=key_sia_vader),
                               axis=1)
        dfs.append(df)
        pstr = str(p.absolute())
        df.to_csv(Path(pstr[:-4 ] + '-sia.csv'))
    df = pd.concat(dfs)
    return df

def agg_reddit_by_day(df):
    # 1. groupby date
    # 2. formular: upvote / #comments
    # 3. mean/sum
    # 4. write to csv
    # independence
    # df_agg_sia = df.groupby("date").agg({
    #     'sia_bert': 'sum',
    #     'sia_vader': 'sum',
    #     'title': 'count'
    # }).reset_index()
    df_agg_sia = df.groupby("date").agg(
        sia_bert_sum=('sia_bert', 'sum'),
        sia_vader_sum=('sia_vader', 'sum'),
        sia_sum=('sia', 'sum'),
        num_posts=('title', 'count'),
        num_comments=('num_comments', 'sum'),
    ).reset_index()
    df_X = df_agg_sia # [['date', 'sia_bert_sum', 'sia_vader', 'title']]
    # dependency
    df_features = pd.read_csv(data_path / 'btc-indicators.csv', parse_dates=["Time"])
    df_features['date'] = pd.to_datetime(df_features['Time']).dt.date
    df_Y = df_features[
        [
            'date',
            'BTC / Closing Price',
            'BTC / Active Addr Cnt',
            'BTC / NVT',
            'BTC / Tx Cnt',
            'MACD',
            'SIGNAL',
            'EMA',
            'RSI',
            'volume'
        ]
    ]
    df_m = df_X.merge(df_Y, how="left", on="date")
    return df_m

def main(which='reddit'):
    submission_path = reddit_data_path / 'all-submissions-sia.csv'
    reddit_df = sia_reddit(bert_ratio=0.3)
    reddit_df.to_csv(submission_path)
    reddit_df = pd.read_csv(submission_path, parse_dates=["date"])
    reddit_df['date'] = pd.to_datetime(reddit_df['created_utc'], unit='s').dt.date

    df = agg_reddit_by_day(reddit_df)
    df.to_csv(data_path / ('agg_sia_ind_' + which + '.csv'))

    return df


if __name__ == '__main__':
    main()
