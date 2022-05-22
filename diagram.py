'''
## 画图
1. 预处理、ad、ner 各损失了多少
#2. bert和vader的分布图 CMarket
#3. chat和post的词云

4. comments posts 数量点图

---需要机器

5. 几个自变量和因变量的corr图
6. 每个变量的相关性
xgboost?
8. 预测
9. chat用于预测
'''
from pathlib import Path
from wordcloud import WordCloud
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_path = Path('data')

def word_cloud(which='reddit'):
    df = pd.read_csv(data_path/ which /'all-submissions-sia.csv')
    df['text'] = df['text'].astype(str)
    wordcloud = WordCloud().generate(' '.join(df['text']))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.savefig('wordcloud-' + which + '.png')
    plt.show()

# word_cloud('reddit')
# word_cloud('telegram')

def vader_vs_bert():
    df = pd.read_csv(data_path / 'reddit/CryptoMarkets-submissions-sia-with-bert.csv')
    ax_v = sns.distplot(df['sia_vader'], rug=True)
    plt.show()
    ax_v.figure.savefig('vader_dist.png')
    ax_b = sns.distplot(df['sia_bert'], rug=True)
    plt.show()
    ax_b.figure.savefig('bert_dist.png')

# vader_vs_bert()

def n_comments_posts():
    df = pd.read_csv(data_path / 'reddit/all-submission-sia.csv')
    ax_v = sns.distplot(df['sia_vader'], rug=True)
    plt.show()
    ax_v.figure.savefig('vader_dist.png')
    ax_b = sns.distplot(df['sia_bert'], rug=True)
    plt.show()
    ax_b.figure.savefig('bert_dist.png')

def training_curve(model='lstm'):
    import pickle
    p = data_path / 'diagram-train-{}.pickle'.format(model)
    with open(p, 'rb') as f:
        data = pickle.load(f)
        '''
            'mean_test_score': grid_search.cv_results_['mean_test_score'],
            'best_param': grid_search.best_params_,
            {not in xgb} 'best_model_history':grid_search.best_estimator_.model.history.history
        '''
        y = data['mean_test_score']
        x = list(range(1, len(y)+1))
        plt.plot(x, y)
        plt.title('Grid Search Scores')
        plt.xlabel('Model')
        plt.ylabel('Mean Test Score')
        plt.savefig('training-curve-' + model + '.png')
        plt.show()


# training_curve('lstm')
# training_curve('xgb')

def corr_heatmap(which='telegram'):
    '''
        'BTC / Closing Price',
        'BTC / Active Addr Cnt',
        'BTC / NVT',
        'BTC / Tx Cnt',
        'BTC / Xfer Cnt',
        'BTC / Market Cap (USD)',
    '''
    # p = data_path / 'agg_sia_ind.csv'
    p = data_path / 'agg_sia_ind_{}.csv'.format(which)
    df = pd.read_csv(p)
    df = df.drop(['date'], axis=1)
    #cm = df.columns.tolist()
    df.columns = df.columns.str.replace('BTC / ', '')
    xcorr = df.corr()
    # 设置右上三角不绘制

    # mask为 和相关系数矩阵xcorr一样大的 全0(False)矩阵
    mask = np.zeros_like(xcorr, dtype=np.bool)
    # 将mask右上三角(列号》=行号)设置为True
    mask[np.triu_indices_from(mask)] = True
    # cmap是设置热图的颜色
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # 绘制热图
    ax = sns.heatmap(xcorr, mask=mask, cmap=cmap, square=True, annot=True, fmt='0.2f')
    plt.show()
    ax.figure.savefig('corr-btc-{}.png'.format(which))
    #  xcorr : 数据矩阵
    #  mask : 为True的元素对应位置不会画出来（mask面具的意义）
    #  cmap: 颜色设置
    #  square: （True）代表行列长度一致，且绘制的每个也是方格
    #  annot ： 在格内显示数据
    #  fmt ：数据格式

corr_heatmap('reddit')