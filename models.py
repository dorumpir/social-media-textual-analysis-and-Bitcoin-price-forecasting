# id: 'date'

# features
#         sia_bert_sum=('sia_bert', 'sum'),
#         sia_vader_sum=('sia_vader', 'sum'),
#         sia_sum=('sia', 'sum'),
#         num_posts=('title', 'count')

# labels
#'BTC / USD Denominated Closing Price', 'BTC / Active Addr Cnt', 'BTC / NVT', 'BTC / Tx Cnt', 'BTC / Xfer Cnt', 'BTC / Market Cap (USD)'

##################
# f
# sia_sum, #posts, #comments, [indicators], @up/down

# t
# price, @up/down

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor

import xgboost as xgb
import joblib

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
# import seaborn as sns
from sklearn.model_selection import GridSearchCV


from pathlib import Path

for _ in range(10):
    for which in ['telegram', 'reddit']:
        for model_str in ['xgb', 'lstm']:
            for target in ['Log(Price)', 'close']:

    # which = 'telegram'
    # model_str = 'xgb'
                train_model = True
                data_path = Path('data')

                ts_window_size = 30
                test_data_size = ts_window_size + 30 # round(len(df)*0.20)

                features = [
                    #'date',
                    'sia_sum',
                    'num_posts',
                    'num_comments',
                    'BTC / Active Addr Cnt',
                    'BTC / NVT',
                    'BTC / Tx Cnt',
                    'MACD',
                    'SIGNAL',
                    'EMA',
                    'RSI',
                    'volume'
                ]
                targets = [
                    #'Log(Price)',
                    #'close',
                    target
                ]
                all_inputs = targets + features

                # read data
                df = pd.read_csv(data_path / 'agg_sia_ind_{}.csv'.format(which))
                df = df[targets + features]
                # split data set
                df_for_training=df[:-test_data_size]
                df_for_testing=df[-test_data_size:]
                print(df_for_training.shape)
                print(df_for_testing.shape)

                scaler = MinMaxScaler(feature_range=(0, 1))
                df_for_training_scaled = scaler.fit_transform(df_for_training)
                df_for_testing_scaled = scaler.transform(df_for_testing)

                def createXY(dataset,n_past):
                    # TODO here
                    dataX = []
                    dataY = []
                    for i in range(n_past, len(dataset)):
                            dx = dataset[i - n_past:i, 0:dataset.shape[1]]
                            dataX.append(dx if model_str == 'lstm' else dx.flatten())
                            dataY.append(dataset[i,0])
                    return np.array(dataX),np.array(dataY)

                trainX,trainY=createXY(df_for_training_scaled, ts_window_size)
                testX,testY=createXY(df_for_testing_scaled, ts_window_size)
                print("trainX Shape-- ",trainX.shape)
                print("trainY Shape-- ",trainY.shape)
                print("testX Shape-- ",testX.shape)
                print("testY Shape-- ",testY.shape)

                ###

                # model
                if model_str == 'lstm':
                    def build_model(optimizer):
                        grid_model = Sequential()
                        grid_model.add(LSTM(50, return_sequences=True, input_shape=(ts_window_size, len(all_inputs))))
                        grid_model.add(LSTM(50))
                        grid_model.add(Dropout(0.2))
                        grid_model.add(Dense(1))

                        grid_model.compile(loss='mse',optimizer=optimizer)
                        return grid_model

                    if train_model:
                    # grid search GTX 3090
                        # TODO validation data
                        grid_model = KerasRegressor(build_fn=build_model,verbose=1,)#validation_data=(testX,testY))
                        parameters = {
                            'batch_size' : (8, 16),#, 20),
                            'epochs' : (8,16),
                            'optimizer' : ('adam','Adadelta')
                        }

                        grid_search  = GridSearchCV(estimator = grid_model,
                                                    param_grid = parameters,
                                                    cv = 2)

                        grid_search = grid_search.fit(trainX,trainY)
                        my_model=grid_search.best_estimator_.model
                        mean_test_score = grid_search.cv_results_['mean_test_score']
                        best_param = grid_search.best_params_
                        diagram_dict = {
                            'mean_test_score': grid_search.cv_results_['mean_test_score'],
                            'best_param': grid_search.best_params_,
                            'best_model_history':grid_search.best_estimator_.model.history.history
                        }
                        print(diagram_dict)
                        import pickle
                        with open(data_path / 'diagram-train-lstm.pickle', 'wb') as f:
                            pickle.dump(diagram_dict, f)

                    else:
                        my_model = keras.models.load_model('.')#("lstm_model.h5")


                if model_str == 'xgb':
                    if train_model:
                    # grid search
                        grid_model = xgb.XGBRegressor(tree_method='gpu_hist', gpu_id=0, verbose=0)
                        parameters = {
                            'max_depth':[1,2,5,10,20],
                            'n_estimators':[20,30,50,70,100],
                            'learning_rate':[0.1,0.2,0.3,0.4,0.5]
                        }

                        grid_search  = GridSearchCV(estimator = grid_model,
                                                    param_grid = parameters,
                                                    cv = 2)

                        grid_search = grid_search.fit(trainX,trainY)
                        my_model=grid_search.best_estimator_#.model
                        diagram_dict = {
                            'mean_test_score': grid_search.cv_results_['mean_test_score'],
                            'best_param': grid_search.best_params_,
                            #'best_model_history': grid_search.best_estimator_.history.history
                        }
                        print(diagram_dict)
                        import pickle

                        with open(data_path / 'diagram-train-xgb.pickle', 'wb') as f:
                            pickle.dump(diagram_dict, f)
                    else:
                        my_model = joblib.load('xgb_model.pkl')


                ###
                # predict y
                prediction=my_model.predict(testX)
                #print("prediction\n", prediction)
                print("\nPrediction Shape-",prediction.shape)

                # inverse scaling of y
                prediction_copies_array = np.repeat(prediction, len(all_inputs), axis=-1)
                pred = scaler.inverse_transform(
                            np.reshape(
                                prediction_copies_array,
                                (
                                    len(prediction),
                                    len(all_inputs)
                                )
                            )
                        )[:, 0]
                original_copies_array = np.repeat(testY, len(all_inputs), axis=-1)
                original = scaler.inverse_transform(
                                np.reshape(
                                    original_copies_array,
                                    (
                                        len(testY),
                                        len(all_inputs)
                                    )
                                )
                            )[:, 0]

                from sklearn.metrics import mean_squared_error
                mse = mean_squared_error(testY, prediction, squared=False)
                print('normalized price mse: ', mse)
                print('real price mse: ', mean_squared_error(original, pred, squared=False))


                import pickle
                values = {'truth': original, 'prediction': pred}
                with open(data_path / 'btc-prices-prediction.pickle', 'wb') as f:
                    pickle.dump(values, f)

                # plot (train data)
                prefix = 'Log: ' if 'Log' in targets[0] else ''
                plt.plot(original, color = 'red', label = prefix + 'Real Crypto Price')
                plt.plot(pred, color = 'blue', label = prefix + 'Predicted Crypto Price')
                mname, sname = {'lstm': 'LSTM', 'xgb': 'Gradient Boosting'}, {'reddit': 'Reddit', 'telegram': 'Telegram'}
                plt.title('{} model / {} data'.format(mname[model_str], sname[which]))
                plt.xlabel('Time')
                plt.ylabel(prefix + 'BTC Crypto Price')
                plt.legend()
                for i in range(10000):
                    from pathlib import Path
                    figname = 'out-{}-{}-{}-{}.png'.format(model_str, which, mse, i)
                    if not Path(figname).is_file():
                        plt.savefig(figname)
                        plt.title('')
                        plt.savefig('notitle-{}-{}-{}-{}.png'.format(model_str, which, mse, i))
                        break
                plt.show()

                ### now on future test data
                # omitted
                if train_model and model_str == 'lstm':
                    my_model.save('lstm_model.h5')
                    print('Model Saved!')
                if train_model and model_str == 'xgb':
                    joblib.dump(my_model, 'xgb_model.pkl')
                    print('Model Saved!')