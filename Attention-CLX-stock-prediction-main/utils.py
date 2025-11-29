import numpy as np
import pandas as pd
from sklearn import metrics
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm  # acf,pacf plot
import matplotlib.pyplot as plt

def adf_test(temp):
    # p-value>0.562 or Critical Value(1%)>-3.44, non-stationary
    t = adfuller(temp)
    output = pd.DataFrame(index=['Test Statistic Value', 'p-value', 'Lags Used', 'Number of Observations Used', 'Critical Value(1%)', 'Critical Value(5%)', 'Critical Value(10%)'], columns=['value'])
    output['value']['Test Statistic Value'] = t[0]
    output['value']['p-value'] = t[1]
    output['value']['Lags Used'] = t[2]
    output['value']['Number of Observations Used'] = t[3]
    output['value']['Critical Value(1%)'] = t[4]['1%']
    output['value']['Critical Value(5%)'] = t[4]['5%']
    output['value']['Critical Value(10%)'] = t[4]['10%']
    print(output)

def acf_pacf_plot(seq,acf_lags=20,pacf_lags=20):
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(seq, lags=acf_lags, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(seq, lags=pacf_lags, ax=ax2)
    plt.show()

def order_select_ic(training_data_diff):
    (p, q) = sm.tsa.arma_order_select_ic(training_data_diff, max_ar=6, max_ma=4, ic='bic')['bic_min_order']  # AIC
    print(p, q)  # 2 0

def order_select_search(training_set):
    df2 = training_set['close'].diff(1).dropna()
    # pmax = int(len(df2) / 10)
    # qmax = int(len(df2) / 10)
    pmax = 5
    qmax = 5
    bic_matrix = []
    print('^', pmax, '^^', qmax)
    for p in range(pmax + 1):
        temp3 = []
        for q in range(qmax+1):
            try:
                # print('!', ARIMA(data['close'], order=(p, 1, q)).fit().bic)
                # temp.append(ARIMA(data['close'], order=(p, 1, q)).fit().bic)
                temp3.append(sm.tsa.ARIMA(training_set['close'], order=(p, 1, q)).fit().bic)
            except:
                temp3.append(None)
        bic_matrix.append(temp3)
    bic_matrix = pd.DataFrame(bic_matrix) 
    # print('&', bic_matrix)
    # print('&&', bic_matrix.stack())
    # print('&&&', bic_matrix.stack().astype('float64'))
    p, q = bic_matrix.stack().astype('float64').idxmin()
    print('p and q: %s,%s' % (p, q)) 

def create_dataset(dataset, look_back=20):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back),:]
        dataX.append(a)
        dataY.append(dataset[i + look_back,:])
    TrainX = np.array(dataX)
    Train_Y = np.array(dataY)

    return TrainX, Train_Y

def evaluation_metric(y_test,y_hat):
    MSE = metrics.mean_squared_error(y_test, y_hat)
    RMSE = MSE**0.5
    MAE = metrics.mean_absolute_error(y_test,y_hat)
    R2 = metrics.r2_score(y_test,y_hat)
    print('MSE: %.5f' % MSE)
    print('RMSE: %.5f' % RMSE)
    print('MAE: %.5f' % MAE)
    print('R2: %.5f' % R2)

def GetMAPE(y_hat, y_test):
    sum = np.mean(np.abs((y_hat - y_test) / y_test)) * 100
    return sum

def GetMAPE_Order(y_hat,y_test):
    zero_index = np.where(y_test == 0)
    y_hat = np.delete(y_hat, zero_index[0])
    y_test = np.delete(y_test, zero_index[0])
    sum = np.mean(np.abs((y_hat - y_test) / y_test)) * 100
    return sum

# def NormalizeMult(data):
#     """
#       data: np.ndarray, shape (N, D)
#       return:
#           normalized_data: same shape
#           normalize: (D, 2)  -> [[min, max], ...]
#       """
#     data = np.array(data, dtype='float64')
#     N, D = data.shape
#
#     normalize = np.zeros((D, 2), dtype='float64')
#     normalized = np.zeros_like(data)
#
#     for i in range(D):
#         col_min = data[:, i].min()
#         col_max = data[:, i].max()
#         normalize[i, 0] = col_min
#         normalize[i, 1] = col_max
#         delta = col_max - col_min
#
#         if delta == 0:
#             normalized[:, i] = 0
#         else:
#             normalized[:, i] = (data[:, i] - col_min) / delta
#
#     return normalized, normalize
def NormalizeMult(data):
    """
    训练阶段使用：
    输入原始 data，输出标准化后的 data_norm 和 (mean, std) 参数 normalize
    """
    data = np.array(data, dtype='float64')
    N, D = data.shape

    data_norm = np.zeros_like(data)
    normalize = np.zeros((D, 2), dtype='float64')  # 每列的 mean, std

    for i in range(D):
        col = data[:, i]
        mean_i = col.mean()
        std_i  = col.std()

        normalize[i, 0] = mean_i
        normalize[i, 1] = std_i

        if std_i == 0 or np.isclose(std_i, 0.0):
            data_norm[:, i] = 0.0
        else:
            data_norm[:, i] = (col - mean_i) / std_i

    return data_norm, normalize


# def FNormalizeMult(data, normalize):
#     #inverse NormalizeMult
#     data = np.array(data)
#     listlow = normalize[0]
#     listhigh = normalize[1]
#     delta = listhigh - listlow
#     if delta != 0:
#         for i in range(len(data)):
#             data[i, 0] = data[i, 0] * delta + listlow
#     return data


# def FNormalizeMult(data, normalize):
#     """
#     data: normalized array shape (N, D)
#     normalize: (D, 2)
#     return: restored data (N, D)
#     """
#     data = np.array(data, dtype='float64')
#     N, D = data.shape
#     restored = np.zeros_like(data)
#
#     for i in range(D):
#         col_min = normalize[i, 0]
#         col_max = normalize[i, 1]
#         delta = col_max - col_min
#
#         if delta == 0:
#             restored[:, i] = data[:, i]
#         else:
#             restored[:, i] = data[:, i] * delta + col_min
#
#     return restored
#
# def NormalizeMultUseData(data,normalize):
#     data = np.array(data)
#     for i in range(0, data.shape[1]):
#         listlow = normalize[i, 0]
#         listhigh = normalize[i, 1]
#         delta = listhigh - listlow
#         if delta != 0:
#             for j in range(0,data.shape[0]):
#                 data[j,i]  =  (data[j,i] - listlow)/delta
#     return  data

def FNormalizeMult(data, normalize):
    """
    data:  标准化后的数组，shape = (N, D)
    normalize: 每一列的 (mean, std)，shape = (D, 2)
               normalize[i, 0] = mean_i
               normalize[i, 1] = std_i
    return: 还原到原始尺度的数据，shape = (N, D)
    """
    data = np.array(data, dtype='float64')

    # 保证是 2D
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    N, D = data.shape
    restored = np.zeros_like(data)

    for i in range(D):
        mean_i = normalize[i, 0]
        std_i  = normalize[i, 1]

        if std_i == 0 or np.isclose(std_i, 0.0):
            # 这一列在训练集中是常数：直接还原成 mean
            restored[:, i] = mean_i
        else:
            restored[:, i] = data[:, i] * std_i + mean_i

    return restored



def NormalizeMultUseData(data, normalize):
    """
    data:      原始数据，shape = (N, D)
    normalize: (mean, std) 参数，shape = (D, 2)
    return:    标准化后的数据，shape = (N, D)

    标准化公式:  x_norm = (x - mean) / std
    """
    data = np.array(data, dtype='float64')
    N, D = data.shape
    normed = np.zeros_like(data)

    for i in range(D):
        mean_i = normalize[i, 0]
        std_i  = normalize[i, 1]

        if std_i == 0 or np.isclose(std_i, 0.0):
            # 这一列在训练集中是常数：统一置 0
            normed[:, i] = 0.0
        else:
            normed[:, i] = (data[:, i] - mean_i) / std_i

    return normed


def data_split(sequence, n_timestamp):
    X = []
    y = []
    for i in range(len(sequence)):
        end_ix = i + n_timestamp

        if end_ix > len(sequence) - 1:
            break

        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def prepare_data(series, n_test, n_in, n_out):
    values = series.values
    supervised_data = series_to_supervised(values, n_in, n_out)
    print('supervised_data', supervised_data)
    train, test = supervised_data.loc[:3499, :], supervised_data.loc[3500:, :]
    return train, test