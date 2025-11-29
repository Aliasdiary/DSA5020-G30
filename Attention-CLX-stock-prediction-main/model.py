# import numpy as np
# import xgboost as xgb
# import tensorflow as tf
# import keras.backend as K
#
# from utils import *
#
# from keras.models import Model, Sequential
# from keras.layers import (
#     Input, Dense, LSTM, Conv1D, Dropout, Bidirectional,
#     Multiply, Permute, RepeatVector, Flatten, Lambda
# )
#
#
# def attention_3d_block_merge(inputs,single_attention_vector = False):
#     # inputs.shape = (batch_size, time_steps, input_dim)
#     input_dim = int(inputs.shape[2])
#     a = inputs
#     # a = Permute((2, 1))(inputs)
#     # a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
#     a = Dense(input_dim, activation='softmax')(a)
#     if single_attention_vector:
#         a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
#         a = RepeatVector(input_dim)(a)
#     a_probs = Permute((1, 2), name='attention_vec')(a)
#
#     output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
#     return output_attention_mul
#
#
# def attention_block(inputs):
#     """
#     inputs: (batch, time_steps, hidden_size)
#     output: (batch, time_steps, hidden_size)
#     """
#     time_steps = int(inputs.shape[1])      # e.g., 20
#     hidden_size = int(inputs.shape[2])     # e.g., 128
#
#     # 1. 计算每个时间步的注意力分数 => (batch, time_steps, 1)
#     score = Dense(1, activation='tanh')(inputs)
#
#     # 2. softmax 归一化 => 还是 (batch, time_steps, 1)
#     score = Lambda(
#         lambda x: tf.nn.softmax(x, axis=1),
#         output_shape=lambda s: s
#     )(score)
#
#     # 3. 重复 hidden_size 次 => (batch, time_steps, hidden_size)
#     score = Lambda(
#         lambda x: tf.repeat(x, repeats=hidden_size, axis=2),
#         output_shape=lambda s: (s[0], s[1], hidden_size)
#     )(score)
#
#     # 4. 加权
#     output = Multiply()([inputs, score])   # (batch, time_steps, hidden_size)
#     return output
#
#
# # =============================
# # Main Attention Model
# # =============================
# def attention_model(INPUT_DIMS=7, TIME_STEPS=20, HIDDEN=128):
#     inputs = Input(shape=(TIME_STEPS, INPUT_DIMS))
#
#     # LSTM layer
#     lstm_out = LSTM(HIDDEN, return_sequences=True)(inputs)
#
#     # Attention layer
#     attention_out = attention_block(lstm_out)
#
#     # Time pooling (Keras3: use tf.reduce_mean)
#     time_pool = Lambda(
#         lambda x: tf.reduce_mean(x, axis=1),
#         output_shape=lambda s: (s[0], s[2])
#     )(attention_out)
#
#     # Final Dense layer
#     output = Dense(1)(time_pool)
#
#     model = Model(inputs, output)
#     return model
#
#
# def PredictWithData(data,data_yuan,name,modelname,INPUT_DIMS = 13,TIME_STEPS = 20):
#     print(data.columns)
#     yindex = data.columns.get_loc(name)
#     data = np.array(data, dtype='float64')
#     data, normalize = NormalizeMult(data)
#     data_y = data[:, yindex]
#     data_y = data_y.reshape(data_y.shape[0], 1)
#
#     testX, _ = create_dataset(data)
#     _, testY = create_dataset(data_y)
#     print("testX Y shape is:", testX.shape, testY.shape)
#     if len(testY.shape) == 1:
#         testY = testY.reshape(-1, 1)
#
#     model = attention_model(INPUT_DIMS)
#     model.load_weights(modelname)
#     model.summary()
#     y_hat =  model.predict(testX)
#     testY, y_hat = xgb_scheduler(data_yuan, y_hat)
#     return y_hat, testY
#
# def lstm(model_type,X_train,yuan_X_train):
#     if model_type == 1:
#         # single-layer LSTM
#         model = Sequential()
#         model.add(LSTM(units=50, activation='relu',
#                     input_shape=(X_train.shape[1], 1)))
#         model.add(Dense(units=1))
#         yuan_model = Sequential()
#         yuan_model.add(LSTM(units=50, activation='relu',
#                     input_shape=(yuan_X_train.shape[1], 5)))
#         yuan_model.add(Dense(units=5))
#     if model_type == 2:
#         # multi-layer LSTM
#         model = Sequential()
#         model.add(LSTM(units=50, activation='relu', return_sequences=True,
#                     input_shape=(X_train.shape[1], 1)))
#         model.add(LSTM(units=50, activation='relu'))
#         model.add(Dense(1))
#
#         yuan_model = Sequential()
#         yuan_model.add(LSTM(units=50, activation='relu', return_sequences=True,
#                     input_shape=(yuan_X_train.shape[1], 5)))
#         yuan_model.add(LSTM(units=50, activation='relu'))
#         yuan_model.add(Dense(5))
#     if model_type == 3:
#         # BiLSTM
#         model = Sequential()
#         model.add(Bidirectional(LSTM(50, activation='relu'),
#                                 input_shape=(X_train.shape[1], 1)))
#         model.add(Dense(1))
#
#         yuan_model = Sequential()
#         yuan_model.add(Bidirectional(LSTM(50, activation='relu'),
#                                     input_shape=(yuan_X_train.shape[1], 5)))
#         yuan_model.add(Dense(5))
#
#     return model,yuan_model
#
# def xgb_scheduler(data,y_hat):
#     close = data.pop('close')
#     data.insert(5, 'close', close)
#     train, test = prepare_data(data, n_test=len(y_hat), n_in=6, n_out=1)
#     testY, y_hat2 = walk_forward_validation(train, test)
#     return testY, y_hat2
#
# def xgboost_forecast(train, testX):
#     # transform list into array
#     train = np.asarray(train)
#     # print('train', train)
#     # split into input and output columns
#     trainX, trainy = train[:, :-1], train[:, -1]
#     # print('trainX', trainX, 'trainy', trainy)
#     # fit model
#     model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=20)
#     model.fit(trainX, trainy)
#     # make a one-step prediction
#     yhat = model.predict(np.asarray([testX]))
#     return yhat[0]
#
# def walk_forward_validation(train, test):
#     predictions = list()
#     train = train.values
#     history = [x for x in train]
#     # print('history', history)
#     for i in range(len(test)):
#         testX, testy = test.iloc[i, :-1], test.iloc[i, -1]
#         # print('i', i, testX, testy)
#         yhat = xgboost_forecast(history, testX)
#         predictions.append(yhat)
#         history.append(test.iloc[i, :])
#         print(i+1, '>expected=%.6f, predicted=%.6f' % (testy, yhat))
#     return test.iloc[:, -1],predictions






# # ============================
# #  model.py (FINAL FIXED VERSION)
# #  - Fully Keras 3 compatible
# #  - Supports ATTENTION / TCN / NBEATS
# #  - Includes missing xgb_scheduler() and related functions
# # ============================
#
# from keras.layers import (
#     Input, Dense, LSTM, Conv1D, Dropout, Bidirectional,
#     Multiply, Permute, RepeatVector, Flatten
# )
# from keras.models import Model, Sequential
# import keras.backend as K
# import tensorflow as tf
# import numpy as np
# import xgboost as xgb
# from utils import *
#
#
# # =========================================================
# # 1. ATTENTION BLOCK (Keras 3 Compatible)
# # =========================================================
# def attention_block(inputs):
#     hidden_size = int(inputs.shape[2])
#
#     score = Dense(1, activation='tanh')(inputs)
#
#     score = tf.keras.layers.Lambda(
#         lambda x: tf.nn.softmax(x, axis=1)
#     )(score)
#
#     score = tf.keras.layers.Lambda(
#         lambda x: tf.repeat(x, repeats=hidden_size, axis=2)
#     )(score)
#
#     return Multiply()([inputs, score])
#
#
# def attention_model(INPUT_DIMS=7, TIME_STEPS=20, HIDDEN=128):
#     inputs = Input(shape=(TIME_STEPS, INPUT_DIMS))
#
#     lstm_out = LSTM(HIDDEN, return_sequences=True)(inputs)
#
#     att_out = attention_block(lstm_out)
#
#     time_pool = tf.keras.layers.GlobalAveragePooling1D()(att_out)
#
#     output = Dense(1)(time_pool)
#     return Model(inputs, output)
#
#
# # =========================================================
# # 2. TCN Model
# # =========================================================
# class TCNBlock(tf.keras.Model):
#     def __init__(self, filters, kernel, dilation):
#         super().__init__()
#         self.conv = Conv1D(filters, kernel, dilation_rate=dilation,
#                            padding="causal", activation="relu")
#         self.norm = tf.keras.layers.BatchNormalization()
#         self.drop = Dropout(0.2)
#
#     def call(self, x):
#         y = self.conv(x)
#         y = self.norm(y)
#         return self.drop(y)
#
#
# def build_TCN_model(INPUT_DIMS=7, TIME_STEPS=20):
#     inputs = Input(shape=(TIME_STEPS, INPUT_DIMS))
#     x = inputs
#
#     for d in [1, 2, 4, 8, 16]:
#         x = TCNBlock(filters=64, kernel=3, dilation=d)(x)
#
#     x = tf.keras.layers.GlobalAveragePooling1D()(x)
#     x = Dense(64, activation='relu')(x)
#     outputs = Dense(1)(x)
#
#     return Model(inputs, outputs)
#
#
# # =========================================================
# # 3. N-BEATS Model
# # =========================================================
# def nbeats_block(x, units):
#     fc1 = Dense(units, activation="relu")(x)
#     fc2 = Dense(units, activation="relu")(fc1)
#
#     backcast = Dense(x.shape[-1])(fc2)
#     forecast = Dense(1)(fc2)
#     return backcast, forecast
#
#
# def build_NBEATS_model(INPUT_DIMS=7, TIME_STEPS=20, units=256, blocks=3):
#     inputs = Input(shape=(TIME_STEPS, INPUT_DIMS))
#     x = Flatten()(inputs)
#
#     residual = x
#     forecasts = []
#
#     for _ in range(blocks):
#         backcast, forecast = nbeats_block(residual, units)
#         residual = residual - backcast
#         forecasts.append(forecast)
#
#     output = tf.keras.layers.Add()(forecasts)
#     return Model(inputs, output)
#
#
# # =========================================================
# # 4. Unified Model Builder
# # =========================================================
# def build_model(model_type, INPUT_DIMS=7, TIME_STEPS=20):
#     model_type = model_type.upper()
#
#     if model_type == "ATTENTION":
#         return attention_model(INPUT_DIMS, TIME_STEPS)
#     elif model_type == "TCN":
#         return build_TCN_model(INPUT_DIMS, TIME_STEPS)
#     elif model_type == "NBEATS":
#         return build_NBEATS_model(INPUT_DIMS, TIME_STEPS)
#     else:
#         raise ValueError(f"Unknown model type: {model_type}")
#
#
# # =========================================================
# # 5. FIXED PredictWithData (added model_type & xgb scheduler)
# # =========================================================
# def PredictWithData(data, data_yuan, name, modelname,
#                     INPUT_DIMS=13, TIME_STEPS=20, model_type="ATTENTION"):
#
#     print("Using model:", model_type)
#
#     yindex = data.columns.get_loc(name)
#
#     data = np.array(data, dtype='float64')
#     data, normalize = NormalizeMult(data)
#
#     data_y = data[:, yindex].reshape(-1, 1)
#
#     testX, _ = create_dataset(data)
#     _, testY = create_dataset(data_y)
#
#     model = build_model(model_type, INPUT_DIMS, TIME_STEPS)
#     model.load_weights(modelname)
#
#     y_hat = model.predict(testX)
#
#     # Hybrid prediction with ARIMA + XGB residual scheduler
#     testY, y_hat = xgb_scheduler(data_yuan.copy(), y_hat)
#     return y_hat, testY
#
#
# # =========================================================
# # 6. (RESTORED) HYBRID SCHEDULING FUNCTIONS
# # =========================================================
#
# def xgb_scheduler(data, y_hat):
#     """
#     Schedules XGBoost predictions based on ARIMA residuals
#     """
#     close = data.pop('close')
#     data.insert(5, 'close', close)
#
#     train, test = prepare_data(data, n_test=len(y_hat), n_in=6, n_out=1)
#
#     testY, y_hat2 = walk_forward_validation(train, test)
#     return testY, y_hat2
#
#
# def xgboost_forecast(train, testX):
#     train = np.asarray(train)
#     trainX, trainy = train[:, :-1], train[:, -1]
#
#     model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=20)
#     model.fit(trainX, trainy)
#
#     yhat = model.predict(np.asarray([testX]))
#     return yhat[0]
#
#
# def walk_forward_validation(train, test):
#     predictions = []
#     train = train.values
#     history = [x for x in train]
#
#     for i in range(len(test)):
#         testX, testy = test.iloc[i, :-1], test.iloc[i, -1]
#         yhat = xgboost_forecast(history, testX)
#         predictions.append(yhat)
#         history.append(test.iloc[i, :])
#
#     return test.iloc[:, -1], predictions







# ============================
#  model.py (FINAL + CNN-TCN + TCN-Attention)
#  - Keras 3 兼容
#  - 支持:
#       ATTENTION
#       TCN
#       NBEATS
#       CNN_TCN
#       TCN_ATTENTION
#  - 兼容原来的 PredictWithData + xgb_scheduler
# ============================

from keras.layers import (
    Input, Dense, LSTM, Conv1D, Dropout, Bidirectional,
    Multiply, Permute, RepeatVector, Flatten
)
from keras.models import Model, Sequential
import keras.backend as K
import tensorflow as tf
import numpy as np
import xgboost as xgb
from utils import *
from tensorflow.keras.layers import Input, LSTM, Dense


# =========================================================
# 1. ATTENTION BLOCK (Keras 3 Compatible)
# =========================================================
# def attention_block(inputs):
#     """
#     输入: (batch, time_steps, hidden_size)
#     输出: 同形状, 每个时间步乘以注意力权重
#     """
#     hidden_size = int(inputs.shape[2])
#
#     # (batch, time_steps, 1)
#     score = Dense(1, activation='tanh')(inputs)
#
#     # softmax over time dimension
#     score = tf.keras.layers.Lambda(
#         lambda x: tf.nn.softmax(x, axis=1)
#     )(score)
#
#     # repeat to hidden_size: (batch, time_steps, hidden_size)
#     score = tf.keras.layers.Lambda(
#         lambda x: tf.repeat(x, repeats=hidden_size, axis=2)
#     )(score)
#
#     # element-wise multiply
#     return Multiply()([inputs, score])

from tensorflow.keras.layers import Dense, Softmax, Multiply, Lambda
import tensorflow as tf

def attention_block(inputs, name_prefix="time_att"):
    """
    标准时间维注意力：
    inputs:  (batch, time, hidden)
    return:  (batch, hidden)
    全程只用 Keras 的 Layer（Dense / Softmax / Multiply / Lambda），
    避免直接用 tf.nn.softmax + KerasTensor 触发报错。
    """
    # 1) 每个时间步打一个 score，形状 (batch, time, 1)
    score = Dense(1, name=f"{name_prefix}_score")(inputs)

    # 2) 在 time 维度做 softmax，得到注意力权重 (batch, time, 1)
    weights = Softmax(axis=1, name=f"{name_prefix}_softmax")(score)

    # 3) 按权重加权原始特征，(batch, time, hidden)
    weighted = Multiply(name=f"{name_prefix}_weighted")([inputs, weights])

    # 4) 对时间维做加权求和，得到上下文向量 (batch, hidden)
    context = Lambda(
        lambda x: tf.reduce_sum(x, axis=1),
        name=f"{name_prefix}_context"
    )(weighted)

    return context


# def attention_block(inputs, name_prefix="att"):
#     """
#     标准的时间维注意力池化：
#     inputs: (batch, time, hidden)
#     输出:   (batch, hidden)
#     """
#     # (batch, time, 1) —— 每个时间步一个 score
#     score = Dense(1, name=f"{name_prefix}_score")(inputs)
#
#     # 在 time 维度做 softmax 得到权重
#     weights = tf.nn.softmax(score, axis=1, name=f"{name_prefix}_softmax")  # (batch, time, 1)
#
#     # 加权求和得到上下文向量 (batch, hidden)
#     context = tf.reduce_sum(inputs * weights, axis=1, name=f"{name_prefix}_context")
#
#     return context

from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.models import Model

def attention_model(INPUT_DIMS=7, TIME_STEPS=20, HIDDEN=128):
    """
    修正版 Attention-LSTM 模型：
    输入:  (batch, TIME_STEPS, INPUT_DIMS)
    流程:  LSTM(return_sequences=True) → attention_block → Dense(1)
    输出:  (batch, 1)
    """
    inputs = Input(shape=(TIME_STEPS, INPUT_DIMS))

    # LSTM 输出整个时间序列 (batch, time, hidden)
    lstm_out = LSTM(HIDDEN, return_sequences=True)(inputs)

    # 时间维注意力池化，得到上下文 (batch, hidden)
    context = attention_block(lstm_out, name_prefix="time_att")

    # 最后映射到标量预测
    output = Dense(1, name="pred_dense")(context)

    model = Model(inputs, output, name="Attention_LSTM")

    # 如果你在外面统一 compile，就可以不在这里 compile；
    # 如果你习惯在这里 compile，可以加上：
    # model.compile(optimizer="adam", loss="mse")

    return model

# def attention_model(INPUT_DIMS=7, TIME_STEPS=20, HIDDEN=128):
#     """
#     修正后的 Attention-LSTM 模型：
#     LSTM → Attention Pooling → Dense(1)
#     """
#     inputs = Input(shape=(TIME_STEPS, INPUT_DIMS))
#
#     # 返回整个时间序列
#     lstm_out = LSTM(HIDDEN, return_sequences=True)(inputs)
#
#     # 注意力池化，得到 (batch, HIDDEN)
#     context = attention_block(lstm_out, name_prefix="time_att")
#
#     # 输出单步预测
#     output = Dense(1)(context)
#
#     model = Model(inputs, output)
#
#     # 是否在这里 compile 看你原来训练代码怎么写
#     # 比如：
#     # model.compile(optimizer="adam", loss="mse")
#
#     return model



# =========================================================
# 2. TCN 基础模块
# =========================================================
class TCNBlock(tf.keras.Model):
    def __init__(self, filters, kernel, dilation):
        super().__init__()
        self.conv = Conv1D(
            filters,
            kernel,
            dilation_rate=dilation,
            padding="causal",
            activation="relu"
        )
        self.norm = tf.keras.layers.BatchNormalization()
        self.drop = Dropout(0.2)

    def call(self, x):
        y = self.conv(x)
        y = self.norm(y)
        return self.drop(y)


def build_TCN_model(INPUT_DIMS=7, TIME_STEPS=20):
    """
    纯 TCN 模型
    """
    inputs = Input(shape=(TIME_STEPS, INPUT_DIMS))
    x = inputs

    for d in [1, 2, 4, 8, 16]:
        x = TCNBlock(filters=64, kernel=3, dilation=d)(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(1)(x)

    return Model(inputs, outputs)


# =========================================================
# 3. N-BEATS Model
# =========================================================
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, LayerNormalization, Subtract, Add
from tensorflow.keras.models import Model

def build_NBEATS_model(INPUT_DIMS=7,
                       TIME_STEPS=20,
                       hidden=64,
                       blocks=4,
                       weight_decay=1e-4):
    """
    更稳定的 N-BEATS 实现：
    - 输入: (TIME_STEPS, INPUT_DIMS)
    - 先 Flatten 得到长度 D = TIME_STEPS * INPUT_DIMS 的向量
    - 每个 block: 4 层 FC + LayerNorm，输出 backcast (D) 和 forecast (1)
    - residual = residual - backcast
    - 最终 forecast 为所有 block forecast 的求和
    """

    inputs = Input(shape=(TIME_STEPS, INPUT_DIMS), name="nbeats_input")

    # 1. Flatten 过去窗口
    x0 = Flatten(name="flatten_input")(inputs)          # 形状: (None, D)
    residual = x0                                       # 初始 residual
    input_dim = int(x0.shape[-1])                       # D

    forecasts = []

    for b in range(blocks):
        x = residual
        # 2. 4 层全连接 + ReLU
        for i in range(4):
            x = Dense(
                hidden,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                name=f"block{b}_fc{i+1}"
            )(x)

        # 3. LayerNorm 稍微稳一下分布
        x = LayerNormalization(name=f"block{b}_ln")(x)

        # 4. backcast & forecast
        backcast = Dense(
            input_dim,
            kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
            name=f"block{b}_backcast"
        )(x)                                            # (None, D)

        forecast = Dense(
            1,
            kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
            name=f"block{b}_forecast"
        )(x)                                            # (None, 1)

        # 5. residual 更新: residual = residual - backcast
        residual = Subtract(name=f"block{b}_residual")([residual, backcast])
        forecasts.append(forecast)

    # 6. 所有 block 的 forecast 求和
    if len(forecasts) > 1:
        output = Add(name="forecast_sum")(forecasts)
    else:
        output = forecasts[0]

    model = Model(inputs, output, name="NBEATS")

    return model


# =========================================================
# 4. CNN-TCN 混合结构
# =========================================================
def build_CNN_TCN_model(INPUT_DIMS=7, TIME_STEPS=20):
    """
    CNN + TCN:
    - 前几层 CNN 提取局部形态特征
    - 后面用 TCN 扩大感受野并建模长程依赖
    """
    inputs = Input(shape=(TIME_STEPS, INPUT_DIMS))

    # CNN 部分
    x = Conv1D(32, kernel_size=3, padding="causal", activation="relu")(inputs)
    x = Conv1D(32, kernel_size=3, padding="causal", activation="relu")(x)
    x = Dropout(0.2)(x)

    # TCN 部分
    for d in [1, 2, 4, 8]:
        x = TCNBlock(filters=64, kernel=3, dilation=d)(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = Dense(64, activation="relu")(x)
    outputs = Dense(1)(x)

    return Model(inputs, outputs)


# =========================================================
# 5. TCN + Attention 混合结构
# =========================================================
def build_TCN_Attention_model(INPUT_DIMS=7, TIME_STEPS=20):
    """
    TCN + Attention:
    - TCN 先提取时间序列特征 (保留时间维)
    - 在 TCN 输出上做时序注意力, 强调关键时间步
    """
    inputs = Input(shape=(TIME_STEPS, INPUT_DIMS))

    x = inputs
    for d in [1, 2, 4, 8]:
        x = TCNBlock(filters=64, kernel=3, dilation=d)(x)
        # TCNBlock 输出 shape: (batch, time_steps, filters)

    # 在 TCN 输出上做注意力
    att_out = attention_block(x)     # (batch, time_steps, hidden)

    # x = tf.keras.layers.GlobalAveragePooling1D()(att_out)
    x = att_out
    x = Dense(64, activation="relu")(x)
    outputs = Dense(1)(x)

    return Model(inputs, outputs)


# =========================================================
# 6. Unified Model Builder
# =========================================================
def build_model(model_type, INPUT_DIMS=7, TIME_STEPS=20):
    """
    model_type 可选:
        "ATTENTION"
        "TCN"
        "NBEATS"
        "CNN_TCN"
        "TCN_ATTENTION"
    """
    model_type = model_type.upper()

    if model_type == "ATTENTION":
        return attention_model(INPUT_DIMS, TIME_STEPS)
    elif model_type == "TCN":
        return build_TCN_model(INPUT_DIMS, TIME_STEPS)
    elif model_type == "NBEATS":
        return build_NBEATS_model(INPUT_DIMS, TIME_STEPS)
    elif model_type == "CNN_TCN":
        return build_CNN_TCN_model(INPUT_DIMS, TIME_STEPS)
    elif model_type == "TCN_ATTENTION":
        return build_TCN_Attention_model(INPUT_DIMS, TIME_STEPS)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# =========================================================
# 7. PredictWithData（带 ARIMA+XGB 混合）
# =========================================================
# def PredictWithData(data2, data_yuan, dimname, modelname,
#                     INPUT_DIMS=7, TIME_STEPS=20, model_type="TCN"):
#
#     # 加载训练时保存的 normalize
#     # normalize = np.load("stock_normalize.npy", allow_pickle=True)
#
#     # 归一化 data2
#     # data2_norm = NormalizeMultUseData(np.array(data2), normalize)
#     data2_norm = NormalizeMult(np.array(data2))
#
#     # 提取目标列
#     yindex = data2.columns.get_loc(dimname)
#     y2_norm = data2_norm[:, yindex].reshape(-1, 1)
#
#     # 构造测试窗口
#     testX, _ = create_dataset(data2_norm, TIME_STEPS)
#     _, testY_norm = create_dataset(y2_norm, TIME_STEPS)
#
#     # 构建同结构模型并加载权重
#     model = build_model(model_type, INPUT_DIMS, TIME_STEPS)
#     model.load_weights(modelname)
#
#     # 预测（归一化空间）
#     y_hat_norm = model.predict(testX)
#
#     # 反归一化 close 列（index=3）
#     y_hat = FNormalizeMult(y_hat_norm, normalize[3])
#     testY = FNormalizeMult(testY_norm, normalize[3])
#
#     return y_hat.flatten(), testY.flatten()
import numpy as np
from tensorflow.keras.models import Model

def PredictWithData(
    data2,          # DataFrame：测试数据
    data_yuan,      # 原始数据（如果你画图/对比可以用，不用也没关系）
    dimname,        # 目标列名，比如 'Close'
    modelname,      # 模型权重文件路径
    INPUT_DIMS=7,
    TIME_STEPS=20,
    model_type="TCN"
):
    """
    使用与训练阶段一致的归一化参数进行预测，
    并将预测值和真实值一起反归一化后返回。
    """

    # 1. 加载训练时保存的归一化参数 (形状: [num_features, 2])
    normalize = np.load("stock_normalize.npy", allow_pickle=True)
    # print("# normalize:", normalize)  # 调试用，可以注释掉

    # 2. 使用训练时的 normalize 对测试数据做归一化
    data2_np = np.array(data2, dtype=np.float32)
    # 注意：这里要用和训练时同一个函数
    data2_norm = NormalizeMultUseData(data2_np, normalize)

    # 3. 提取目标列 index
    yindex = data2.columns.get_loc(dimname)

    # 4. 构造测试窗口
    #   testX: (N, TIME_STEPS, INPUT_DIMS)
    testX, _ = create_dataset(data2_norm, TIME_STEPS)

    #   目标列归一化后的序列
    y_norm_col = data2_norm[:, yindex].reshape(-1, 1)
    _, testY_norm = create_dataset(y_norm_col, TIME_STEPS)  # (N, 1)

    print(testX.shape, testY_norm.shape)  # 调试用，确认形状

    # 5. 构建同结构模型并加载权重
    model = build_model(model_type, INPUT_DIMS, TIME_STEPS)
    model.load_weights(modelname)

    # 6. 预测（归一化空间）
    y_hat_norm = model.predict(testX)  # (N, 1)

    # 7. 反归一化目标列
    #    关键修正：取出 yindex 对应的归一化参数，但保持二维形状 (1, 2)
    norm_single = normalize[yindex:yindex + 1]   # slice 而不是 normalize[yindex]

    y_hat = FNormalizeMult(y_hat_norm, norm_single)    # (N, 1) -> (N, 1)
    testY = FNormalizeMult(testY_norm, norm_single)    # (N, 1) -> (N, 1)

    return y_hat.flatten(), testY.flatten()





# =========================================================
# 8. HYBRID SCHEDULING FUNCTIONS (ARIMA + XGBoost)
# =========================================================
def xgb_scheduler(data, y_hat):
    """
    使用 ARIMA 残差 + XGBoost 进行混合预测
    """
    close = data.pop('close')
    data.insert(5, 'close', close)

    train, test = prepare_data(data, n_test=len(y_hat), n_in=6, n_out=1)
    testY, y_hat2 = walk_forward_validation(train, test)
    return testY, y_hat2


def xgboost_forecast(train, testX):
    train = np.asarray(train)
    trainX, trainy = train[:, :-1], train[:, -1]

    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=20)
    model.fit(trainX, trainy)

    yhat = model.predict(np.asarray([testX]))
    return yhat[0]


def walk_forward_validation(train, test):
    predictions = []
    train = train.values
    history = [x for x in train]

    for i in range(len(test)):
        testX, testy = test.iloc[i, :-1], test.iloc[i, -1]
        yhat = xgboost_forecast(history, testX)
        predictions.append(yhat)
        history.append(test.iloc[i, :])

    return test.iloc[:, -1], predictions
