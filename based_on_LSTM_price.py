mport tushare as ts
import numpy as np
import pandas as pd
import math
from matplotlib import pyplot as plt
from skimage import data
import tensorflow as tf
import baostock as bs
import pandas as pd

#### 登陆系统 ####
lg = bs.login()
# 显示登陆返回信息
print('login respond error_code:'+lg.error_code)
print('login respond  error_msg:'+lg.error_msg)

#### 获取沪深A股历史K线数据 ####
# 详细指标参数，参见“历史行情指标参数”章节；“分钟线”参数与“日线”参数不同。
df = bs.query_history_k_data_plus("sh.600019",
    "date,code,open,high,low,close,preclose,volume,turn",
    start_date='2008-07-01', end_date='2019-5-17',
    frequency="d", adjustflag="3")
print('query_history_k_data_plus respond error_code:'+df.error_code)
print('query_history_k_data_plus respond  error_msg:'+df.error_msg)


data_list = []
while (df.error_code == '0') & df.next():
    # 获取一条记录，将记录合并在一起
    data_list.append(df.get_row_data())
df = pd.DataFrame(data_list, columns=df.fields)

#### 结果集输出到csv文件 ####
df.to_csv("/home/cqiuac/worldquant_project/history_A_stock_k_data.csv", index=False)
print(df)

#### 登出系统 ####
bs.logout()

a = np.arange(1,len(df['open'])+1)
df.insert(2,'index',a)
df.fillna(0)
print(df.columns.values.tolist())
data= df.values
data = data[:,2:]
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        # print([i,j])
        if data[i,j] == '':
            data[i, j]=0
        data[i,j] = float(data[i,j])
data.astype(float)


rnn_unit = 10

lstm_layers = 2

input_size = 5

output_size = 1
lr = 0.001

save_name ='.model.ckpt'
cpt_name ='/home/cqiuac/'

def get_train_data(batch_size=60, time_step=20, train_begin=0, train_end=2000):
    batch_index = []
    data_train = data[train_begin:train_end]

    data_train = np.array(data_train, dtype=np.float32)
    normalized_train_data = (data_train - np.mean(data_train, axis=0)) / np.std(data_train, axis=0)

    train_x, train_y = [], []
    for i in range(len(normalized_train_data) - time_step):
        if i % batch_size == 0:
            batch_index.append(i)
        x = normalized_train_data[i:i + time_step, :5]

        y = normalized_train_data[i:i + time_step, 5, np.newaxis]
        train_x.append(x.tolist())
        train_y.append(y.tolist())
    batch_index.append((len(normalized_train_data) - time_step))
    return batch_index, train_x, train_y

def get_test_data(time_step=20, test_begin=2001):
    data_test = data[test_begin:]
    data_test = np.array(data_test, dtype=np.float64)
    mean = np.mean(data_test, axis=0)
    std = np.std(data_test, axis=0)
    normalized_test_data = (data_test - mean) / std
    size = (len(normalized_test_data) + time_step - 1) // time_step
    test_x, test_y = [], []
    for i in range(size - 1):
        x = normalized_test_data[i * time_step:(i + 1) * time_step, :5]
        y = normalized_test_data[i * time_step:(i + 1) * time_step, 5]
        test_x.append(x.tolist())
        test_y.extend(y)
        x = normalized_test_data[i * time_step:(i + 1) * time_step, :5]
        y = normalized_test_data[i * time_step:(i + 1) * time_step, 5]
        test_x.append(x.tolist())
        test_y.extend(y)
    test_x.append((normalized_test_data[(i + 1) * time_step:, :5]).tolist())
    test_y.extend((normalized_test_data[(i + 1) * time_step:, 5]).tolist())
    return mean, std, test_x, test_y

weights = {
    'in': tf.Variable(tf.random_normal([input_size, rnn_unit])),
    'out': tf.Variable(tf.random_normal([rnn_unit, 1]))
}
biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[rnn_unit, ])),
    'out': tf.Variable(tf.constant(0.1, shape=[1, ]))
}
keep_prob = tf.placeholder(tf.float32, name='keep_prob')


def lstmCell():

    basicLstm = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    # dropout
    drop = tf.nn.rnn_cell.DropoutWrapper(basicLstm, output_keep_prob=keep_prob)
    return basicLstm


def lstm(X):
    batch_size = tf.shape(X)[0]
    time_step = tf.shape(X)[1]
    w_in = weights['in']
    b_in = biases['in']
    input = tf.reshape(X, [-1, input_size])
    input_rnn = tf.matmul(input, w_in) + b_in
    input_rnn = tf.reshape(input_rnn, [-1, time_step, rnn_unit])
    cell = tf.nn.rnn_cell.MultiRNNCell([lstmCell() for i in range(lstm_layers)])
    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state, dtype=tf.float32)
    output = tf.reshape(output_rnn, [-1, rnn_unit])
    w_out = weights['out']
    b_out = biases['out']
    pred = tf.matmul(output, w_out) + b_out
    return pred, final_states



def train_lstm(save_name,batch_size=60, time_step=20, train_begin=2000, train_end=6500):
    X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
    Y = tf.placeholder(tf.float32, shape=[None, time_step, output_size])
    batch_index, train_x, train_y = get_train_data(batch_size, time_step, train_begin, train_end)
    with tf.variable_scope("sec_lstm"):
        pred, _ = lstm(X)
    loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(Y, [-1])))
    train_op = tf.train.AdamOptimizer(lr).minimize(loss)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(1000):
            for step in range(len(batch_index) - 1):
                _, loss_ = sess.run([train_op, loss], feed_dict={X: train_x[batch_index[step]:batch_index[step + 1]],
                                                                 Y: train_y[batch_index[step]:batch_index[step + 1]],
                                                                 keep_prob: 0.5})
            print("Number of iterations:", i, " loss:", loss_)
        print("model_save: ", saver.save(sess, save_name))
        print("The train has finished")

def prediction(time_step=1):
    X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
    mean, std, test_x, test_y = get_test_data(time_step)
    with tf.variable_scope("sec_lstm", reuse=tf.AUTO_REUSE):
        pred, _ = lstm(X)
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:

        module_file = tf.train.latest_checkpoint(cpt_name)
        saver.restore(sess, module_file)
        test_predict = []
        for step in range(len(test_x) - 1):
            prob = sess.run(pred, feed_dict={X: [test_x[step]], keep_prob: 1})
            predict = prob.reshape((-1))
            test_predict.extend(predict)
        test_y = np.array(test_y) * std[5] + mean[5]
        test_predict = np.array(test_predict) * std[5] + mean[5]
        # print(test_predict.tolist())
        acc = np.average(np.abs(test_predict - test_y[:len(test_predict)]) / test_y[:len(test_predict)])
        print("The accuracy of this predict:", acc)

        plt.figure()
        plt.plot(list(range(len(test_predict))), test_predict, color='b', )
        plt.plot(list(range(len(test_y))), test_y, color='r')
        plt.show()

train_lstm(save_name)
prediction()








