import tushare as ts
import numpy as np
import pandas as pd
import math
from matplotlib import pyplot as plt
from skimage import data
import tensorflow as tf
ts.set_token('8f1eecb8f91731fe2d35be60bc2611087aa9730ddf05562e99907431')
pro = ts.pro_api()
df = ts.pro_bar(ts_code='600519.SZ', start_date='20080701', end_date='20180718')# freq='5MIN')
df = df.dropna()

# import baostock as bs
# import pandas as pd
#
# #### 登陆系统 ####
# lg = bs.login()
# # 显示登陆返回信息
# print('login respond error_code:'+lg.error_code)
# print('login respond  error_msg:'+lg.error_msg)
#
# #### 获取沪深A股历史K线数据 ####
# # 详细指标参数，参见“历史行情指标参数”章节；“分钟线”参数与“日线”参数不同。
# df = bs.query_history_k_data_plus("sh.601398",
#     "date,code,open,high,low,close,preclose,volume,turn",
#     start_date='2008-07-01', end_date='2019-5-17',
#     frequency="d", adjustflag="3")
# print('query_history_k_data_plus respond error_code:'+df.error_code)
# print('query_history_k_data_plus respond  error_msg:'+df.error_msg)
#
# #### 打印结果集 ####
# data_list = []
# while (df.error_code == '0') & df.next():
#     # 获取一条记录，将记录合并在一起
#     data_list.append(df.get_row_data())
# df = pd.DataFrame(data_list, columns=df.fields)
#
# #### 结果集输出到csv文件 ####
# df.to_csv("/home/cqiuac/worldquant_project/history_A_stock_k_data.csv", index=False)
# print(df)
#
# #### 登出系统 ####
# bs.logout()
#
# a = np.arange(1,len(df['open'])+1)
# df.insert(2,'index',a)
# df.fillna(0)
# print(df.columns.values.tolist())
# data= df.values
# data = data[:,2:]
# for i in range(data.shape[0]):
#     for j in range(data.shape[1]):
#         # print([i,j])
#         if data[i,j] == '':
#             data[i, j]=0
#         data[i,j] = float(data[i,j])
# data.astype(float)

# #data visualization
# plt.figure(figsize = (18,9))
# #plt.plot(range(df.shape[0]),(df['low']+df['high'])/2.0)
# plt.plot(range(df['trade_time'],(df['low']+df['high'])/2.0)
# #plt.xticks(range(0,df.shape[0],500),df['trade_time'].loc[::500],rotation=45)
# # plt.xlabel('Date',fontsize=18)
# # plt.ylabel('Mid Price',fontsize=18)
# plt.show()
# plt.savefig('/home/cqiuac/worldquant_project/raw_data')

#df.to_csv('/home/cqiuac/000159.csv')

#df=pd.read_csv('/home/cqiuac/000159.csv')
rnn_unit = 10       #hidden layer level
input_size = 6
output_size = 1
lr = 0.001         #learning rate

n = len(data)
train_ratio = 0.8   #decide the data ratio as training data
save_name ='.model.ckpt'
cpt_name ='/home/cqiuac/'



#generating the standardized training data
def get_train_data(batch_size=60, time_step=20, train_begin=0, train_end=2000):#float(n*train_ratio)):
    batch_index=[]
    data_train=data[train_begin:train_end]
    data_train = np.array(data_train, dtype=np.float32)
    normalized_train_data=(data_train-np.mean(data_train,axis=0))/np.std(data_train,axis=0)
    train_x,train_y=[],[]
    for i in range(len(normalized_train_data)-time_step):
        if i % batch_size==0:
            batch_index.append(i)
        x = normalized_train_data[i:i+time_step,:input_size]
        y = normalized_train_data[i:i+time_step,input_size,np.newaxis]
        train_x.append(x.tolist())
        train_y.append(y.tolist())
    batch_index.append((len(normalized_train_data)-time_step))
    return batch_index,train_x,train_y

#generating the standardized test data
def get_test_data(time_step=20, test_begin=2001):#float(n*train_ratio)):
    data_test=data[test_begin:]
    data_test = np.array(data_test, dtype=np.float64)
    mean=np.mean(data_test,axis=0)
    std=np.std(data_test,axis=0)
    normalized_test_data=(data_test-mean)/std
    size=(len(normalized_test_data)+time_step-1)//time_step
    test_x,test_y=[],[]
    for i in range(size-1):
        x=normalized_test_data[i*time_step:(i+1)*time_step,:input_size]
        y=normalized_uytest_data[i*time_step:(i+1)*time_step,input_size]
        test_x.append(x.tolist())
        test_y.extend(y)
        test_x.append((normalized_test_data[(i+1)*time_step:,:input_size]).tolist())
        test_y.extend((normalized_test_data[(i+1)*time_step:,input_size]).tolist())
    return mean,std,test_x,test_y

#setting the weights and biases
weights={
         'in':tf.Variable(tf.random_normal([input_size,rnn_unit])),
         'out':tf.Variable(tf.random_normal([rnn_unit,1]))
        }
biases={
        'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),
        'out':tf.Variable(tf.constant(0.1,shape=[1,]))
       }

#set the LSTM model
def lstm(X):
    batch_size = tf.shape(X)[0]
    time_step = tf.shape(X)[1]
    w_in = weights['in']
    b_in = biases['in']
    input = tf.reshape(X, [-1, input_size])
    input_rnn = tf.matmul(input, w_in) + b_in
    input_rnn = tf.reshape(input_rnn, [-1, time_step, rnn_unit])
    cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state, dtype=tf.float32)
    output = tf.reshape(output_rnn, [-1, rnn_unit])
    w_out = weights['out']
    b_out = biases['out']
    pred = tf.matmul(output, w_out) + b_out
    return pred, final_states

#generating the training function
def train_lstm(save_name, batch_size=60, time_step=20, train_begin=0, train_end=2000,iteration=10):#float(n*train_ratio), iteration=10):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size])
    batch_index,train_x,train_y=get_train_data(batch_size, time_step, train_begin, train_end)
    with tf.variable_scope("sec_lstm"):
        pred,_=lstm(X)
    loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1])))
    train_op=tf.train.AdamOptimizer(lr).minimize(loss)
    saver=tf.train.Saver(tf.global_variables(),max_to_keep=15)
    #module_file = tf.train.latest_checkpoint()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #saver.restore(sess, module_file)
        for i in range(iteration):
            for step in range(len(batch_index)-1):
                _,loss_=sess.run([train_op,loss],feed_dict={X:train_x[batch_index[step]:batch_index[step+1]],Y:train_y[batch_index[step]:batch_index[step+1]]})
                print("Number of iterations:",i," loss:",loss_)
        print("model_save: ",saver.save(sess,save_name))
        print("The train has finished")


#run the training
train_lstm(save_name)


def prediction(time_step=1):
    X = tf.placeholder(tf.int32, shape=[None, time_step, input_size])
    mean, std, test_x, test_y = get_test_data(time_step)
    with tf.variable_scope("sec_lstm", reuse=tf.AUTO_REUSE):
        pred, _ = lstm(X)
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        # 参数恢复
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
        acc = np.average(np.abs(test_predict - test_y[:len(test_predict)]) / test_y[:len(test_predict)])  # 偏差程度
        print("The accuracy of this predict:", acc)
        # 以折线图表示结果
        plt.figure()
        plt.plot(list(range(len(test_predict))), test_predict, color='b', )
        plt.plot(list(range(len(test_y))), test_y, color='r')
        plt.show()
# prediction(cpt_name)

#plot function
def plot_result(x, y,label,save_dir):
    plt.figure()
    plt.plot(list(range(len(test_predict))), x, color='b',)
    plt.plot(list(range(len(test_y))), y, color='r')
    plt.savefig(save_dir)
    plt.show()

# #traading strategy
# def buy_signal(test_predict,test_y):
#     cost = 0.003
#     new_nv =[1]
#     sig = test_predict > 0
#     for i in range(len(sig)-1):
#         if (sig[i+1] == True) and (sig[i] == True):
#             new_nv.append(new_nv[-1]*(1+test_y[i]))
#         elif (sig[i+1] == True) and (sig[i] == False):
#             new_nv.append(new_nv[-1]*(1-cost))
#         elif (sig[i+1] == False) and (sig[i] == False):
#             new_nv.append(new_nv[-1])
#         elif (sig[i+1] == False) and (sig[i] == True):
#             new_nv.append(new_nv[-1]*(1+test_y[i])*(1-cost))
#     return new_nv
#
#Prediction result
test_predict,test_y = prediction()
plot_result(test_predict, test_y,['predict return','test return'],'/home/cqiuac/worldquant_project/result1_1.png')

# ynv = [1]
# prenv = [1]
# def fit_nv(test_predict,test_y):
#     for i in range(len(test_predict)):
#         ynv.append(ynv[-1]*(1+test_y[i]))
#         print(ynv[-1]*(1+test_y[i]))
#         prenv.append(prenv[-1]*(1+test_predict[i]))
#     plot_result(prenv,ynv,['predict net value','test net value'],'/home/cqiuac/worldquant_project/result1_2.png')
# fit_nv(test_predict,test_y)
#
# # accuracy of buying signal
# num = 0
# for i in range(len(test_y)):
#     if np.sign(test_predict[i])== np.sign(test_y[i]):
#         num = num + 1
# acc = num / len(test_y)
# print(acc)
