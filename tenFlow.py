# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import pandas as pd
from keras import backend as K
import matplotlib.pyplot as plt
import logging


logging.basicConfig(format='%(asctime)-15s '   )

# 定义一个创建层的函数 
def create_layer(X,n,activation):
    ndim = int(X.shape[0])
    stddev = 2/ np.sqrt(ndim)
    initialization = tf.truncated_normal((n,ndim),stddev = stddev)
    W = tf.Variable(initialization )
    b = tf.Variable(tf.zeros([n,1]))
    Z = tf.matmul(W,X)+b
    return activation(Z)

# 使用创建的层的函数来搭建网络
n_dim = 784
n1 = 600
n2 = 600
n3 = 600
n4 = 600
n_output = 10

X = tf.placeholder(tf.float32,[n_dim,None])
Y = tf.placeholder(tf.float32,[10,None])

learning_rate = tf.placeholder(tf.float32,shape=[])

hidden1 = create_layer(X,n1,activation = tf.nn.relu)
hidden2 = create_layer(hidden1,n2,activation = tf.nn.relu)
hidden3 = create_layer(hidden2,n3,activation = tf.nn.relu)
hidden4 = create_layer(hidden3,n4,activation = tf.nn.relu)
outputs = create_layer(hidden4,n_output,activation = tf.identity)

y_ = tf.nn.softmax(outputs)

cost = -tf.reduce_mean(Y*tf.log(y_)+(1-Y)*tf.log(1-y_))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# 定义个model函数来运行模型
def model(process,minibatch,training_epochs,features,classes,logging_step = 100,learning_r=0.001):
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    cost_history = []
    
    for epoch in range(training_epochs):
        for i in range(0,features.shape[1],minibatch):
            X_train_mini = features[:,i:i+minibatch]
            y_train_mini = classes[:,i:i+minibatch]
            
            sess.run(optimizer,feed_dict={X:X_train_mini,Y:y_train_mini,learning_rate:learning_r})
            
        cost_ = sess.run(cost,feed_dict={X:features,Y:classes,learning_rate:learning_r})
        
        cost_history = np.append(cost_history,cost_)
        
        if (epoch % logging_step == 0):
            print(process,' Reached epoch ',epoch,'cost J = ',cost_)
    
    # 增加精度打印
    correct_predictions = tf.equal(tf.argmax(y_,0),tf.argmax(Y,0))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions,'float'))
    print('Accuracy: ',accuracy.eval({X:features,Y:classes,learning_rate:learning_r},session=sess),' with ',epoch,' epochs')
    
    # plot
    plt.figure()
    plt.plot(cost_history)
    plt.title(process+' Datasets:'+str(features.shape[1])+' batch size:'+str(minibatch)+' Final Accuracy: '+str(accuracy.eval({X:features,Y:classes,learning_rate:learning_r},session=sess)))
    plt.xlabel('epoch')
    plt.ylabel('cost')
    plt.savefig(process+'.jpg')
            
    return sess,cost_history

# 寻找数据做好预处理，调用训练模型就可以啦

# 训练数据与标签 
train = pd.read_csv('fashion-mnist_train.csv')
labels = train['label'].values.reshape(1,60000)  # 根据lable键值提取标签 
train_ = train.drop('label',axis =1).transpose() # 在训练数据中不需要包含标签列，将其删除，其他保留，并转换为张量 
trainX = np.array(train_/255.0)  # 归一化

# 训练标签转化为softmax分类器的one-hot编码
labels_ = np.zeros((60000,10))
labels_[np.arange(60000),labels] = 1
labels_ = labels_.transpose()
labels_ = np.array(labels_)


# 测试数据与标签 
test = pd.read_csv('fashion-mnist_test.csv')
labels_dev = test['label'].values.reshape(1,10000) #根据label键值提取标签 
test_ = test.drop('label',axis =1).transpose()
testX = np.array(test_/255.0)

# 测试标签转为为softmax分类器的one-hot编码 
labels_dev_ = np.zeros((10000,10))               
labels_dev_[np.arange(10000),labels_dev] = 1
labels_dev_ = labels_dev_.transpose()
labels_dev_ = np.array(labels_dev_)

'''
correcttion_predictions = tf.equal(tf.argmax(y_,0),tf.argmax(Y,0))
accuracy = tf.reduce_mean(tf.cast(correcttion_predictions,'float'))
print('Accuracy:',accuracy.eval({X:,Y:,learning_rate:},session=sess))
'''

logger = logging.getLogger().warning('',extra={'clientip': '', 'user': '  '})
train_sess,train_history = model(r'Train',minibatch=50,training_epochs=6,features=trainX,classes=labels_,logging_step = 2,learning_r=0.01)

logger = logging.getLogger().warning('',extra={'clientip': '', 'user': '  '})

test_sess,test_history = model(r'Test',minibatch=50,training_epochs=6,features=testX,classes=labels_dev_,logging_step = 2,learning_r=0.01)
logger = logging.getLogger().warning('',extra={'clientip': '', 'user': '  '})



            
            
            
            
            
            
            
            
            
            
            
            
            
            