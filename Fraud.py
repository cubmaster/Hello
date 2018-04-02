import numpy as np

import tensorflow as tf

import pandas as pa
from Helpers.DataHelpers import *
from Helpers.training import *

# get data
data = pa.read_csv('./Data/syn_fraud.csv')


# print(correlation(data, True))
# show_null_by_column(data)

xtrain, xtest, ytrain, ytest = splitdata(data,
                                             ['step',
                                              'typeID',
                                              'amount',
                                              'oldbalanceOrg',
                                              'newbalanceOrig',
                                              'oldbalanceDest',
                                              'newbalanceDest',
                                              'idOrig',
                                              'idDest'],
                                             ['isFraud'],
                                             0.30)


# Build Model
x = tf.placeholder(dtype=tf.dtypes.float32,shape=[none,len(xtrain)],name="x")
b = tf.Variable(dtype=tf.dtypes.float32,name="b")
m = tf.Variable(dtype=tf.dtypes.float32,name="m")

y = tf.nn.softmax(tf.matmul(x, m)+b)

y_ = tf.placeholder(dtype=tf.dtypes.float32,shape=[None,len(ytrain)])

# Solve for

errors = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=_y,logits=y))


model = tf.train.GradientDescentOptimizer(0.5).minimize(errors)

sess = tf.Session(tf.global_variables_initializer())

with sess:
    sess.run(model, feed_dict={x: xtrain, y_: ytrain})



print("complete")