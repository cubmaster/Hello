import pandas as pd
import numpy as np
import tensorflow as tf
import time
from sklearn.metrics import classification_report,  accuracy_score

# https://www.kaggle.com/samkirkiles/credit-card-fraud/data
credit_card_data = pd.read_csv('./Data/creditcard.csv')

# 1  Randomize data
shuffled_data = credit_card_data.sample(frac=1)

# 2  Encode Data
one_hot_data = pd.get_dummies(shuffled_data, columns=['Class'])

# 3  Normalize - puts everything in terms of percentages of totals
normalized_data = (one_hot_data - one_hot_data.min()) / (one_hot_data.max() - one_hot_data.min())

# 4  split x/y
df_X = normalized_data.drop(['Class_0','Class_1'], axis=1)
df_y = normalized_data[['Class_1','Class_0']]


# 5  Data_frames -< numpy arrays for Tenserflow
ar_X = np.asarray(df_X.values, dtype='float32')
ar_y = np.asarray(df_y.values, dtype='float32')

# 6  Split data to test/train data
train_size = int(0.8 * len(ar_X))
(raw_X_train, raw_y_train) = (ar_X[:train_size], ar_y[:train_size])
(raw_X_test, raw_y_test) = (ar_X[train_size:], ar_y[train_size:])


# 7 eliminate bias (logit weighting)

count_legit, count_fraud = np.unique(credit_card_data['Class'], return_counts=True)[1]
fraud_ratio = float(count_fraud / (count_legit + count_fraud))
print('Percent of fraudulent transactions: ', fraud_ratio)

weighting = 1 / fraud_ratio

raw_y_train[:, 1] = raw_y_train[:, 1] * weighting

#8 Build computational Graph using TF

#Set up nodes and variables

input_dimensions = ar_X.shape[1]
output_dimensions = ar_y.shape[1]
num_layer_1_cells = 100
num_layer_2_cells = 150

X_train_node = tf.placeholder(tf.float32, [None, input_dimensions], name='X_train')
y_train_node = tf.placeholder(tf.float32, [None, output_dimensions], name='y_train')

X_test_node = tf.constant(raw_X_test, name='X_test')
y_test_node = tf.constant(raw_y_test, name='y_test')

weight_1_node = tf.Variable(tf.zeros([input_dimensions, num_layer_1_cells]),  name='weight_1')
biases_1_node = tf.Variable(tf.zeros([num_layer_1_cells]), name='biases_1')

weight_2_node = tf.Variable(tf.zeros([num_layer_1_cells, num_layer_2_cells]),  name='weight_2')
biases_2_node = tf.Variable(tf.zeros([num_layer_2_cells]), name='biases_2')

weight_3_node = tf.Variable(tf.zeros([num_layer_2_cells, output_dimensions]),  name='weight_3')
biases_3_node = tf.Variable(tf.zeros([output_dimensions]), name='biases_3')

# build functions

def network(input_tensor):
    layer1 = tf.nn.sigmoid(tf.matmul(input_tensor, weight_1_node) + biases_1_node)
    layer2 = tf.nn.dropout(tf.nn.sigmoid(tf.matmul(layer1, weight_2_node) + biases_2_node), keep_prob=0.85)
    layer3 = tf.nn.softmax(tf.matmul(layer2, weight_3_node) + biases_3_node)
    return layer3


y_train_prediction = network(X_train_node)
y_test_prediction = network(X_test_node)

# set loss function
cross_entropy = tf.losses.softmax_cross_entropy(y_train_node, y_train_prediction)

#goal seek
optimizer = tf.train.AdamOptimizer(learning_rate=0.005).minimize(cross_entropy)


def calculate_accuracy(actual, predicted):
    actual = np.argmax(actual, 1)
    predicted = np.argmax(predicted, 1)
    return (100 * np.sum(np.equal(predicted, actual)) / predicted.shape[0])


# start TF Session

num_epochs = 100

with tf.Session() as session:
    tf.global_variables_initializer().run()
    for epoch in range(num_epochs):
        start_time = time.time()
        _, cross_entropy_score = session.run([optimizer, cross_entropy],
                                             feed_dict={X_train_node: raw_X_train, y_train_node: raw_y_train})

        if epoch % 10 == 0:
            timer = time.time() - start_time
            print('Epoch: {}'.format(epoch), 'Current loss: {0:.4f}'.format(cross_entropy_score),
                  'Elapsed time: {0:.2f} seconds'.format(timer))
            final_y_test = y_test_node.eval()
            final_y_test_prediction = y_test_prediction.eval()
            final_accuracy = calculate_accuracy(final_y_test, final_y_test_prediction)
            print("Current accuracy: {0:.2f}%".format(final_accuracy))

    final_y_test = y_test_node.eval()
    final_y_test_prediction = y_test_prediction.eval()
    final_accuracy = calculate_accuracy(final_y_test, final_y_test_prediction)
    print("Final accuracy: {0:.2f}%".format(final_accuracy))

    final_fraud_y_test = final_y_test[final_y_test[:, 1] == 1]
    final_fraud_y_test_prediction = final_y_test_prediction[final_y_test[:, 1] == 1]
    final_fraud_accuracy = calculate_accuracy(final_fraud_y_test, final_fraud_y_test_prediction)
    print('Final fraud specific accuracy: {0:.2f}%'.format(final_fraud_accuracy))

print(classification_report(final_y_test, final_y_test_prediction))
print('done')