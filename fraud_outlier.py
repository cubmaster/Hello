from sklearn.neighbors import LocalOutlierFactor
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time


data = pd.read_csv('./Data/creditcard.csv')
print('Data')

unique = data.drop(['V28','V27','V26','V25','V24','V23','V22','V20','V15','V13','V8','Time'], axis=1)
y = unique['Class']
x = unique.drop(['Class'], axis=1)

start_time = time.time()
print(start_time)

num_epochs = 10

for epoch in range(num_epochs):
    kn = max(1, epoch * 10)

    start_time = time.time()
    print(start_time)

    lof = LocalOutlierFactor(n_neighbors=kn)
    result = lof.fit_predict(x)
    print('This took:', (time.time() - start_time) / 60)

    np_results = np.asarray(result)
    np_results = np_results.reshape(-1, len(np_results))
    unique['result'] = np_results[0, np_results[0]]

    print('analysis')
    total_outliers = unique['result'].shape[0]
    true_positive = unique[(unique.Class == 1) & (unique.result == 1)].shape[0]
    false_positive = unique[(unique.Class == 0) & (unique.result == 1)].shape[0]
    true_negitive = unique[(unique.Class == 0) & (unique.result == -1)].shape[0]
    false_negitive = unique[(unique.Class == 1) & (unique.result == -1)].shape[0]

    print('neighbors', kn)
    print('outliers', total_outliers)

    print('True Positive:{0:2d}'.format(true_positive))
    print('False Positive:{0:2d}'.format(false_positive))
    print('True Negative:{0:2d}'.format(true_negitive))
    print('False Negative:{0:2d}'.format(false_negitive))
print('Done')