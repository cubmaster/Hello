import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.cluster import KMeans

# Import and store dataset
data = pd.read_csv('./Data/creditcard.csv')
print('Data')

fraud = data.loc[data['Class'] == 1]
nonfraud = data.loc[data['Class'] == 0]

print('Analysis')


#for column in data.keys():
#    plt.hist(fraud[column], bins=50, alpha=0.5, label=['f'], normed=True)
#    plt.hist(nonfraud[column], bins=50, alpha=0.5, label=['n'], normed=True)
#    plt.legend(loc='upper left')
#    plt.title(column)
#    plt.show()#

#Drop all features that have like distributions
unique = data.drop(['V28','V27','V26','V25','V24','V23','V22','V20','V15','V13','V8'], axis=1)
y = unique['Class']
x = unique.drop(['Class'], axis=1)

xtrain, ytrain, xtest, ytest = train_test_split(x, y, test_size=0.3, random_state=40)


model = KMeans(n_clusters=4).fit_transform(xtrain,ytrain)


