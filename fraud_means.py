import pandas as pd
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

# Import and store dataset
data = pd.read_csv('./Data/creditcard.csv')
print('Data')

fraud = data.loc[data['Class'] == 1]
nonfraud = data.loc[data['Class'] == 0]

print('Analysis')

#Plot each feature to see fraud's significatnce vs. Non-fraud
#for column in data.keys():
#    plt.hist(fraud[column], bins=50, alpha=0.5, label=['f'], normed=True)
#    plt.hist(nonfraud[column], bins=50, alpha=0.5, label=['n'], normed=True)
#    plt.legend(loc='upper left')
#    plt.title(column)
#    plt.show()#

#Drop all features that have like distributions
unique = data.drop(['V28','V27','V26','V25','V24','V23','V22','V20','V15','V13','V8','Time'], axis=1)
y = unique['Class']
x = unique.drop(['Class'], axis=1)



#xtrain, ytrain, xtest, ytest = train_test_split(x, y, test_size=0.3, random_state=40)


kmeans = KMeans(n_clusters=20)

print('start fit_predict')
result = kmeans.fit_predict(x)






unique['result'] = result

plt.hist(unique[unique['Class'] == 1], max(unique['result']))
plt.show()



print('done')

#pca = PCA(n_components=12)
#scaler = StandardScaler()
#pipeline = make_pipeline(scaler, pca)
#
#pipeline.fit(unique)
#
#features = range(pca.n_components_)
#plt.bar(features, pca.explained_variance_)
#plt.xlabel('PCA feature')
#plt.ylabel('variance')
#plt.xticks(features)
#plt.show()
#
#x = pca.components_[:,0]
#y = pca.components_[:,1]
#
#plt.scatter(x,y)
#plt.axis('equal')
#plt.show()




