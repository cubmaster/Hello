from Helpers.DataHelpers import *
from Helpers.training import *
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


# create the transform


data = get_csv(r"data/spam.csv")
print(data.describe())

xtrain, xtest, ytrain, ytest = splitdata(data, ['v2'], ['v1'], 0.30)

print(xtrain.shape)

vect = CountVectorizer()

# tokenize and build vocab
xtrain_dtm = vect.fit_transform(xtrain.ravel())
xtest_dtm = vect.transform(xtest.ravel())


model = naive_bayes_Multinomial(xtrain_dtm, ytrain.ravel())

print('have model')

analysis(model, xtest_dtm.toarray(), ytest)

print('done')

man_test = np.array(['you won 1,000,000'])
mantest_dtm = vect.transform(man_test.ravel())

result = model.predict_proba(mantest_dtm)

print(result)


