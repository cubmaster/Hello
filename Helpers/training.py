from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from matplotlib import pyplot as plt



def splitdata(df, feature_col_names, predicted_class_names, size):
    x = df[feature_col_names].values
    y = df[predicted_class_names].values

    return train_test_split(x, y, test_size=size, random_state=40)


def naive_bayes_gaussian(x, y):
    print("Naive Bayes Gaussian")
    model = GaussianNB()
    model.fit(x, y.ravel())
    return model


def random_forrest(x, y):
    print("Random Forrest")
    model = RandomForestClassifier(random_state=40)
    model.fit(x, y.ravel())
    return model


def logistic_regression(x, y, c):
    print("Logistic Regression")
    model = LogisticRegression(C=c, random_state=40, class_weight="balanced")
    model.fit(x, y.ravel())
    return model

def logistic_recgression_cross(x,y):
    print("Logistic Regression CV")
    model = LogisticRegressionCV(n_jobs=-1, Cs=3, cv=10, refit=False, class_weight="balanced")
    model.fit(x, y.ravel())
    return model


def logistic_regression_fit(x, y):
    c_start = 0.1
    c_end = 5
    c_inc = 0.1

    c_values, recall_scores = [], []

    c_val = c_start
    best_recall_score = 0
    while c_val < c_end:
        c_values.append(c_val)
        model = logistic_regression(x, y, c_val)
        recall_score = metrics.recall_score(y, model.predict(x))
        recall_scores.append(recall_score)
        if recall_score > best_recall_score:
            best_recall_score = recall_score
            best_model = model

        c_val = c_val + c_inc
    print('graph---')
    plt.plot(c_values, recall_scores, "-")
    plt.show()

    return best_model


def analysis(model, x, y):
    predictions = model.predict(x)
    print("Accuracy: {0:.4f}".format(metrics.accuracy_score(y, predictions)))
    print("Confusion Matrix:")
    print("{0}".format(metrics.confusion_matrix(y, predictions)))
    print("Classification Report:")
    print(metrics.classification_report(y, predictions))
