from Helpers.DataHelpers import *
from Helpers.training import *
import dill as pickle


def main():
    print("hello")
    df = get_csv(r"data/pima-data.csv")
    del df['skin']

    #print("Null Fields:{0}".format(simple_null_check(df)))
    print(correlation(df,  False))
    xtrain, xtest, ytrain, ytest = splitdata(df,
                                             ['num_preg', 'glucose_conc', 'diastolic_bp', 'thickness', 'insulin', 'bmi', 'diab_pred', 'age'],
                                             ['diabetes'],
                                             0.30)

    #print("{0:0.2f}% in training set".format((len(xtrain)/len(df.index))*100))
    #print("{0:0.2f}% in test set".format((len(xtest)/len(df.index))*100))

    #print("# rows {0}".format(len(df)))

    #for column_name in df.columns:
        #print("# rows with zero in {0} => {1}".format(column_name, len(df.loc[df[column_name] == 0])))

    xtrain = mean_imputer(0, xtrain)
    xtest = mean_imputer(0, xtest)

    model = logistic_regression_fit(xtrain, ytrain)

    print("Accuracy vs. Training Data")
    analysis(model, xtrain, ytrain)

    print("Accuracy vs. Testing Data")
    analysis(model, xtest, ytest)

    filename = 'model_v1.pk'
    with open('./models/' + filename, 'wb') as file:
        pickle.dump(model, file)



if __name__ == '__main__':
    main()

