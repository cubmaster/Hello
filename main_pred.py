import dill as pickle
import pandas as pd
from sklearn.externals import joblib


with open('./models/model_v1.pk','rb') as f:
    loaded_model = pickle.load(f)


test = [10, 168, 74, 0, 0, 38, 0.537, 34, 0]

predict = loaded_model.predict(test)

print(str(predict))