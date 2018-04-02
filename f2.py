import pandas as pa

from Helpers.DataHelpers import *
from Helpers.training import *


data = pa.read_csv('./Data/syn_fraud.csv')

print(data.head(10))


correlation(data, True)