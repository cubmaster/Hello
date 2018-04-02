import matplotlib.pyplot as plt
import numpy as np


height = np.round(np.random.normal(1.75, 0.2, 1000000), 2)
weight = np.round(np.random.normal(60.32, 15, 1000000), 2)

np_group = np.column_stack((height, weight))

plt.hist(height,1000)
plt.show()

