import pyod
from sklearn.metrics import balanced_accuracy_score
import numpy as np

# Ex. 1
x, _, y, _ = pyod.utils.data.generate_data(n_train=1000,
                                           n_test=0,
                                           n_features=3,
                                           contamination=0.1)


z_scores = (x - np.mean(x, axis=0)) / np.std(x, axis=0)

threshold = np.quantile(z_scores, 0.3)
print(f"threshold: {threshold}")

# y_pred = [0 if i > threshold else 1 for i in z_scores]

# print(f"accuracy: {balanced_accuracy_score(y, y_pred)}")