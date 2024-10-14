import pyod
from sklearn.metrics import balanced_accuracy_score
import numpy as np

# Ex. 1
dataset = pyod.utils.data.generate_data(n_train=1000,
                                        n_test=0,
                                        n_features=1,
                                        contamination=0.1)

x = dataset[0]
y = dataset[2]

z_scores = abs((x - np.mean(x)) / np.std(x))

threshold = np.quantile(z_scores, 1 - 0.1)
print(f"threshold: {threshold}")

y_pred = [0 if i < threshold else 1 for i in z_scores]

print(f"accuracy: {balanced_accuracy_score(y, y_pred)}")