import pyod
import matplotlib.pyplot as plt
from pyod.models.knn import KNN
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, roc_curve

# Ex. 1
dataset = pyod.utils.data.generate_data(n_train=400,
                                        n_test=100,
                                        n_features=2,
                                        contamination=0.1)

x_train = dataset[0]
x_test = dataset[1]
y_train = dataset[2]
y_test = dataset[3]
colors = ["red" if label == 0 else "green" for label in y_train]

# plt.scatter(x_train[:,0], x_train[:,1], c=colors)
# plt.show()

# Ex. 2
model = KNN(contamination=0.3)
model.fit(x_train)

y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

train_scores = model.decision_scores_

cm_train = confusion_matrix(y_train, y_train_pred)
print(cm_train)
train_accuracy = balanced_accuracy_score(y_train, y_train_pred)
print(train_accuracy)
fpr_train, tpr_train, thresholds = roc_curve(y_train, train_scores)

plt.plot(fpr_train, tpr_train)
plt.show()

cm_test = confusion_matrix(y_test, y_test_pred)
print(cm_test)
test_accuracy = balanced_accuracy_score(y_test, y_test_pred)
print(test_accuracy)

fpr_test, tpr_test, thresholds = roc_curve(y_test, y_test_pred)

# plt.plot(fpr_test, tpr_test)
# plt.show()