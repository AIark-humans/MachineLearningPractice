from sklearn.datasets import fetch_openml
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.linear_model import  SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve,  roc_curve
from sklearn.ensemble import RandomForestClassifier


mnist = fetch_openml("mnist_784", version=1)
X, y = mnist.data, mnist.target

#-----------------------------------BINARY CLASSIFICATION EXAMPLE-------------------------------------------------
#Just to take a look at the data
some_digit = X.iloc[600].to_numpy()
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap = matplotlib.cm.binary, interpolation="nearest")
plt.axis("off")
plt.show()

X_train, X_test, y_train, y_test = X.iloc[:60000], X.iloc[60000:], y.iloc[:60000], y.iloc[60000:]
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train.iloc[shuffle_index], y_train.iloc[shuffle_index]

y_train_5 = (y_train == "5").astype(int)
y_test_5 = (y_test == "5")

sgd_clf = SGDClassifier(random_state=42)
# sgd_clf.fit(X_train, y_train_5)

# cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy") #returns the accuracy of each time
# #0.94995, 0.9656, 0.9648 <-- High accuracies, but only because 10% of the database are 5s. A model which guesses no for all of them would have a 90% accuracy

# y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv = 3) #returns each of the predictions one each instance 
# conf_matrix = confusion_matrix(y_train_5, y_train_pred)
# # [[53157 1422] [[TN FP]
# # [1016 4405]]  [FN TP]]
# precision = precision_score(y_train_5, y_train_pred)
# print(precision)
# recall = recall_score(y_train_5, y_train_pred)
# print(recall)
# f1 = f1_score(y_train_5, y_train_pred)
# print(f1)

# y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv = 3, method= "decision_function")#gives scores from the descision function not the categorizations
forest_clf = RandomForestClassifier(random_state=42)
# y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv =3, method="predict_proba")
# y_scores_forest = y_probas_forest[:, 1] #score = proba of positive class
# fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)


# def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
#     plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
#     plt.plot(thresholds, recalls[:-1], "g-", label = "Recall")
#     plt.xlabel("Threshold")
#     plt.legend(loc="upper left")
#     plt.ylim([0, 1])

# def plot_precision_vs_recall(precisions, recalls):
#     plt.plot(recalls[:-1], precisions[:-1], "r-")
#     plt.xlabel("Recall")
#     plt.ylabel("Precision")

# def plot_roc_curves(fprs, tprs, label=None, color="b"):
#     plt.plot(fprs, tprs, f"{color}-", label = label)
#     plt.xlabel("False Positive Rate")
#     plt.ylabel("True Positive Rate")
#     plt.legend(loc="lower right")


# precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
# plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
# plt.show()
# fprs, tprs, thresholds = roc_curve(y_train_5, y_scores)
# plot_precision_vs_recall(precisions, recalls)
# plt.show()
# plot_roc_curves(fprs, tprs, "Default")
# plot_roc_curves(fpr_forest, tpr_forest, "RandomForest", "r")
# plt.show()

#--------------------------------MULTICLASS CLASSIFICATION EXAMPLE---------------------------------------
sgd_clf.fit(X_train, y_train) # sklearn automatically trains 10 models under the hood
forest_clf.fit(X_train, y_train) #no need for OvA RandomForest is perfectly fine for Multinomial classficiation

# print(sgd_clf.predict([some_digit]))
# some_digit_scores = sgd_clf.decision_function([some_digit])
# print(some_digit_scores)


# print(forest_clf.predict([some_digit]))
# print(forest_clf.predict_proba([some_digit]))

#Example
some_digits = np.random.randint(0, len(X_train), 10)
for i in range(len(some_digits)):
    some_digit = X_train.iloc[some_digits[i]].to_numpy()
    some_digit_image = some_digit.reshape(28, 28)
    plt.imshow(some_digit_image, cmap = matplotlib.cm.binary, interpolation="nearest")
    plt.axis("off")
    plt.show()
    print(f"Predicted: {forest_clf.predict([some_digit])} Actual: {y_train.iloc[some_digits[i]]}")



