import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import random
import joblib

def choose_split_criterion(is_gini):
    """Selects the criterion for the Random Forest based on input."""
    return "gini" if is_gini else "entropy"

def add_null_indicator(data, null_ratio):
    """Adds null values to the dataset based on the specified ratio."""
    null_indicator = np.random.choice([1, 0], data.shape, p=[1 - 1/null_ratio, 1/null_ratio])
    return data * null_indicator, null_indicator

# Load and preprocess the dataset
breast_cancer = pd.read_csv('./data45.csv')
breast_cancer.set_index('id', inplace=True)
breast_cancer['diagnosis'] = breast_cancer['diagnosis'].map({'M': 1, 'B': 0})

X, y = breast_cancer.drop('diagnosis', axis=1), breast_cancer['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6)

null_ratio = random.randint(2, 6)
X_train, train_null_indicators = add_null_indicator(X_train, null_ratio)
X_test, test_null_indicators = add_null_indicator(X_test, null_ratio)

# Stage 1 - Training the Random Forest Classifier
trees = 800
split_criterion = choose_split_criterion(True)
clf1 = RandomForestClassifier(n_estimators=trees, criterion=split_criterion)
clf1.fit(X_train, y_train)

# Save the first trained model
filename1 = 'finalized_model_layer1.sav'
joblib.dump(clf1, filename1)

# Stage 2 - Training the Random Forest Classifier with previous predictions
y_pred_train = clf1.predict(X_train)
y_pred_test = clf1.predict(X_test)
X_train_stage2 = np.concatenate((X_train, y_pred_train.reshape(-1, 1)), axis=1)
X_test_stage2 = np.concatenate((X_test, y_pred_test.reshape(-1, 1)), axis=1)

clf2 = RandomForestClassifier(n_estimators=trees, criterion=choose_split_criterion(False))
clf2.fit(X_train_stage2, y_train)

# Save the second trained model
filename2 = 'finalized_model_layer2.sav'
joblib.dump(clf2, filename2)

# Predictions and accuracy calculation for both stages
y_pred_train_stage2 = clf2.predict(X_train_stage2)
y_pred_test_stage2 = clf2.predict(X_test_stage2)
train_accuracy_stage2 = metrics.accuracy_score(y_train, y_pred_train_stage2)
test_accuracy_stage2 = metrics.accuracy_score(y_test, y_pred_test_stage2)

# Feature importance analysis and plotting for stage 2
importances2 = clf2.feature_importances_
std2 = np.std([tree.feature_importances_ for tree in clf2.estimators_], axis=0)
indices2 = np.argsort(importances2)[::-1]

plt.figure(figsize=(20, 5))
plt.title("Feature Importances - Stage 2")
plt.bar(range(X_train_stage2.shape[1]), importances2[indices2], color="r", yerr=std2[indices2], align="center")
plt.xticks(range(X_train_stage2.shape[1]), indices2)
plt.xlim([-1, X_train_stage2.shape[1]])
plt.savefig('./feature_importances_stage2.pdf', bbox_inches='tight')

# Printing model accuracy
print(f"Stage 1 - Training Accuracy: {metrics.accuracy_score(y_train, y_pred_train)}")
print(f"Stage 1 - Test Accuracy: {metrics.accuracy_score(y_test, y_pred_test)}")
print(f"Stage 2 - Training Accuracy: {train_accuracy_stage2}")
print(f"Stage 2 - Test Accuracy: {test_accuracy_stage2}")

# Additional analysis and plots can be added as needed
