import pandas as pd
import numpy as np
import time
import os
np.random.seed(0)  # set random seed before importing sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
import seaborn as sns

if "Figs" not in os.listdir():
	os.system("mkdir Figs")


## DATA PREPARATION ##
# Read in data and set feature names accordingly
data = pd.read_csv("Data/spam_data.txt", sep=" ", header=None)
cols_df = pd.read_csv("Data/spambase.names", header=None, skiprows=33)
cols = cols_df[0].tolist()
data.columns = [col.split(":")[0] for col in cols] + ["spam"]

# Split into training and test sets
X = data.drop("spam", axis=1)
y = data[["spam"]].values.ravel()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

print("TRAINING SAMPLES:", X_train.shape[0])
print("TEST SAMPLES:", X_test.shape[0])
print("---")


## A SIMPLE CLASSIFIER ##
def predict(X):
	# Simple classifier that predicts spam whenever the longest 
	# uninterrupted sequence of capital letters exceeds 20
	preds = X["capital_run_length_longest"] > 20
	return np.array(preds)

y_pred = predict(X_test)
print("One-rule classifier test accuracy:", np.sum(y_pred == y_test) / X_test.shape[0])
print("---")

# Plot confusion matrix for simple classifier applied to spam data
cm = confusion_matrix(y_test, y_pred)
plt.rcParams["figure.figsize"] = [3, 3]
fig, ax = plot_confusion_matrix(cm, show_normed=True, colorbar=True)
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
fig.savefig("Figs/CM_simple.png", dpi=600, bbox_inches="tight")
plt.show()


## FITTING A DECISION TREE ##
dt = DecisionTreeClassifier(random_state=0, min_samples_leaf=1).fit(X_train, y_train)
print("Decision tree (d=1) test accuracy:", dt.score(X_test, y_test))
print("\tTree depth:", dt.get_depth())
print("\tNo. leaves:", dt.get_n_leaves())
print("---")

# Plot confusion matrix for decision tree fit to spam data
cm = confusion_matrix(y_test, dt.predict(X_test))
plt.rcParams["figure.figsize"] = [3, 3]
fig, ax = plot_confusion_matrix(cm, show_normed=True, colorbar=True)
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
fig.savefig("Figs/CM_DT.png", dpi=600, bbox_inches="tight")
plt.show()


# BAGGING DECISION TREES ##
# Wall time of training bagged classifier with 500 trees
s = time.time()
bag = BaggingClassifier(base_estimator=DecisionTreeClassifier(random_state=0, min_samples_leaf=1),
					    n_estimators=500).fit(X_train, y_train)
print(round(time.time() - s, 3), "seconds to train a bagged classifier with B=500 trees")

# Wall time of inference for one email
ex = X_test.iloc[0, :].to_numpy().reshape(1, -1)
s = time.time()
bag.predict(ex)
print(round(time.time() - s, 3), "seconds to make a prediction with the above bagged classifier")
print("---")

print("Bagged classifier (B=500) test accuracy:", bag.score(X_test, y_test))
print("---")

cm = confusion_matrix(y_test, bag.predict(X_test))
fig, ax = plot_confusion_matrix(cm, show_normed=True, colorbar=True)
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
fig.savefig("Figs/CM_bagging.png", dpi=600, bbox_inches="tight")
plt.show()

# Fit the same above bagged classifier 30 times
accs = []
FPRs = []  # false positive rates
for _ in range(30):
	bag = BaggingClassifier(base_estimator=DecisionTreeClassifier(random_state=0, min_samples_leaf=1),
						    n_estimators=500).fit(X_train, y_train)
	accs.append(bag.score(X_test, y_test))
	y_pred = bag.predict(X_test)
	FP = 0
	for i in range(y_test.shape[0]):
		if y_pred[i] == 1 and y_test[i] != y_pred[i]:
			FP += 1
	FPRs.append(FP / np.sum(y_test == 0))

print("Test accuracy from 30 fits of above bagged classifier:", np.mean(accs), "+/-", np.std(accs))
print("False positive rate from 30 fits of above bagged classifier:", np.mean(FPRs), "+/-", np.std(FPRs))


## RANDOM FORESTS ##
# Wall time of training random forest with 500 trees
s = time.time()
rf = RandomForestClassifier(n_estimators=500, random_state=0, min_samples_leaf=1).fit(X_train, y_train)
print(round(time.time() - s, 3), "seconds to train a random forest of B=500 trees")

s = time.time()
rf.predict(ex)
print(round(time.time() - s, 3), "seconds to make a prediction with the above random forest")

print(f"Random forest (B=500) test accuracy: {rf.score(X_test, y_test)}")
print("---")

# Plot confusion matrix for random forest
cm = confusion_matrix(y_test, rf.predict(X_test))
fig, ax = plot_confusion_matrix(cm, show_normed=True, colorbar=True)
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
fig.savefig("Figs/CM_RF.png", dpi=600, bbox_inches="tight")
plt.show()

# Create feature importance plot
importance_df = pd.DataFrame({"Predictors": data.columns.values[:-1], "Importance": rf.feature_importances_})
importance_df = importance_df.sort_values(['Importance'])
plt.rcParams["figure.figsize"] = [6.4, 4.8]  # reset to default figure size (for me)
sns.set(font_scale=0.5)
ax = sns.barplot(y='Predictors', x='Importance', data=importance_df, orient='h')
ax.tick_params(axis='x', labelsize=8)
ax.set_xlabel("Feature Importance", fontsize=10)
ax.set_ylabel('')
ax.get_figure().savefig("Figs/feature_imp_RF.png", dpi=600, bbox_inches='tight')
plt.show()
sns.reset_orig()

# Grid search with cross-validation (takes a while... ~30 min)
param_grid = {'max_features': list(range(1, 8)),
			  'min_samples_leaf': list(range(1, 4)),
			  'n_estimators': [100, 250, 500, 1000, 2000]}

grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv=10, verbose=2)
grid_search.fit(X_train, y_train)

# Save full grid search results to .csv
res = pd.DataFrame(grid_search.cv_results_)
res.to_csv("Data/RF_GridSearchCV.csv", index=False)

print("Tuned random forest test accuracy:", grid_search.best_estimator_.score(X_test, y_test))
print("---")

# Plot confusion matrix for tuned random forest
cm = confusion_matrix(y_test, grid_search.best_estimator_.predict(X_test))
fig, ax = plot_confusion_matrix(cm, show_normed=True, colorbar=True)
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
fig.savefig("Figs/CM_RF_tuned.png", dpi=600, bbox_inches="tight")
plt.show()


## COMPARE BAGGING TO RANDOM FORESTS ##
B_vals = [10, 25, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1250, 1500] 

# Fit bagged classifier & random forest for above values of B and record test accuracies
bag_accs = []
rf_accs = []
for B in B_vals:
	bag = BaggingClassifier(base_estimator=DecisionTreeClassifier(random_state=0, min_samples_leaf=1),
						    n_estimators=B).fit(X_train, y_train)
	bag_accs.append(bag.score(X_test, y_test))

	RF = RandomForestClassifier(random_state=0, min_samples_leaf=1, n_estimators=B).fit(X_train, y_train)
	rf_accs.append(RF.score(X_test, y_test))

fig, ax = plt.subplots(figsize=(6,6))
ax.plot([1] + B_vals, [dt.score(X_test, y_test)] + bag_accs, '-o', label='Bagging')
ax.plot(B_vals, rf_accs, '-o', label='Random Forest')
ax.legend(loc='lower right')
ax.set_ylim(0.9, 0.95)
ax.set_xlabel("Number of Trees (B)")
ax.set_ylabel("Mean Test Accuracy")
fig.savefig("Figs/bag_vs_RF.png", dpi=600, bbox_inches="tight")
plt.show()


## EXTRA-TREES ON THE SPAM DATA ##
ET = ExtraTreesClassifier(random_state=0, min_samples_leaf=1, n_estimators=500).fit(X_train, y_train)
print("Extra-Trees (B=500, d=1, k=7) test accuracy:", ET.score(X_test, y_test))
print("---")

cm = confusion_matrix(y_test, ET.predict(X_test))
fig, ax = plot_confusion_matrix(cm, show_normed=True, colorbar=True)
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
fig.savefig("Figs/CM_Extra-Trees.png", dpi=600, bbox_inches="tight")
plt.show()


## COMPARE EXTRA-TREES TO RANDOM FORESTS ##
B_vals = [10, 50, 100, 250, 500, 750, 1000, 1250, 1500]

# Fit bagged classifier, random forest, & Extra-Trees for above values of B and record test accuracies
bag_accs = []
rf_accs = []
extra_accs = []
for B in B_vals:
	bag = BaggingClassifier(base_estimator=DecisionTreeClassifier(random_state=0, min_samples_leaf=1),
						    n_estimators=B).fit(X_train, y_train)
	bag_accs.append(bag.score(X_test, y_test))

	RF = RandomForestClassifier(random_state=0, min_samples_leaf=1, n_estimators=B).fit(X_train, y_train)
	rf_accs.append(RF.score(X_test, y_test))

	ET = ExtraTreesClassifier(random_state=0, min_samples_leaf=1, n_estimators=B).fit(X_train, y_train)
	extra_accs.append(ET.score(X_test, y_test))

fig, ax = plt.subplots(figsize=(6,6))
ax.plot([1] + B_vals, [dt.score(X_test, y_test)] + bag_accs, '-o', label='Bagging')
ax.plot(B_vals, rf_accs, '-o', label='Random Forest')
ax.plot(B_vals, extra_accs, '-o', label='Extra-Trees')
ax.legend(loc='lower right')
ax.set_ylim(0.9, 0.96)
ax.set_xlabel("Number of Trees (B)")
ax.set_ylabel("Mean Test Accuracy")
fig.savefig("Figs/Extra-Trees_vs_RF.png", dpi=600, bbox_inches="tight")
plt.show()
