#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn import cross_validation
from time import time
import sys
from sklearn.metrics import confusion_matrix, precision_score, recall_score, classification_report
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)


features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.3, random_state=42)


### it's all yours from here forward!  
clf = tree.DecisionTreeClassifier()

clf = tree.DecisionTreeClassifier()
t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"
t0 = time()
pred = clf.predict(features_test)
print "predict time:", round(time()-t0, 3), "s"
accuracy = accuracy_score(labels_test, pred)

# print confusion_matrix(labels_test, pred)
# print precision_score(labels_test, pred)
# print recall_score(labels_test, pred)

predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1] 
truelabels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]
print confusion_matrix(truelabels, predictions)

print classification_report(truelabels, predictions)

