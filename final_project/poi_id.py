#!/usr/bin/python

import sys
import pickle
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn import grid_search
from sklearn.metrics import *
from sklearn.cross_validation import StratifiedShuffleSplit
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary','exercised_stock_options','bonus','shared_receipt_with_poi','total_stock_value']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
del(data_dict['TOTAL'])
### Task 3: Create new feature(s)

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)

labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)



# use GridSearchCV to search for best  DecisionTreeClassifier estimator
param_grid = {
              "criterion": ["gini", "entropy"],
              "min_samples_split": [2,4],
              "max_depth": [None,2,4],
              "min_samples_leaf": [1,3,5]
              }

modeloptimal = grid_search.GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=param_grid, scoring='f1', cv=5)
modeloptimal.fit(features_train, labels_train)

clf = modeloptimal.best_estimator_


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)