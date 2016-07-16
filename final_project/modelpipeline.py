#!/usr/bin/python

import sys
import pickle
import numpy 
from numpy import nan
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.feature_selection import SelectKBest
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import *
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn.metrics import *
from sklearn import grid_search
from sklearn import cross_validation
from sklearn.metrics import matthews_corrcoef
import copy
result = open('modelselect.txt','w')

#load data 
features_list = ['poi','salary', 'to_messages', 'deferral_payments', 'total_payments',
       'exercised_stock_options', 'bonus', 'restricted_stock',
       'shared_receipt_with_poi', 'restricted_stock_deferred',
       'total_stock_value', 'expenses', 'loan_advances', 'from_messages',
       'other', 'from_this_person_to_poi', 'director_fees',
       'deferred_income', 'long_term_incentive',
       'from_poi_to_this_person','poi_connect','ratio_from_poi','ratio_to_poi','ratio_poi_connect'] # You will need to use more features
# features_list = ['poi','salary', 'deferral_payments', 'total_payments',
#        'exercised_stock_options', 'bonus', 'restricted_stock',
#        'shared_receipt_with_poi', 'restricted_stock_deferred',
#        'total_stock_value', 'expenses', 'loan_advances','director_fees',
#        'deferred_income', 'long_term_incentive',
#        'poi_connect']
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
del(data_dict['TOTAL'])
### generate new features

for person in data_dict:
    try:
        data_dict[person]['poi_connect'] = str(float(data_dict[person]['from_poi_to_this_person'])+float(data_dict[person]['from_this_person_to_poi']))
    except:
        data_dict[person]['poi_connect'] = '0'

    try:
        data_dict[person]['ratio_from_poi'] = str(float(data_dict[person]['from_poi_to_this_person'])+float(data_dict[person]['from_messages']))
    except:
        data_dict[person]['ratio_from_poi'] = '0'
        
    try:
        data_dict[person]['ratio_to_poi'] = str(float(data_dict[person]['from_this_person_to_poi'])+float(data_dict[person]['to_messages']))
    except:
        data_dict[person]['ratio_to_poi'] = '0'
        
    try:
        data_dict[person]['ratio_poi_connect'] = str(float(data_dict[person]['from_poi_to_this_person'])+float(data_dict[person]['from_this_person_to_poi']))
    except:
        data_dict[person]['ratio_poi_connect'] = '0'
        
    for i in ['poi_connect','ratio_to_poi','ratio_from_poi','ratio_poi_connect']:
        if data_dict[person][i] == "nan":
            data_dict[person][i]  = '0'

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# spliting into training and testing set
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

# feature scale
min_max_scaler = preprocessing.MinMaxScaler()
min_max_scaler.fit(features_train)
features_train_tranform = min_max_scaler.transform(features_train)
features_test_tranform = min_max_scaler.transform(features_test)

# feature selection
# First rank all features
select = SelectKBest(k='all')
select.fit(features_train_tranform, labels_train)
scores = select.scores_
features_list = features_list[1:]
feature_names = [features_list[i] for i in select.get_support(indices=True)]
allscores = [str(scores[i]) for i in select.get_support(indices=True)]
result.write('rank_features:\n'+','.join(feature_names)+'\n')
result.write('features score:\n'+','.join(allscores)+'\n')

# Then select 5 top features
select = SelectKBest( k=5)
select.fit(features_train_tranform, labels_train)
features_train_tranform = select.transform(features_train_tranform)
features_test_tranform = select.transform(features_test_tranform)

# output top 5 feature names to result file
scores = select.scores_
feature_names = [features_list[i] for i in select.get_support(indices=True)]
fivescores = [str(scores[i]) for i in select.get_support(indices=True)]
result.write('top_features:\n'+','.join(feature_names)+'\n')
result.write('features score:\n'+','.join(fivescores)+'\n')

# try 8 different classifiers
names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree",
         "Random Forest", "AdaBoost", "Naive Bayes","GradientBoostingClassifier"]
classifiers = [
    KNeighborsClassifier(),
    SVC(kernel="linear"),
    SVC(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GaussianNB(),
    GradientBoostingClassifier()]

# train, test, and output classification report of test set
for name, clf in zip(names, classifiers):
    clf.fit(features_train_tranform, labels_train)
    predictions = clf.predict(features_test_tranform)
    result.write(name+'\n')
    try:
        result.write(classification_report(labels_test, predictions))
    except:
        result.write('error')
    result.write('\n')

# use DecisionTreeClassifier 
# cross_validation on the whole dataset
features_tranform = select.transform(features)
features_tranform = min_max_scaler.fit_transform(features_tranform)
features_tranform = numpy.array(features_tranform)
labels = numpy.array(labels)
k_fold = cross_validation.KFold(len(features_tranform), n_folds=10)
cv_pred = []
cv_label = []
clf = DecisionTreeClassifier()
for train, test in k_fold:  
    train_feature = features_tranform[train]
    train_label = labels[train]
    test_feature = features_tranform[test]
    test_label = labels[test]
    cv_label += list(test_label)
    clf.fit(train_feature,train_label)
    cv_pred += list(clf.predict(test_feature))
print classification_report(cv_label, cv_pred)   
print "cross validation score: \n"

# parameter tuning
dt_param_grid = {
              "criterion": ["gini", "entropy"],
              "min_samples_split": [2,4],
              "max_depth": [None,2,4],
              "min_samples_leaf": [1,3,5]
              }


clf = grid_search.GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=dt_param_grid, scoring='f1', cv=5)
clf.fit(features_train_tranform, labels_train)

predictions = clf.predict(features_test_tranform)
result.write('GridSearchCV  DecisionTreeClassifier:\n')
try:
    result.write(classification_report(labels_test, predictions))
except:
    result.write('error')
mcc = matthews_corrcoef(labels_test, predictions)  
result.write('\nMCC = %s' % str(mcc))
result.write('\n')
