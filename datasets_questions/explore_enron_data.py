#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle
import pandas
from pydoc import describe
enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
print type(enron_data)
person = enron_data.keys()
features = enron_data[person[0]].keys()
print "num of person:",len(enron_data)
print "num of feature:",len(enron_data[person[0]])

df = pandas.DataFrame(enron_data.values(),index=enron_data.keys())
# print len(df.loc[df['poi'] == True])
# subdf = df.loc[['SKILLING JEFFREY K','LAY KENNETH L','FASTOW ANDREW S']]
# print subdf['total_payments'].argmax()
# print subdf['total_payments'].max()
# print df.loc['LAY KENNETH L']['total_payments']
#print len(df.loc[(df['total_payments']=='NaN') & (df['poi']==True)])
print len(df.loc[(df['poi']==True)])
print len(df.loc[(df['total_payments']!='NaN') & (df['poi']==True)])
print pandas.to_numeric(df['exercised_stock_options'], errors='coerce').describe()
print pandas.to_numeric(df.loc[:,'exercised_stock_options'], errors='coerce').max(skipna=True)