import sys
import pickle
import numpy 
from numpy import nan
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit

with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

features_list = ['salary', 'to_messages', 'deferral_payments', 'total_payments',
       'exercised_stock_options', 'bonus', 'restricted_stock',
       'shared_receipt_with_poi', 'restricted_stock_deferred',
       'total_stock_value', 'expenses', 'loan_advances', 'from_messages',
       'other', 'from_this_person_to_poi', 'director_fees',
       'deferred_income', 'long_term_incentive','email_address',
       'from_poi_to_this_person','poi']
  
fout = open('data2.csv','w')
fout.write("name,")
fout.write(",".join(features_list))
fout.write("\n")
for person in data_dict:
    line = person+','
    for f in features_list:
        line += str(data_dict[person][f])+','
    line = line[:-1]+'\n'
    fout.write(line)
fout.close()