#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
# from tester import dump_classifier_and_data
import pandas as pd
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from datetime import datetime
# from tester.py import dump_classifier_and_data

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression




### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

##################################################################################
#Data Exploration
##################################################################################
enron_df = (pd.DataFrame(data_dict)).T

text_feature_list = ['email_address']
numeric_features_list = ['salary', 'to_messages', 'deferral_payments', 'total_payments', \
                         'exercised_stock_options', 'bonus', 'restricted_stock', 'shared_receipt_with_poi', \
                         'restricted_stock_deferred', 'total_stock_value', 'expenses', 'loan_advances', \
                         'from_messages', 'other', 'from_this_person_to_poi', 'director_fees', \
                         'deferred_income', 'long_term_incentive', 'from_poi_to_this_person']

#total number of data points
# print 'Total number of data points:', enron_df.shape[0]
#number of features used
# print 'Number of features:', (enron_df.shape[1]-1)
#are there features with many missing values?
Missing_values = {}
for feature in numeric_features_list:
    NaN_count = 0
    for i, value in enumerate(enron_df[feature]):
        if value == 'NaN':
            NaN_count +=1
            enron_df[feature][i] = 0
    if NaN_count > 90:
        Missing_values[feature] = NaN_count
        
for feature in numeric_features_list:
    current_NaN_count = 0
    for i, value in enumerate(enron_df[feature]):
        if value == 'NaN':
            current_NaN_count +=1
            enron_df[feature][i] = 0
# print current_NaN_count
        
#allocation across classes (POI/non-POI)
sns.set_style("whitegrid")
for feature in numeric_features_list:
    enron_df[feature] = pd.to_numeric(enron_df[feature])
# print enron_df.groupby('poi').describe()

print 'There are', sum(enron_df['poi']), 'POIs in the dataset.'
        
#Deal with text features:
# enron_df['email_address'] = enron_df['email_address'].apply(str)
# from sklearn.feature_extraction.text import TfidfVectorizer
# vectorizer = TfidfVectorizer(stop_words='english')
# result = vectorizer.fit_transform(enron_df['email_address'])
# print len(vectorizer.get_feature_names())

#Remove the features that has more than 90 nans
for feature in  Missing_values.keys():
    numeric_features_list.remove(feature)
# print 'Number of features used:', len(numeric_features_list)
# print numeric_features_list

#Outliers
feature_df = enron_df[numeric_features_list]
#i find those who have 0 data
# for person in feature_df.index:
#     count_temp = 0
#     for value in feature_df.loc[person]:
#         if value != 0:
#             count_temp += 1
#     if count_temp == 0:
#         print person, 'has no data'
            
#ii find those who have the most data

# def plotmax(i, feature):
#     index = i%4 + 1
#     plt.subplot(2, 2, index)
#     ax = sns.boxplot(x = feature, data = feature_df, color = 'b') 
#     ax = sns.swarmplot(x = feature, data = feature_df, size = 1)
#     plt.plot(max(feature_df[feature]), 0, marker='o', markersize = 6, color = 'r')
#     ax.annotate('Max:'+ feature_df[feature].argmax(), color = 'r', \
#             xy = (max(feature_df[feature]), 0),\
#             xytext = (max(feature_df[feature]*0.65), -0.05))
#     plt.xlabel(feature)

# for i, feature in enumerate(feature_df):
#     plotmax(i, feature)
#     if (i+1)%4 == 0:
#         plt.show()
#     if i == 14:
#         plt.show()
# print 'Delete as an LOCKHART EUGENE E oulier'
# print 'Delete Total as an outlier'
# print 'Delete THE TRAVEL AGENCY IN THE PARK as an outlier'

enron_df = enron_df.drop('TOTAL', axis=0)
enron_df = enron_df.drop('LOCKHART EUGENE E', axis=0)
enron_df = enron_df.drop('THE TRAVEL AGENCY IN THE PARK', axis=0)

del feature_df

##################################################################################
#Create new features (related lesson: "Feature Selection")
##################################################################################
features = enron_df[numeric_features_list]

def ratio_feature(numerator, denominator):
    temp_feature = {}
    for i, value in enumerate(features[denominator]):
        if value == 0:
            temp_feature[features.index[i]] = 0
        else:
            temp_feature[features.index[i]] = (features[numerator][i]/value)
    return temp_feature

ratio_payments_stock = pd.Series(ratio_feature('total_payments','total_stock_value'))
enron_df = pd.concat([enron_df, ratio_payments_stock.rename("ratio_payments_stock")], axis=1)
features = pd.concat([features, ratio_payments_stock.rename("ratio_payments_stock")], axis=1)

poi_this_person = pd.Series(enron_df['from_poi_to_this_person'] + enron_df['from_this_person_to_poi'])
enron_df = pd.concat([enron_df, poi_this_person.rename("poi_this_person")], axis=1)
features = pd.concat([features, poi_this_person.rename("poi_this_person")], axis=1)


##################################################################################
#Intelligently select features (related lesson: "Feature Selection")
##################################################################################
# for i, feature in enumerate(features.columns):
#     plt.subplot(2,2,i%4+1)
#     ax = sns.violinplot(x = enron_df['poi'], y = features[feature])
#     if (i+1)%4 == 0:
#         plt.show()
print len(features)
cor_matrix = features.corr()
for i, row in enumerate(cor_matrix.index):
    for j, col in enumerate(cor_matrix.columns):
        if j>=i:
            break
        temp = cor_matrix.get_value(row, col)
        if temp > 0.7:
            print row, 'and', col, 'have correlation greater than 0.7'
print numeric_features_list

corr_remove_list = ['shared_receipt_with_poi', 'exercised_stock_options', \
                    'restricted_stock', 'other', 'to_messages', \
                    'from_this_person_to_poi', 'from_poi_to_this_person']


numeric_features_list = list(features.columns) # add the two new features into feature list
for item in corr_remove_list:
    numeric_features_list.remove(item)

numeric_features_list.remove('poi_this_person')
numeric_features_list.remove('ratio_payments_stock')


features = enron_df[numeric_features_list]

print '\n'
print 'The final features used in the model are:', list(features.columns)
print '\n'


#traintest split
enron_df['poi'] = pd.to_numeric(enron_df['poi'])
labels = enron_df['poi']
from sklearn.cross_validation import train_test_split

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

# print SelectKBest(k = 1).fit(features_train, labels_train).scores_ 
# print SelectKBest(k = 2).fit(features_train, labels_train).scores_ 
# print SelectKBest(k = 3).fit(features_train, labels_train).scores_ 
# print SelectKBest(k = 4).fit(features_train, labels_train).scores_ 

    
#Transformers
min_max_scaler = preprocessing.MinMaxScaler()
std_scaler = preprocessing.StandardScaler()
pca = PCA(random_state = 42)
slt = SelectKBest()
# N_Options = np.linspace(1,7,7).astype(int)
# K_Options = np.linspace(1,7,7).astype(int)
N_Options = np.array([2,4,6])
K_Options = np.array([2,4,6])

##FeatureUnion
dim_reduction = FeatureUnion([('pca', pca), ("slt", slt)])

svm = SVC(random_state = 42)
C_Options = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
Kernel_Options = ['rbf','sigmoid']
Gamma_Options = [0.001, 0.01, 0.1, 1,'auto']

linsvm = LinearSVC(random_state = 42)
Loss_Options = ['hinge', 'squared_hinge']
Intercept_Options = [True, False]
Maxiter_Options = [500, 1000, 1500, 2000, 3000, 4000, 5000]

#KNN
knn = KNeighborsClassifier()
Algorithm_Options = ['auto', 'ball_tree', 'kd_tree', 'brute']
Neighbor_Options = np.linspace(2,20,19).astype(int)
Knnweights_Options = ['uniform', 'distance']
Knnp_Options = [1, 2, 3]

#Tree
tree = DecisionTreeClassifier(random_state = 42)
Criterion_Options = ['gini', 'entropy']
Splitter_Options = ['best', 'random']
Maxfeature_Options = ['auto', 'sqrt', 'log2', None]
Weight_Options = ['balanced', None]

#Logistic Regression
LR = LogisticRegression()
Penalty_Options = ['l1', 'l2']
LR_C_Options = [0.1, 1, 10, 100]

#RandomForest
rf = RandomForestClassifier(random_state = 42)
Noestimators_Options = np.linspace(1,20,20).astype(int)

#NaiveBayes
NB = GaussianNB()



svc_est = [Pipeline([('scaler', min_max_scaler), ('dim_reduct', dim_reduction), ('clf', svm)]),\
           Pipeline([('scaler', std_scaler), ('dim_reduct', dim_reduction), ('clf', svm)])]
svc_param = {'dim_reduct__pca__n_components':N_Options,\
                  'dim_reduct__slt__k': K_Options,\
                  'clf__C': C_Options,\
                  'clf__kernel': Kernel_Options,\
                  'clf__gamma': Gamma_Options}

linsvm_est = [Pipeline([('scaler', min_max_scaler), ('dim_reduct', dim_reduction), ('clf', linsvm)]),\
              Pipeline([('scaler', std_scaler), ('dim_reduct', dim_reduction), ('clf', linsvm)])]
linsvm_param = {'dim_reduct__pca__n_components': N_Options,\
                'dim_reduct__slt__k': K_Options,\
                'clf__C': C_Options,\
                'clf__loss': Loss_Options,\
                'clf__fit_intercept': Intercept_Options,\
                'clf__max_iter': Maxiter_Options}  

knn_est = [Pipeline([('scaler', min_max_scaler), ('dim_reduct', dim_reduction), ('clf', knn)]),\
           Pipeline([('scaler', std_scaler), ('dim_reduct', dim_reduction), ('clf', knn)])]
knn_param = {'dim_reduct__pca__n_components': N_Options,\
              'dim_reduct__slt__k': K_Options,\
              'clf__weights': Knnweights_Options,\
              'clf__algorithm': Algorithm_Options,\
              'clf__n_neighbors': Neighbor_Options,\
              'clf__p': Knnp_Options}  


tree_est = Pipeline([('dim_reduct', dim_reduction), ('clf', tree)])
tree_param = {'dim_reduct__pca__n_components': N_Options,\
              'dim_reduct__slt__k': K_Options,\
              'clf__criterion': Criterion_Options,\
              'clf__splitter': Splitter_Options,\
              'clf__max_features': Maxfeature_Options,\
              'clf__class_weight': Weight_Options}   

rf_est = Pipeline([('dim_reduct', dim_reduction), ('clf', rf)])
rf_param = {'dim_reduct__pca__n_components': N_Options,\
              'dim_reduct__slt__k': K_Options,\
              'clf__n_estimators': Noestimators_Options,\
              'clf__criterion': Criterion_Options,\
              'clf__max_features': Maxfeature_Options,\
              'clf__class_weight': Weight_Options}

NB_est = Pipeline([('dim_reduct', dim_reduction), ('clf', NB)])
NB_param = {'dim_reduct__pca__n_components': N_Options,\
              'dim_reduct__slt__k': K_Options}


LR_est = [Pipeline([('scaler', min_max_scaler), ('dim_reduct', dim_reduction), ('clf', LR)]),\
              Pipeline([('scaler', std_scaler), ('dim_reduct', dim_reduction), ('clf', LR)])]
LR_param = {'dim_reduct__pca__n_components': N_Options,\
                'dim_reduct__slt__k': K_Options,\
                'clf__penalty': Penalty_Options,\
                'clf__C': LR_C_Options}  



#Classifiers
clf_est = {'svc_min_max': {'estimator': svc_est[0], 'param':svc_param},
           'svc_std': {'estimator': svc_est[1], 'param':svc_param},
          'linsvc_min_max': {'estimator': linsvm_est[0], 'param':linsvm_param},
           'linsvc_std': {'estimator': linsvm_est[1], 'param':linsvm_param},
           'knn_min_max': {'estimator': knn_est[0], 'param':knn_param},
           'knn_std': {'estimator': knn_est[1], 'param':knn_param},
           'tree': {'estimator': tree_est, 'param':tree_param},
           'NaiveBayes': {'estimator': NB_est, 'param':NB_param},
           'random_forest': {'estimator': rf_est, 'param':rf_param},
           # 'lg_regression_min_max':{'estimator': LR_est[0], 'param':LR_param},
           # 'lg_regression_std': {'estimator': LR_est[1], 'param':LR_param}
           }
training_order = ['NaiveBayes', 'knn_min_max', 'knn_std',\
                     'svc_min_max','svc_std', 'linsvc_min_max', 'linsvc_std',\
                     'tree', 'random_forest']
def get_time(time1, time2, name):
    delta = time2 - time1
    secs  = delta.seconds%60
    mins = delta.seconds/60
    print '###########################################################'
    print 'Time report of', name, 'model'
    print 'Model',name, 'started from', time1, 'ended at', time2
    print 'It took', mins, 'minutes',secs, 'seconds','to run the model'
    print '###########################################################'

def train_tune_model(est, param, name):
    time1 = datetime.now()    
    grid_search = GridSearchCV(estimator = est, param_grid = param).fit(features_train, labels_train)
    # print grid_search.score(features_test, labels_test)
    clf = grid_search.best_estimator_
    dump_classifier(clf, name)
    time2 = datetime.now()
    get_time(time1, time2, name)

     #store Decision Tree model as final model
    if name == 'tree':
        with open(CLF_PICKLE_FILENAME, "w") as clf_outfile:
                pickle.dump(clf, clf_outfile)


#store data and features
CLF_PICKLE_FILENAME = 'my_classifier.pkl'
DATASET_PICKLE_FILENAME = "my_dataset.pkl"
FEATURE_LIST_FILENAME = "my_feature_list.pkl"


def clf_filename(name):
    return 'best_' + name + '_classifier.pkl'
def dump_classifier(clf, name):
    with open(clf_filename(name), "w") as clf_outfile:
        pickle.dump(clf, clf_outfile)

def dump_data_and_feature(dataset, feature_list):
    with open(DATASET_PICKLE_FILENAME, "w") as dataset_outfile:
        pickle.dump(dataset, dataset_outfile)
    with open(FEATURE_LIST_FILENAME, "w") as featurelist_outfile:
        pickle.dump(feature_list, featurelist_outfile)

def fine_tune_classifiers():
    for item in training_order:
        # if item == 'tree':
        name = item
        est = clf_est[name]['estimator']
        param = clf_est[item]['param']
        train_tune_model(est, param, name)

            

    
fine_tune_classifiers()

my_dataset = dict(enron_df.T)
features_list = ['poi'] + list(features.columns)
dump_data_and_feature(my_dataset, features_list)





