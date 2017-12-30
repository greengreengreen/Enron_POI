## Enron Person of Interest Model
1. Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?  [relevant rubric items: “data exploration”, “outlier investigation”]

    * The goal of this project is to identify person of interest in Enron company by his salary, bonus, etc;
    * Machine learning is useful on training models using Enron dataset with features like salary, bonus in order to predict POI;
    * Enron corporation was an American energy, commodity, services company. In 2001, it collapsed and went bankruptcy because of the well-known accounting fraud, known as Enron Scandal;
    * In this project, I will use the Enron dataset with 146 samples and 21 features. 
        * There are many features with missing values, displayed as ‘NaN’ in the dataset. Since there exists sum relationship between features, for example, ‘Total_Payments’ is the sum of ‘salary’, ‘bonus’, etc, including ‘deferral_payments’, I transformed all the NaNs into 0;
            
            The Enron dataset has 3 outliers in total:
                'Total': It’s not any person. It’s just the sum of all other samples.
                'LOCKHART EUGENE E’: This sample has no data.
                'THE TRAVEL AGENCY IN THE PARK’: This sample doesn’t seem related to Enron.
            
            Therefore, I deleted these three outliers. 
        * The allocation of features across classes(POI/non-POI):
        ![](/Users/xuanguo/Downloads/NANO/Project5/ud120-projects/final_project/Figures/Figure_1.png)
                                                                                            Figure1
        ![](/Users/xuanguo/Downloads/NANO/Project5/ud120-projects/final_project/Figures/Figure_2.png)
                                                                                            Figure2
        ![](/Users/xuanguo/Downloads/NANO/Project5/ud120-projects/final_project/Figures/Figure_3.png)
                                                                                            Figure3
        ![](/Users/xuanguo/Downloads/NANO/Project5/ud120-projects/final_project/Figures/Figure_4.png)
                                                                                            Figure4                        
        From Figure1-4, we can see that total_payments, exercised_stock_options, other and from_this_person_to_poi have similar allocations on POI/non_POI. 



2.What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.  [relevant rubric items: “create new features”, “intelligently select features”, “properly scale features”]

Feature Selection:
* Among all 21 features, there are 4 features that have more than 90 NaNs. They are ‘loan_advances’, which has 142 NaNs,  ‘deferred_income’, which has 97 NaNs, ‘restricted_stock_deferred’, which has 128 NaNs, ‘deferral_payments’, which has 107 NaNs, ‘director_fees’, which has129 NaNs. I deleted these 4 features.

* Also, I deleted the highly-correlated features referring to the correlation matrix. 
```
    cor_matrix = features.corr()
    for i, row in enumerate(cor_matrix.index):
        for j, col in enumerate(cor_matrix.columns):
            if j>=i:
                break
            temp = cor_matrix.get_value(row, col)
            if temp > 0.7:
                print row, 'and', col, 'have correlation greater than 0.7'
```
```
    shared_receipt_with_poi and to_messages have correlation greater than 0.7
    total_stock_value and exercised_stock_options have correlation greater than 0.7
    total_stock_value and restricted_stock have correlation greater than 0.7
    other and total_payments have correlation greater than 0.7
    from_poi_to_this_person and shared_receipt_with_poi have correlation greater than 0.7
    poi_this_person and to_messages have correlation greater than 0.7
    poi_this_person and shared_receipt_with_poi have correlation greater than 0.7
    poi_this_person and from_this_person_to_poi have correlation greater than 0.7
    poi_this_person and from_poi_to_this_person have correlation greater than 0.7
```
    The features I ended up using are: 'salary', 'total_payments', 'bonus', 'total_stock_value', 'expenses', 'from_messages', 'long_term_incentive'(7 features in total)
    
* I use scaling only in svm, linear svm and Nearest Neighbors model because these three models operate in the coordinate plane.
* I created two new features:
ratio_payments_stock: the ratio of ‘total_payments’ and ‘total_stock_value’. Since ‘total_payments’ and ‘total_stock_value’ are all the sum of other features, the ratio of them perhaps could be a good substitute to them. 
poi_this_person: the sum of ‘from_poi_to_this_person’ and ‘from_this_person_to_poi’. Perhaps the sum of the emails could substitute the two features and simplify the model. 
*  In every estimator, I applied pca and SelectKBest packed as FeatureUnion into it. And for each of the FeatureUnion, I have the parameters set as: N_Options =  np.array([2,4,6])
    K_Options =  np.array([2,4,6]). I tested all the parameters in the GridsearchCV. Therefore, I can’t get feature scores from SelectKBest because it is combined in FeatureUnion with pca in each model in my project. 
* I use pca before SelectKBest in Decision Tree model. The best parameters through GridsearchCv is as below:
    ```
   Model tree report
    Pipeline(steps=[('dim_reduct', FeatureUnion(n_jobs=1,
       transformer_list=[('pca', PCA(copy=True, iterated_power='auto', n_components=4, random_state=42,
    svd_solver='auto', tol=0.0, whiten=False)), ('slt', SelectKBest(k=6, score_func=<function f_classif at 0x109503140>))],
       transformer_weights=No...it=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=42, splitter='random'))])
    ```
    The feature imporatances of the combined features(from pca and SelectKBest) delivered by PCA and SelectKBest are: [ 0.  0.08123249  0.02189781  0.09067251  0.03604759  0.01633803  0.26202001  0.0952381   0.35484223  0.04171123]
    
3.What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?  [relevant rubric item: “pick an algorithm”]

I ended up using Decision Tree model because it has good performance on precision and recall. I've tried Naive Bayes, KNN, SVM, linear SVM and random forest models. Those models have different performance on recall and precision. 


Model|Parameters|Precision|Recall
-----|----------|---------|------
Naive Bayes|Pipeline(steps=[('dim_reduct', FeatureUnion(n_jobs=1,transformer_list=[('pca', PCA(copy=True, iterated_power='auto', n_components=4, random_state=42,svd_solver='auto', tol=0.0, whiten=False)), ('slt', SelectKBest(k=2, score_func=<function f_classif at 0x108cf1140>))],transformer_weights=None)), ('clf', GaussianNB(priors=None))])|0.37441|0.27950
KNN(minmax scaled)|Pipeline(steps=[('scaler', MinMaxScaler(copy=True, feature_range=(0, 1))), ('dim_reduct', FeatureUnion(n_jobs=1,transformer_list=[('pca', PCA(copy=True, iterated_power='auto', n_components=4, random_state=42,svd_solver='auto', tol=0.0, whiten=False)), ('slt', SelectKBest(k=2, score_func=<function ...owski',metric_params=None, n_jobs=1, n_neighbors=2, p=2,weights='uniform'))])|0.26267|0.0570
KNN(std scaled)|Pipeline(steps=[('scaler', StandardScaler(copy=True, with_mean=True, with_std=True)), ('dim_reduct', FeatureUnion(n_jobs=1,transformer_list=[('pca', PCA(copy=True, iterated_power='auto', n_components=6, random_state=42,svd_solver='auto', tol=0.0, whiten=False)), ('slt', SelectKBest(k=2, score_func...owski',metric_params=None, n_jobs=1, n_neighbors=7, p=1,weights='uniform'))])|0.06897|0.0040
SVM(minmax scaled)|Pipeline(steps=[('scaler', MinMaxScaler(copy=True, feature_range=(0, 1))), ('dim_reduct', FeatureUnion(n_jobs=1,transformer_list=[('pca', PCA(copy=True, iterated_power='auto', n_components=2, random_state=42,svd_solver='auto', tol=0.0, whiten=False)), ('slt', SelectKBest(k=2, score_func=<function ...d',max_iter=-1, probability=False, random_state=42, shrinking=True,tol=0.001, verbose=False))])|0.87500|0.01750
SVM(std scaled)|Pipeline(steps=[('scaler', StandardScaler(copy=True, with_mean=True, with_std=True)), ('dim_reduct', FeatureUnion(n_jobs=1,transformer_list=[('pca', PCA(copy=True, iterated_power='auto', n_components=6, random_state=42,svd_solver='auto', tol=0.0, whiten=False)), ('slt', SelectKBest(k=6, score_func...d',max_iter=-1, probability=False, random_state=42, shrinking=True,tol=0.001, verbose=False))])|0.42614|0.11250
Linear SVM(minmax scaled)|Pipeline(steps=[('scaler', MinMaxScaler(copy=True, feature_range=(0, 1))), ('dim_reduct', FeatureUnion(n_jobs=1,transformer_list=[('pca', PCA(copy=True, iterated_power='auto', n_components=6, random_state=42,svd_solver='auto', tol=0.0, whiten=False)), ('slt', SelectKBest(k=2, score_func=<function ...inge', max_iter=500, multi_class='ovr',penalty='l2', random_state=42, tol=0.0001, verbose=0))])|0.01515|0.00050
Linear SVM(std scaled)|Pipeline(steps=[('scaler', StandardScaler(copy=True, with_mean=True, with_std=True)), ('dim_reduct', FeatureUnion(n_jobs=1,transformer_list=[('pca', PCA(copy=True, iterated_power='auto', n_components=2, random_state=42,svd_solver='auto', tol=0.0, whiten=False)), ('slt', SelectKBest(k=4, score_func... max_iter=1000,multi_class='ovr', penalty='l2', random_state=42, tol=0.0001,verbose=0))])|0.24038|0.24050
Decision Tree|Pipeline(steps=[('dim_reduct', FeatureUnion(n_jobs=1,transformer_list=[('pca', PCA(copy=True, iterated_power='auto', n_components=4, random_state=42,svd_solver='auto', tol=0.0, whiten=False)), ('slt', SelectKBest(k=6, score_func=<function f_classif at 0x108cf1140>))],transformer_weights=No...it=2, min_weight_fraction_leaf=0.0,presort=False, random_state=42, splitter='random'))])|0.35672|0.32800
Random Forest|Pipeline(steps=[('dim_reduct', FeatureUnion(n_jobs=1,transformer_list=[('pca', PCA(copy=True, iterated_power='auto', n_components=2, random_state=42,svd_solver='auto', tol=0.0, whiten=False)), ('slt', SelectKBest(k=6, score_func=<function f_classif at 0x108cf1140>))],transformer_weights=No...estimators=5, n_jobs=1,oob_score=False, random_state=42, verbose=0, warm_start=False))])|0.27678|0.17700

4.What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? What parameters did you tune? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier).  [relevant rubric items: “discuss parameter tuning”, “tune the algorithm”]

By tuning the parameters of an algorithm, I could find the optimal set of parameter set that fits the model best. If I don’t do the tuning well, the model won’t work even given tons of training data. 
eg. SVM model:
```
N_Options = np.array([2,4,6])
K_Options = np.array([2,4,6])

dim_reduction = FeatureUnion([('pca', pca), ("slt", slt)])

svm = SVC(random_state = 42)
C_Options = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
Kernel_Options = ['rbf','sigmoid']
Gamma_Options = [0.001, 0.01, 0.1, 1,'auto']

svc_param = {'dim_reduct__pca__n_components':N_Options,\
                  'dim_reduct__slt__k': K_Options,\
                  'clf__C': C_Options,\
                  'clf__kernel': Kernel_Options,\
                  'clf__gamma': Gamma_Options}
```
                  
I tuned the pca N_Options so that I could get the right numbers of components to keep; SelectKBest K to get K number of features selected; clf_C to decide penalty parameter C of the error term; clf__kernel to choose the proper kernel type for the model; clf_gamma to decide the right kernel coefficient.  

5.What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?  [relevant rubric items: “discuss validation”, “validation strategy”]

In machine learning, validation is the process that a trained model is evaluated by test data. The model could be overfitted and have high variance without correct validation. I validated my analysis using test.py, where I use cross validation function: StratifiedShuffleSplit() to get 1000 folds of training data and test data. By applying the model to fit the training data and predict test data for the 1000 folds, I could get the total true negatives, false_negatives, false_positives and true_positives, which could be used to calculate important metrics like accuracy, precision, recall, etc. Through comparing the metrics of each model, I could validate the models.


6.Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance. [relevant rubric item: “usage of evaluation metrics”]

The evaluation metrics and the average performance are showed in the table in question3. 
In the table, the precision and recall of the Decision Tree model are 0.35672, 0.32800, which means that among the positive predictions of POI given by the model, 35.672% of them are correct; among the POIs, about 32.800% can be correctly predicted by the trained model. 

Reference:

https://discussions.udacity.com/t/what-is-the-proper-way-to-handle-nan/24593

http://scikit-learn.org/stable/


