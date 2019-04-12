''' Import required libraries in python '''

import os

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import RandomForestClassifier
#%matplotlib inline
import matplotlib.pyplot as plt


#import matplotlib.pyplot as plt

'''  Creating the working  directory '''

os.chdir("F:\\Poverty T-Test")

DATA_DIR = os.path.join('F:\\Poverty T-Test', 'data', 'processed')

data_paths__indiv = {'A': {'train': os.path.join(DATA_DIR, 'A', 'A_indiv_train.csv'), 
                    'test':  os.path.join(DATA_DIR, 'A', 'A_indiv_test.csv')}, 
              
              'B': {'train': os.path.join(DATA_DIR, 'B', 'B_indiv_train.csv'), 
                    'test':  os.path.join(DATA_DIR, 'B', 'B_indiv_test.csv')}, 
              
              'C': {'train': os.path.join(DATA_DIR, 'C', 'C_indiv_train.csv'), 
                    'test':  os.path.join(DATA_DIR, 'C', 'C_indiv_test.csv')}}

''' load testing data '''
a_test_indiv = pd.read_csv(data_paths__indiv['A']['test'], index_col='id')
b_test_indiv = pd.read_csv(data_paths__indiv['B']['test'], index_col='id')
c_test_indiv = pd.read_csv(data_paths__indiv['C']['test'], index_col='id')

''' load training data '''
a_train_indiv = pd.read_csv(data_paths__indiv['A']['train'], index_col='id')
b_train_indiv = pd.read_csv(data_paths__indiv['B']['train'], index_col='id')
c_train_indiv = pd.read_csv(data_paths__indiv['C']['train'], index_col='id')


data_paths = {'A': {'train': os.path.join(DATA_DIR, 'A', 'A_hhold_train.csv'), 
                    'test':  os.path.join(DATA_DIR, 'A', 'A_hhold_test.csv')}, 
              
              'B': {'train': os.path.join(DATA_DIR, 'B', 'B_hhold_train.csv'), 
                    'test':  os.path.join(DATA_DIR, 'B', 'B_hhold_test.csv')}, 
              
              'C': {'train': os.path.join(DATA_DIR, 'C', 'C_hhold_train.csv'), 
                    'test':  os.path.join(DATA_DIR, 'C', 'C_hhold_test.csv')}}

''' load testing data '''
a_test = pd.read_csv(data_paths['A']['test'], index_col='id')
b_test = pd.read_csv(data_paths['B']['test'], index_col='id')
c_test = pd.read_csv(data_paths['C']['test'], index_col='id')

''' load training data '''
a_train = pd.read_csv(data_paths['A']['train'], index_col='id')
b_train = pd.read_csv(data_paths['B']['train'], index_col='id')
c_train = pd.read_csv(data_paths['C']['train'], index_col='id')


'''Preview the data sets'''

a_train.head()
b_train.head()
c_train.head()

a_test.head()
b_test.head()
c_test.head()

#combined = pd.merge(df1, df2, how='left', on=['Year', 'Week', 'Colour'])


''' Standardize the numeric variables to avoid bias. '''

def normalize(data_frame, numeric_only=True):
    numeric = data_frame.select_dtypes(include=['int64', 'float64'])
    data_frame[numeric.columns] = (numeric - numeric.mean()) / numeric.std()
    return (data_frame)
    

def pre_process_data(data_frame, enforce_cols=None):
    print("Input shape:\t{}".format(data_frame.shape))
        

    data_frame = normalize(data_frame)
    print("After standardization {}".format(data_frame.shape))
        
    ''' create dummy variables for categoricals '''
    
    data_frame = pd.get_dummies(data_frame)
    print("After converting categoricals:\t{}".format(data_frame.shape))
    

    ''' match test set and training set columns '''
    
    if enforce_cols is not None:
        to_drop = np.setdiff1d(data_frame.columns, enforce_cols)
        to_add = np.setdiff1d(enforce_cols, data_frame.columns)

        data_frame.drop(to_drop, axis=1, inplace=True)
        data_frame = data_frame.assign(**{c: 0 for c in to_add})
    
    data_frame.fillna(0, inplace=True)
    
    return (data_frame)


print("Country A")
aX_train = pre_process_data(a_train.drop('poor', axis=1))
ay_train = np.ravel(a_train.poor)

print("\nCountry B")
bX_train = pre_process_data(b_train.drop('poor', axis=1))
by_train = np.ravel(b_train.poor)

print("\nCountry C")
cX_train = pre_process_data(c_train.drop('poor', axis=1))
cy_train = np.ravel(c_train.poor)


'''MODELLING '''

def train_model(features, labels, **kwargs):
    
    # instantiate model
    model = RandomForestClassifier(n_estimators=150, random_state=1)
    
    # train model
    model.fit(features, labels)
    
    # get a (not-very-useful) sense of performance
    accuracy = model.score(features, labels)
    print("In-sample accuracy:", accuracy)
    
    return model

model_a = train_model(aX_train, ay_train)

model_b = train_model(bX_train, by_train)

model_c = train_model(cX_train, cy_train)


aX_train.to_csv("aX_train.csv")
# load test data
a_test = pd.read_csv(data_paths['A']['test'], index_col='id')
b_test = pd.read_csv(data_paths['B']['test'], index_col='id')
c_test = pd.read_csv(data_paths['C']['test'], index_col='id')


# process the test data
a_test = pre_process_data(a_test, enforce_cols=aX_train.columns)
b_test = pre_process_data(b_test, enforce_cols=bX_train.columns)
c_test = pre_process_data(c_test, enforce_cols=cX_train.columns)


# compute classification accuracy for the logistic regression model

def train_knn_model(train_data,labels_data,test_data, **kwargs):
    y = labels_data['poor'].astype(int)      
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(train_data, y)
    y_pred = knn.predict(test_data)
    return (y_pred)

'''predictions in true and false format'''

pred_a = train_knn_model(aX_train, a_train, a_test)
pred_b = train_knn_model(bX_train, b_train, c_test)
pred_c = train_knn_model(cX_train, b_train, c_test)

'''prediction probabilities'''
def run_prob_cv(train_data, labels_data, test_data, clf_class, n_estimators):
    y = labels_data['poor'].astype(int)  
   # y_prob = np.zeros((len(y),2))
    clf = clf_class(n_estimators)
    clf.fit(train_data,y)
        # Predict probabilities, not classes
    y_prob = clf.predict_proba(test_data)
    return y_prob
      
#run_prob_cv(aX_train, a_train, a_test,KNN)

y = a_train['poor'].astype(int) 
pred_prob_a = run_prob_cv(aX_train, a_train, a_test,KNN,n_estimators=50)
pred_poor_a = pred_prob_a[:,1]


y = b_train['poor'].astype(int) 
pred_prob_b = run_prob_cv(bX_train, b_train, b_test,KNN,n_estimators=60)
pred_poor_b = pred_prob_b[:,1]

y = c_train['poor'].astype(int) 
pred_prob_c = run_prob_cv(cX_train, c_train, c_test,KNN,n_estimators=81)
pred_poor_c = pred_prob_c[:,1]


def make_country_sub(preds, test_feat, country):
    
    ''' make sure we code the country correctly '''
    country_codes = ['A', 'B', 'C']
    
    ''' get just the poor probabilities '''
    country_sub = pd.DataFrame(data=preds[:,1],  # proba p=1
                               columns=['poor'], 
                               index=test_feat.index)

    
    ''' add the country code for joining later '''
    country_sub["country"] = country
    return country_sub[["country", "poor"]]

''' converting arrays to a data frame '''

a_sub = make_country_sub(pred_prob_a, a_test, 'A')
b_sub = make_country_sub(pred_prob_b, b_test, 'B')
c_sub = make_country_sub(pred_prob_c, c_test, 'C')

''' Export the cleaned data sets'''
 
aX_train.to_csv("aX_train.csv")
bX_train.to_csv("bX_train.csv")
cX_train.to_csv("cX_train.csv")

a_test.to_csv('a_test.csv')
b_test.to_csv('b_test.csv')
c_test.to_csv('c_test.csv')

a_preds = model_a.predict_proba(a_test)
b_preds = model_b.predict_proba(b_test)
c_preds = model_c.predict_proba(c_test)



predictions = pd.concat([a_sub, b_sub, c_sub])

predictions.tail()

predictions.to_csv('predictions___39.csv')


''' MORE PREDICTIONS AND ANALYSIS ON THE DATA'''

''' COUNTRY A'''

''' Number of times a predicted probability is assigned to an observation'''
is_poor_a = y == 1
counts = pd.value_counts(pred_poor_a)

''' calculate true probabilities'''
true_prob = {}
for prob in counts.index:
    true_prob[prob] = np.mean(is_poor_a[pred_poor_a == prob])
    true_prob = pd.Series(true_prob)

# pandas-fu
counts = pd.concat([counts,true_prob], axis=1).reset_index()
counts.columns = ['pred_prob', 'count', 'true_prob']
counts

''' Number of times a predicted probability is assigned to an observation'''
is_poor_a = y == 1
counts = pd.value_counts(pred_poor_a)

''' calculate true probabilities'''
true_prob = {}
for prob in counts.index:
    true_prob[prob] = np.mean(is_poor_a[pred_poor_a == prob])
    true_prob = pd.Series(true_prob)

# pandas-fu
counts = pd.concat([counts,true_prob], axis=1).reset_index()
counts.columns = ['pred_prob', 'count', 'true_prob']
counts

""" COUNTRY B """

''' Number of times a predicted probability is assigned to an observation'''
is_poor_b = y == 1
counts = pd.value_counts(pred_poor_b)

''' calculate true probabilities'''
true_prob = {}
for prob in counts.index:
    true_prob[prob] = np.mean(is_poor_b[pred_poor_b == prob])
    true_prob = pd.Series(true_prob)

# pandas-fu
counts = pd.concat([counts,true_prob], axis=1).reset_index()
counts.columns = ['pred_prob', 'count', 'true_prob']
counts

""" COUNTRY C """

''' Number of times a predicted probability is assigned to an observation'''
is_poor_c = y == 1
counts = pd.value_counts(pred_poor_c)

''' calculate true probabilities'''
true_prob = {}
for prob in counts.index:
    true_prob[prob] = np.mean(is_poor_c[pred_poor_c == prob])
    true_prob = pd.Series(true_prob)

# pandas-fu
counts = pd.concat([counts,true_prob], axis=1).reset_index()
counts.columns = ['pred_prob', 'count', 'true_prob']
counts


''' Plots '''
a_train.poor.value_counts().plot.bar(title='Number of Poor for country A')
b_train.poor.value_counts().plot.bar(title='Number of Poor for country B')
c_train.poor.value_counts().plot.bar(title='Number of Poor for country C')