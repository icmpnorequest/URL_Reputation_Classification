import pandas as pd
import numpy as np
import sklearn
import warnings

import sys
# sys.path.append('Feature Comparison/Basic.py')

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


Basic_Feature_path = 'Feature Comparison/Basic_FeatureSet.npy'
Botnet_Feature_path = 'Feature Comparison/Botnet_FeatureSet.npy'
Blacklist_Feature_path = 'Feature Comparison/Blacklist_FeatureSet.npy'
Whois_Feature_path = 'Feature Comparison/Whois_FeatureSet.npy'
Hostbased_Feature_path = 'Feature Comparison/Host_based_FeatureSet.npy'
Lexical_Feature_path = 'Feature Comparison/Lexical_FeatureSet.npy'
Full_ex_wb_path = 'Feature Comparison/Full_except_WB.npy'


warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn", lineno=196)
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn", lineno=433)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn", lineno=436)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn", lineno=438)


pd.set_option('display.max_columns', 10000)
pd.set_option('display.max_rows', 10000)
pd.set_option('display.max_colwidth', 10000)
pd.set_option('display.width',1000)


blacklist_path = 'Data/url_blacklists.txt'
dataset_path = 'Data/train_dataset.csv'


### Read the training Dataset file
df = pd.read_csv(dataset_path, header=0)
label = df['Lable'].values
# print(label.shape)
# (7000,)

# print(df.head(20))



### Read the blackurls
blacklist = []
with open(blacklist_path,'r') as f:
    for i in f.readlines():
        blacklist.append(i)

# print('len(blacklist) = ',len(blacklist))


### Feature Selection
## 1. Basic Feature set
Basic_Feature = np.load(Basic_Feature_path).astype(int)
df_Basic = pd.DataFrame(data=Basic_Feature, columns=['Total dot','Is IP in hostname','Creation date','Is Creation date'])
# print('### Basic Feature Set ###')
# print(df_Basic.head(10))
# print('\n')

## 2. Botnet Feature set
Botnet_Feature = np.load(Botnet_Feature_path).astype(int)
df_Botnet = pd.DataFrame(data=Botnet_Feature, columns=['Is client','is server','Is IP in hostname','Is PTR','Is PTR resolved'])
# print('### Botnet Feature Set ###')
# print(df_Botnet.head(10))
# print('\n')

## 3. Blacklist Feature set
Blacklist_Feature = np.load(Blacklist_Feature_path).astype(int).reshape(7000,1)
df_Blacklist = pd.DataFrame(data=Blacklist_Feature, columns=['Is in blacklist'])
# print('### Blacklist Feature Set ###')
# print(df_Blacklist.head(10))
# print('\n')

## 4. Whois Feature set
Whois_Feature = np.load(Whois_Feature_path).astype(int)
df_Whois = pd.DataFrame(data=Whois_Feature, columns=['Creation date','Last updated','Expiration date','Is registarar','Is registrant'])
# print('### Whois Feature Set ###')
# print(df_Whois.head(10))
# print('\n')

## 5. Host-based Feature set
Hostbased_Feature = np.load(Hostbased_Feature_path).astype(int)
df_Hostbased = pd.DataFrame(data=Hostbased_Feature, columns=['Is in blacklist','Creation date','Last updated','Expiration date','Is registarar','Is registrant','Is client','is server','Is IP in hostname','Is PTR','Is PTR resolved','Is DNS record','TTL record'])
# print(df_Hostbased.columns)
# print('### Hostbased Feature Set ###')
# print(df_Hostbased.head(10))
# print('\n')

## 6. Lexical Feature set
Lexical_Feature = np.load(Lexical_Feature_path).astype(int)
df_Lexical = pd.DataFrame(data=Lexical_Feature, columns=['Total delims','Total hyphens','URL len','Dot num','Host token','Path_token'])
# print(df_Lexical.head(10))

## 7. Lexical + Host-based Feature set
Full_Feature = np.hstack((Lexical_Feature,Hostbased_Feature)).astype(int)
df_Full = pd.DataFrame(data=Full_Feature, columns=['Total delims','Total hyphens','URL len','Dot num','Host token','Path_token','Is in blacklist','Creation date','Last updated','Expiration date','Is registarar','Is registrant','Is client','is server','Is IP in hostname','Is PTR','Is PTR resolved','Is DNS record','TTL record'])
print(len(df_Full.columns))
# 19
# print(df_Full.head(10))

## 8. Full except Whois + Blacklist Feature set
Full_ex_wb = np.load(Full_ex_wb_path).astype(int)
df_Full_except = pd.DataFrame(data=Full_ex_wb, columns=['Total delims','Total hyphens','URL len','Dot num','Host token','Path_token','Is client','is server','Is IP in hostname','Is PTR','Is PTR resolved','Is DNS record','TTL record'])
# print(df_Full_except.head(10))
print(len(df_Full_except.columns))

## 9. Blacklist+Botnet Feature set
Blk_Bot_Feature = np.hstack((Blacklist_Feature,Botnet_Feature)).astype(int)
df_Blk_Bot = pd.DataFrame(data=Blk_Bot_Feature, columns=['Is in blacklist','Is client','is server','Is IP in hostname','Is PTR','Is PTR resolved'])
# print(df_Blk_Bot.head(10))



def Classification_Process(Featureset,label):

    ## 1. [0-1] Normalization
    scaler = MinMaxScaler()

    ## 2. Split training and testing dataset
    X_train, X_test, y_train, y_test = train_test_split(Featureset.astype(float), label, test_size=0.2, random_state=123)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    ## 3. Algorithm
    G_NB = GaussianNB()
    M_NB = MultinomialNB()
    linear_svm = SVC(kernel='linear')
    rbf_svm = SVC(kernel='rbf')
    LR = LogisticRegression(penalty='l1')

    ## 4. Cross validation
    print('####### Cross validation #######')
    print('G_NB cross value: ', cross_val_score(G_NB, X_train, y_train, cv=5, scoring='accuracy').mean())
    print('M_NB cross value: ', cross_val_score(M_NB, X_train, y_train, cv=5, scoring='accuracy').mean())
    print('linear svm cross value: ', cross_val_score(linear_svm, X_train, y_train, cv=5, scoring='accuracy').mean())
    print('rbf svm cross value: ', cross_val_score(rbf_svm, X_train, y_train, cv=5, scoring='accuracy').mean())
    print('LR cross value: ', cross_val_score(LR, X_train, y_train, cv=5, scoring='accuracy').mean())
    print('\n')

    ## 5. Prediction
    y_pred_gnb = G_NB.fit(X_train, y_train).predict(X_test)
    y_pred_mnb = M_NB.fit(X_train, y_train).predict(X_test)
    y_pred_l_svm = linear_svm.fit(X_train, y_train).predict(X_test)
    y_pred_r_svm = rbf_svm.fit(X_train, y_train).predict(X_test)
    y_pred_lr = LR.fit(X_train, y_train).predict(X_test)

    print('######### Prediction ###########')
    print('Gaussian Naive Bayes prediction: ', accuracy_score(y_test, y_pred_gnb))
    print('Multinomial Naive Bayes prediction: ', accuracy_score(y_test, y_pred_mnb))
    print('Linear SVM prediction: ', accuracy_score(y_test, y_pred_l_svm))
    print('RBF SVM prediction: ', accuracy_score(y_test, y_pred_r_svm))
    print('L1-Logistic Regression prediction: ', accuracy_score(y_test, y_pred_lr))

    return 0


############ Classification ###########

print('##### 1. Basic Feature Classification Results #####')
Classification_Process(Basic_Feature,label)
print('\n')


print('##### 2. Botnet Feature Set Classification Results #####')
Classification_Process(Botnet_Feature,label)
print('\n')

print('##### 3. Blacklist Feature Set Classification Results ######')
Classification_Process(Blacklist_Feature,label)
print('\n')

print('##### 4. Blacklist+Botnet Feature Set Classification Results ######')
Classification_Process(Blk_Bot_Feature,label)
print('\n')

print('##### 5. Whois Feature Set Classification Results ######')
Classification_Process(Whois_Feature,label)
print('\n')

print('##### 6. Host-based Feature Set Classification Results #####')
Classification_Process(Hostbased_Feature,label)
print('\n')

print('##### 7. Lexica Feature Set Classification Results #####')
Classification_Process(Lexical_Feature,label)
print('\n')

print('##### 8. Full(Lexica + Host-based) Feature Set Classification Results #####')
Classification_Process(Full_Feature,label)
print('\n')

print('##### 9. Full except Whois+Blacklist Feature Set Classification Results #####')
Classification_Process(Full_ex_wb,label)
print('\n')
