# URL_Reputation_Classification
## 1. Project Decription
It's an experiment based on a 09 KDD paper, Beyond Blacklists: Learning to Detect Malicious Web Sites from Suspicious URLs.

You can read and download this paper via https://cseweb.ucsd.edu/~voelker/pubs/mal-url-kdd09.pdf

## 2. Experiment Purpose
1) Try to build 9 Feature Sets as the paper do (Feature Comparison);
2) Using several classifiers(Naive Bayes, SVM and Logistic Regression) to validate the result under 9 feature set (Classification Comparison).

## 3. URL Dataset Resource
URL dataset comes from https://github.com/Anmol-Sharma/URL_CLASSIFICATION_SYSTEM/blob/master/train_dataset.csv.
Benign URL comes from DMOZ and Malicious URL comes from Phishtank.

URL dataset is Data/train_dataset.csv
Blacklist is Data/url_blacklists.txt

Get_registrar.py aims to scrap the legal registrars from ICANN and the data has been saved as Data/registrar.txt

## 4. Feature Comparison
Build the 9 Feature Sets as paper do.
1) Basic Feature Set
2) Botnet Feature Set
3) Blacklist Feature Set
4) Blacklist + Botnet Feature Set
5) Whois Feature Set
6) Host-based Feature Set
7) Lexical Feature Set
8) Full (Lexical + Host-based) Feature Set
9) Full except Blacklist + Whois Feature Set

## 5. Classification Comparison
Using sklearn packages https://scikit-learn.org/stable/. 

Classification code is in main.py

Classifier:
1) Gaussian Naive Bayes
2) Multinomial Naive Bayes
3) Linear SVM
4) RBF SVM
5) L1-regularization Logistic Regression

If you have any questions, please feel free to issue me.
Plus, if you like the project, you can make a star for me hah.
Thanks in advance!
