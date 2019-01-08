'''
Whois Feature Set
Tthe registration, update, and expiration dates,
a bag-of- words representation of the registrar and registrant.
'''

import whois
from dns import query
from urllib.parse import urlparse
import whois
import tldextract
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 10000)
pd.set_option('display.max_rows', 10000)
pd.set_option('display.max_colwidth', 10000)
pd.set_option('display.width',1000)
np.set_printoptions(threshold=np.inf)

All_Known_TLD = ['com', 'at', 'uk', 'pl', 'be', 'biz', 'co', 'jp', 'co_jp', 'cz', 'de', 'eu', 'fr', 'info', 'it', 'ru', 'lv', 'me', 'name', 'net', 'nz', 'org', 'us']

dataset_path = '../Data/train_dataset.csv'
registrar_path = '../Data/registrar.txt'
Whois_Feature_path = 'Whois_FeatureSet.npy'


# Whois Creation date
def whois_creation_date(url):

    try:
        cd = whois.query(url).creation_date
        if cd is not None or 'None':
            year = str(cd.year)
            month = str(cd.month)
            if len(month) == 1:
                month = '0' + month
            day = str(cd.day)
            if len(day) == 1:
                day = '0'+ day
            return (year+month+day)
        else:
            return 0
    except Exception as e:
            return 0

# Whois Update date
def whois_update_date(url):

    try:
        cd = whois.query(url).last_updated
        if cd is not None or 'None':
            year = str(cd.year)
            month = str(cd.month)
            if len(month) == 1:
                month = '0' + month
            day = str(cd.day)
            if len(day) == 1:
                day = '0'+ day
            return (year+month+day)
        else:
            return 0
    except Exception as e:
            return 0

# Whois expiration date
def whois_expiration_date(url):

    try:
        cd = whois.query(url).expiration_date
        if cd is not None or 'None':
            year = str(cd.year)
            month = str(cd.month)
            if len(month) == 1:
                month = '0' + month
            day = str(cd.day)
            if len(day) == 1:
                day = '0'+ day
            return (year+month+day)
        else:
            return 0
    except Exception as e:
            return 0

# Registrar
def is_registrar_list(url):

    try:
        cd = whois.query(url).registrar
        if cd in registrar_list:
            return 1
        else:
            return 0
    except Exception as e:
            return 0


if __name__ == '__main__':

    ## 1. Read the training Dataset file
    df = pd.read_csv(dataset_path, header=0)
    # print(df.head())

    ## 1.1 Read the registrar.txt
    registrar_list = []
    with open(registrar_path,'r') as f:
        for i in f.readlines():
            registrar_list.append(i)

    print('len(registrar_list) = ',len(registrar_list))


    ## 2. Get the Basic Feature set
    url = df['URL']
    creation_date = []
    last_updated = []
    expiration_date = []
    registrar = []
    registrant = []

    for i in url:
        creation_date.append(whois_creation_date(i))
        last_updated.append(whois_update_date(i))
        expiration_date.append(whois_expiration_date(i))
        registrar.append(is_registrar_list(i))

    registrant = registrar
    print(creation_date)
    print(last_updated)

    ## 3. Form the Whois Feature Set
    Whois_Feature = np.array((creation_date,last_updated,expiration_date,registrar,registrant)).T

    # print('Whois_Feature.shape=',Whois_Feature.shape)
    # (7000,5)

    ## 4. Save the Whois Feature set
    np.save(Whois_Feature_path, Whois_Feature)

    ## 5. Load the Basic Feature set
    whois_feature = np.load(Whois_Feature_path)
    # print('whois_feature.shape=',whois_feature.shape)
    # print(whois_feature)
