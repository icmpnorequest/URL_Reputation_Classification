'''
Basic Feature set
1. Total dom num;
2. Is IP in hostname;
3. The creation date of URL;
4. Is there a creation date for URL.
'''

from urllib.parse import urlparse
import whois
import tldextract
import pandas as pd
import numpy as np
from tempfile import TemporaryFile

pd.set_option('display.max_columns', 10000)
pd.set_option('display.max_rows', 10000)
pd.set_option('display.max_colwidth', 10000)
pd.set_option('display.width',1000)
np.set_printoptions(threshold=np.inf)

All_Known_TLD = ['com', 'at', 'uk', 'pl', 'be', 'biz', 'co', 'jp', 'co_jp', 'cz', 'de', 'eu', 'fr', 'info', 'it', 'ru', 'lv', 'me', 'name', 'net', 'nz', 'org', 'us']

dataset_path = '../Data/train_dataset.csv'
Basic_Feature_path = 'Basic_FeatureSet.npy'

# Get total dot num in URL
def get_dot_num(url):

    dot = '.'
    count = 0
    for i in url:
        if i == dot:
            count += 1
    return count

# Whether IP in host-name.
# If IP in host-name, return 0; else 1
def is_ip_hostname(url):

    hostname = urlparse(url).netloc
    for i in hostname:
        if i.isdigit() == False:
            return 0
    else:
        return 1

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

# Is Creation date
def is_creation_date(url):

    cd = whois_creation_date(url)
    if cd is not 0:
        return 1
    else:
        return 0

if __name__ == '__main__':

    ## 1. Read the training Dataset file
    df = pd.read_csv(dataset_path, header=0)
    # print(df.head())

    ## 2. Get the Basic Feature set
    url = df['URL']
    total_dot = []
    is_ip_in_hostname = []
    creation_date = []
    is_creation_date_value = []

    for i in url:
        total_dot.append(get_dot_num(i))
        is_ip_in_hostname.append(is_ip_hostname(i))
        creation_date.append(whois_creation_date(i))
        # if whois_creation_date(i) is not 0:
            # print('creation date: ',whois_creation_date(i))
            # print('url index', creation_date.index(whois_creation_date(i)))
            # creation date / index: 20070905 / 1482
            # creation date / index: 19860902 / 6734
        is_creation_date_value.append(is_creation_date(i))

    # print(len(total_dot))
    # print(len(is_ip_in_hostname))
    # print(len(creation_date))
    # print(len(is_creation_date_value))
    # print('total dot: ',total_dot[:10])
    # print('is ip in hostname: ',is_ip_in_hostname[:10])
    # print('creation date: ',creation_date[:10])
    # print('is creation date: ',is_creation_date_value[:10])


    ## 3. Form the Basic Feature Set
    Basic_Feature = np.array((total_dot,is_ip_in_hostname,creation_date,is_creation_date_value)).T

    # print('Basic_Feature.shape=',Basic_Feature.shape)
    # (4,7000)


    ## 4. Save the Basic Feature set
    np.save('Basic_FeatureSet.npy', Basic_Feature)


    ## 5. Load the Basic Feature set
    basic = np.load(Basic_Feature_path)
    # print('basic.shape=',basic.shape)
    # print(basic)