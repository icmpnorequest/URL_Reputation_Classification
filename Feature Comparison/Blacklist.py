'''
The “Blacklist” feature set consists of binary variables for membership in six blacklists
(and one white list) run by SORBS, URIBL, SURBL, and Spamhaus.
'''

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
# blacklist_path = '../Data/blackurls.txt'
blacklist_path = '../Data/url_blacklists.txt'
Blacklist_Feature_path = 'Blacklist_FeatureSet.npy'

# Read the blackurls
blacklist = []
with open(blacklist_path,'r') as f:
    for i in f.readlines():
        blacklist.append(i)

# print(len(blacklist))

# print(len(blacklist))
# 1418
# print(blacklist)
# ['.1337x.pl\n', '.1link.io\n', '.1n.pm\n', '.1q2w3.fun\n', '.22apple.com\n', ...]

# Whether url in blacklist
def is_in_blacklist(url):

    if url in blacklist:
        return 1
    else:
        return 0


if __name__ == '__main__':

    ## 1. Read the training Dataset file and blacklist
    df = pd.read_csv(dataset_path, header=0)
    # print(df.head())




    ## 2. Get the Basic Feature set
    url = df['URL']
    is_blacklist = []

    for i in url:
        if i in blacklist:
            is_blacklist.append(1)
        else:
            is_blacklist.append(0)

    # print(len(is_blacklist))
    # print(is_blacklist)

    ## 3. Form the Basic Feature Set
    Blacklist_Feature = np.array((is_blacklist))

    ## 4. Save the Basic Feature set
    np.save('Blacklist_FeatureSet.npy', Blacklist_Feature)

    ## 5. Load the Basic Feature set
    url_blacklist = np.load(Blacklist_Feature_path)
    print('basic.shape=',url_blacklist.shape)
    # print(url_blacklist)
