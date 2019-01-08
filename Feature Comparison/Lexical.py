'''
total number of delimeters
total number of hyphens
the length of the hostname
the length of the entire URL
the number of dots
a binary feature for each token in the hostname
a binary feature for each token in the path
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

# List of Suspicious Words Present in URL
Suspicious_Words=['secure','account','update','banking','login','click','confirm','password','verify','signin','ebayisapi','lucky','bonus']

# List of Suspicious Top Level Domains in URLs
Suspicious_TLD=['zip','cricket','link','work','party','gq','kim','country','science','tk']


dataset_path = '../Data/train_dataset.csv'
Lexical_Feature_path = 'Lexical_FeatureSet.npy'


# Calculate the total number delimeters in a URL
def Total_delims(str):

    delim = ['-', '_', '?', '=', '&']
    count = 0
    for i in str:
        for j in delim:
            if i == j:
                count += 1
    return count

# Calculate the total number of hyphens in a URL
def Total_hyphens(link):

    hyph = '-'
    count = 0
    for i in link:
        if i == hyph:
            count += 1
    return count

# Calculate the length of hostname in a URL
def Hostname_len(url):

    hostname = urlparse(url).netloc
    return len(hostname)

# Calculate the length of a URL
def URL_len(url):

    return len(url)

# Calculate the number of dots in a URL
def get_dot_num(url):

    dot = '.'
    count = 0
    for i in url:
        if i == dot:
            count += 1
    return count

# Binary feature for hostname tokens
def is_known_tld(url):

    tld = tldextract.extract(url).suffix
    if tld in All_Known_TLD:
        return 0
    else:
        return 1

# Binary feature for path tokens
def is_known_path(url):

    path = urlparse(url).path
    for i in Suspicious_Words:
        if i in path:
            return 1
        else:
            continue
    return 0


if __name__ == '__main__':

    ## 1. Read the training Dataset file
    df = pd.read_csv(dataset_path, header=0)
    # print(df.head())

    ## 2. Get the Basic Feature set
    url = df['URL']
    total_delims = []
    total_hyphens = []
    url_len = []
    dot_num = []
    host_token = []
    path_token = []

    for i in url:
        total_delims.append(Total_delims(i))
        total_hyphens.append(Total_hyphens(i))
        url_len.append(URL_len(i))
        dot_num.append(get_dot_num(i))
        host_token.append(is_known_tld(i))
        path_token.append(is_known_path(i))


    ## 3. Form the Lexical Feature Set
    Lexical_Feature = np.array((total_delims,total_hyphens,url_len,dot_num,host_token,path_token)).T
    print(Lexical_Feature.shape)
    # print(Lexical_Feature[:10,:])


    ## 4. Save the Basic Feature set
    np.save(Lexical_Feature_path, Lexical_Feature)


    ## 5. Load the Basic Feature set
    lexical = np.load(Lexical_Feature_path)
    print('lexical.shape=',lexical.shape)
    # print(basic)


