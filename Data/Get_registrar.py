from bs4 import BeautifulSoup
import requests

url = 'https://www.icann.org/registrar-reports/accredited-list.html'
headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36'}

# 1. Request
r = requests.get(url, headers = headers)
content = r.text

# 2. Beautifulsoup
soup = BeautifulSoup(content, 'lxml')
registrar = []
registrar_file = 'registrar.txt'

with open(registrar_file,'w') as f:
    for i in soup.find_all('a'):
        f.write(i.string+'\n')
f.close()