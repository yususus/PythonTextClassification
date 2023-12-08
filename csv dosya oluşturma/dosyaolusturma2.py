import requests
from bs4 import BeautifulSoup
import csv

url = "https://www.haberler.com/haberler/marmara-daki-deprem-oncesi-telefonlara-gelen-16593257-haberi/"
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

p_tags_class1 = soup.find_all('p', {'class': 'haber_spotu'})
p_tags_class2 = soup.find_all('p', {'class': 'hbcMsg'})

# Sadece bir tane veri almak için [0] indeksini kullanabilirsiniz
data_class1 = [p_tags_class1[0].text] if p_tags_class1 else []
# Tüm verileri almak için list comprehension kullanabilirsiniz
data_class2 = [p.text for p in p_tags_class2]

with open('DogalAfetler.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Haber', 'Yorum'])

    # Her bir yorumu ayrı bir satıra yaz
    for row_class1 in data_class1:
        writer.writerow([row_class1, ''])  # Haber başlığı

    for row_class2 in data_class2:
        writer.writerow(['', row_class2])  # Yorum başlığı
