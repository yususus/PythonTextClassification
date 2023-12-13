import requests
from bs4 import BeautifulSoup
import csv

# Farklı url'lerden veri çekiyorum
urls = [
    "https://www.haberler.com/egitim/",
    "https://www.haberler.com/ekonomi/",
    "https://www.haberler.com/magazin/",
    "https://www.haberler.com/spor/",
    "https://www.haberler.com/dunya/",
    "https://www.haberler.com/finans/",
    "https://www.haberler.com/kultur-sanat/",
    "https://www.haberler.com/politika/",
    "https://www.haberler.com/teknoloji/"
]

with open('Haber2.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Haber', 'Yorum'])

    for url in urls:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Sayfa içerisindeki p taglarının classlarından veri çekiyorum
        p_tags_class1 = soup.find_all('h1', {'class': 'h1'})
        p_tags_class2 = soup.find_all('p', {'class': 'hbBoxText'})

        data_class1 = [p_tags_class1[0].text] if p_tags_class1 else []
        data_class2 = [p.text for p in p_tags_class2]

        # URL'den unique bir isim oluştur
        url_parts = url.split('/')
        unique_name = url_parts[-2]  # Using the second-to-last part of the URL as a unique identifier

        # Bunları sütun ve satır haline getirdim
        for i in range(max(len(data_class1), len(data_class2))):
            row_class1 = data_class1[i] if i < len(data_class1) else ''
            row_class2 = data_class2[i] if i < len(data_class2) else ''
            writer.writerow([f'{unique_name}_Haber: {row_class1}', row_class2])
