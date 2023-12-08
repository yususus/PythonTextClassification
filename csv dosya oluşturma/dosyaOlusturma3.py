import requests
from bs4 import BeautifulSoup
import csv

# Farklı url'lerden veri çekiyorum
urls = [
    "https://www.haberler.com/haberler/marmara-daki-deprem-oncesi-telefonlara-gelen-16593257-haberi/",
    "https://www.haberler.com/ekonomi/turk-is-genel-kurulu-nda-asgari-ucret-aciklamasi-16593330-haberi/",
    "https://www.haberler.com/spor/fatih-terim-in-dolandiricilik-olayiyla-ilgisi-olmadigi-aciklandi-16595459-haberi/",
    "https://www.haberler.com/haberler/gosterisli-hayatiyla-taninan-nevra-bilem-ve-esi-3-16598407-haberi/",
    "https://www.haberler.com/haberler/cumhurbaskani-erdogan-6-yil-sonra-atina-da-16604464-haberi-yorumlari/",
]

with open('Toplu.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Haber', 'Yorum'])

    for url in urls:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Sayfa içerisindeki p taglarının classlarından veri çekiyorum
        p_tags_class1 = soup.find_all('p', {'class': 'haber_spotu'})
        p_tags_class2 = soup.find_all('p', {'class': 'hbcMsg'})

        data_class1 = [p_tags_class1[0].text] if p_tags_class1 else []
        data_class2 = [p.text for p in p_tags_class2]

        # URL'den unique bir isim oluştur
        url_parts = url.split('/')
        unique_name = url_parts[-2]  # Using the second-to-last part of the URL as a unique identifier

        # Bunları sütun ve satır haline getirdim
        for row_class1 in data_class1:
            writer.writerow([f'{unique_name}_Haber', ''])  # Haber başlığı

        for row_class2 in data_class2:
            writer.writerow(['', row_class2])  # Yorum başlığı
