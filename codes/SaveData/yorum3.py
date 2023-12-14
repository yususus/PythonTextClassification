import requests
from bs4 import BeautifulSoup
import csv
import itertools

# URL'lerden veri çekme
def get_data(urls):
    with open('HaberYorum.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Haber', 'Yorum'])

        for url in urls:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')

            haberIcerik = soup.find_all('p', {'class': 'haber_spotu'})
            if haberIcerik:
                haber = haberIcerik[0].text
            else:
                haber = ''

            yorum_ = soup.find_all('p', {'class': 'hbcMsg'})
            yorumlar = list(itertools.chain(*yorum_))

            # Verileri CSV dosyasına kaydet
            for yorum in yorumlar:
                writer.writerow([haber, yorum])

#url adresleri
if __name__ == '__main__':
    urls = [
        "https://www.haberler.com/haberler/marmara-daki-deprem-oncesi-telefonlara-gelen-16593257-haberi/",
        "https://www.haberler.com/ekonomi/turk-is-genel-kurulu-nda-asgari-ucret-aciklamasi-16593330-haberi/",
        "https://www.haberler.com/spor/fatih-terim-in-dolandiricilik-olayiyla-ilgisi-olmadigi-aciklandi-16595459-haberi/",
        "https://www.haberler.com/haberler/gosterisli-hayatiyla-taninan-nevra-bilem-ve-esi-3-16598407-haberi/",
        "https://www.haberler.com/haberler/cumhurbaskani-erdogan-6-yil-sonra-atina-da-16604464-haberi-yorumlari/",
        "https://www.haberler.com/magazin/kismetse-olur-yarismacisi-simge-nur-erkoc-16619544-haberi/",
        "https://www.haberler.com/haberler/okuldan-kovulan-ogretmen-uslanmiyor-simdide-16619909-haberi/",
        "https://www.haberler.com/yasam/bodrum-da-derisi-yuzulmus-tilki-bulundu-16621788-haberi/",
        "https://www.haberler.com/politika/tbmm-de-bayilan-sp-milletvekili-hasan-bitmez-in-16621521-haberi/",
        "https://www.haberler.com/spor/hakem-halil-umut-meler-e-yumruklu-saldirida-16619169-haberi/",
        "https://www.haberler.com/haberler/istanbul-da-devrim-gibi-karar-bu-3-ilceye-ozel-16623264-haberi/"
    ]
    get_data(urls)
