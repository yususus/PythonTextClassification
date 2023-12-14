# Python Text Classification

Veri çekme işlemleri Haberler.com adresinden belirli html tagleri kullanılarak çekilmiştir.
Kodlarımda haber içeriğini ve yapılan her yorum çekiyorum.

Data extraction was done from Haberler.com using certain HTML tags.
In my codes, I pull the news content and every comment made.


# News and Comment
'''import requests
from bs4 import BeautifulSoup
import csv
import itertools
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

        // data for csv
            for yorum in yorumlar:
                writer.writerow([haber, yorum])
//url adress
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

'''



 

Yorumları değerlendirdiğim kodlar şu şekilde:
The codes I used to evaluate the comments are as follows:

'''
from sklearn.linear_model import SGDClassifier
from gensim.models import Word2Vec
import pandas as pd
import numpy as np

model = Word2Vec.load("model2.bin")
def vectorize_text(text):
    words = text.split()
    word_vecs = [model.wv[word] for word in words if word in model.wv.key_to_index]
    if not word_vecs:  # Eğer word_vecs boşsa
        return np.zeros(model.vector_size)  
    return np.mean(word_vecs, axis=0)

df_train = pd.read_csv('test_yeni.csv')
X_train = df_train['yorum'].values
y_train = df_train['duygu'].values

X_train_vec = np.array([vectorize_text(text) for text in X_train])
clf = SGDClassifier()
clf.fit(X_train_vec, y_train)

df_new = pd.read_csv('Toplu9.csv')

df_new = df_new.dropna(subset=['Yorum'])
X_new = df_new['Yorum'].values 

X_new_vec = np.array([vectorize_text(text) for text in X_new])

y_pred = clf.predict(X_new_vec)
y_true = df_new['Yorum'].values  

for comment, pred in zip(X_new, y_pred):
    print(f"Yorum: {comment} \nTahmin: {pred}\n")
'''


Kodların çıktısı ise şu şekilde:
When I run, the outputs are like this.
<img width="786" alt="Ekran Resmi 2023-12-13 16 57 10" src="https://github.com/yususus/PythonAI/assets/77053475/b0db1e40-f733-49c9-81b7-7302aff0b79a">








# Haber Kategorizasyonu
500'e yakın haber çekme işlemi gerçekşetirdim. Bu veriler verim setim olacak.
I pulled nearly 500 news stories from the internet and completed the process. These data will be my dataset.

'''
import requests
from bs4 import BeautifulSoup
import csv
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

        // Sayfa içerisindeki p taglarının classlarından veri çekiyorum
        // I am pulling data from the classes of p tags on the page.
        p_tags_class1 = soup.find_all('h1', {'class': 'h1'})
        p_tags_class2 = soup.find_all('p', {'class': 'hbBoxText'})

        data_class1 = [p_tags_class1[0].text] if p_tags_class1 else []
        data_class2 = [p.text for p in p_tags_class2]

       
        url_parts = url.split('/')
        unique_name = url_parts[-2]  

        
        for i in range(max(len(data_class1), len(data_class2))):
            row_class1 = data_class1[i] if i < len(data_class1) else ''
            row_class2 = data_class2[i] if i < len(data_class2) else ''
            writer.writerow([f'{unique_name}_Haber: {row_class1}', row_class2])
'''












Kategorileri belirlediğim kodlarım şu şekilde:
My codes where I determined the categories are as follows:

'''
from sklearn.linear_model import SGDClassifier
from gensim.models import Word2Vec
import pandas as pd
import numpy as np

model = Word2Vec.load("modelHaber.bin")
// Metin verilerini sayısallaştır
// Digitize text data
def vectorize_text(text):
    words = text.split()
    word_vecs = [model.wv[word] for word in words if word in model.wv.key_to_index]
    if not word_vecs:  # Eğer word_vecs boşsa
        return np.zeros(model.vector_size)  
    return np.mean(word_vecs, axis=0)

df_train = pd.read_csv('Haber2.csv')
X_train = df_train['Yorum'].values
y_train = df_train['Haber'].values

X_train_vec = np.array([vectorize_text(text) for text in X_train])
clf = SGDClassifier()
clf.fit(X_train_vec, y_train)

df_new = pd.read_csv('Toplu9.csv') 
df_new = df_new.dropna(subset=['Yorum'])
X_new = df_new['Yorum'].values  

X_new_vec = np.array([vectorize_text(text) for text in X_new])

y_pred = clf.predict(X_new_vec)
y_true = df_new['Yorum'].values  

// Tahmin sonuçlarını göster
// Show prediction results
for comment, pred in zip(X_new, y_pred):
    print(f"Yorum: {comment} \nTahmin: {pred}\n")
'''

Haberleri kategorize ettirdiğim zaman çıktılar bu şekilde olmakta.
When I categorize the news, the outputs are like this.

<img width="953" alt="Ekran Resmi 2023-12-13 16 46 02" src="https://github.com/yususus/PythonAI/assets/77053475/c289f1ff-4790-40d5-beb6-b60797c07906">
