# PythonAI

VERİ ÇEKME CSV DOSYA ÇEVİRME İŞLEMİ
Veri çekme işlemleri Haberler.com adresinden belirli html tagleri kullanılarak çekilmiştir çünkü herhangi bir api hizmeti sunan web sitesi bulunmamaktadır. Kodlar python yazılım dili kullanarak yazılmıştır.
Kodlarımda haber içeriğini yapılan her yorum çekiyorum.

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



Çekilen verileri tablo şeklinde oluşturduğum için kodlarda bu şekilde bir tablo çıktısı alabiliyorum. Yapılan her yorumu haber içeriği ile birlikte farklı satırlarda kaydettim.
Yapılan yorumlar ve haber içerikleri çok uzun olduğu için tam gözükmemekte.

 



Çalışma Sıram: 
İlk önce haberler.com’dan farklı kategorilere ait haberler ve ve bu haberlere yapılan yorumları çektim. 
Kemik yıldız internet sitesinden duygu analizi için 17bin tweet dosyasını indirmek ve kullanmak için mail attım ve dosyayı aldım. Aldığım bu verilerden test_tweets adlı dosyadaki verileri tokenleştirdim ardından da word2vec yöntemi kullanarak sayılaştırdım. Bu sayılaştırdığım veriyi .bin uzantılı bir dosyada sakladım.
Daha sonrasında sonucu gösterecek kodlarımda 
Word2Vec modelini dosyaya ekledim. 
Metin verileri kelime vektörlerinin ortalamaları ile sayılaştırdım. Bu kelimelerin bir temsilini veren vektöre dönüştürülmesini sağlar.
Eğitim verilerinide  dosyaya dahil ettim.
Metin sınıflandırma modelini oluşturur. Bu kodda, basit bir Stochastic Gradient Descent (SGD) sınıflandırıcısı kullanılır.
Eğitim verilerini kullanarak modelimi eğitiyorum.
Bu aşamada kendi verilerimi ekledim ve bu verileride sayılaştırdım.
Eğitilmiş verilere göre kendi verilerim olumlu olumsuz nötr şeklinde sınıflandırılır.
Çıktı olarak yorumu ve yoruma yapılan değerlendirme pycharm terminalinde yazdırdım.
Aşağıda, kodda kullanılan bazı önemli işlevler açıklanmıştır:

vectorize_text() işlevi, metin verilerini sayısallaştırır. Bu işlev, metin örneğini kelimelere ayırır ve her kelimenin Word2Vec modelinde karşılık gelen vektörünü alır. Eğer kelime modelde yoksa, sıfır vektörü döndürür.
SGDClassifier() sınıfı, basit bir Stochastic Gradient Descent sınıflandırıcısını temsil eder. Bu sınıflandırıcı, eğitim verilerini kullanarak vektörleri sınıflandırmak için öğrenen bir ağırlık matrisi kullanır.

Yorumları değerlendirdiğim kodlar şu şekilde:
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











Kodların çıktısı ise şu şekilde:
<img width="786" alt="Ekran Resmi 2023-12-13 16 57 10" src="https://github.com/yususus/PythonAI/assets/77053475/b0db1e40-f733-49c9-81b7-7302aff0b79a">





 













Bu işlemleri haber kategorizasyonu içinde tekrar yaptım ancak haber veri setim olmadığı için haber veri setini kendim oluşturmam gerekti. Bunun için yaklaşık 500’e yakın haber verisi çektim bu işlemi yaptığım kodlar ise şu şekilde:
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

       
        url_parts = url.split('/')
        unique_name = url_parts[-2]  

        # Bunları sütun ve satır haline getirdim
        for i in range(max(len(data_class1), len(data_class2))):
            row_class1 = data_class1[i] if i < len(data_class1) else ''
            row_class2 = data_class2[i] if i < len(data_class2) else ''
            writer.writerow([f'{unique_name}_Haber: {row_class1}', row_class2])













Kategorileri belirlediğim kodlarım şu şekilde:
from sklearn.linear_model import SGDClassifier
from gensim.models import Word2Vec
import pandas as pd
import numpy as np

model = Word2Vec.load("modelHaber.bin")
# Metin verilerini sayısallaştır
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

# Tahmin sonuçlarını göster
for comment, pred in zip(X_new, y_pred):
    print(f"Yorum: {comment} \nTahmin: {pred}\n")


Haberleri kategorize ettirdiğim zaman çıktılar bu şekilde olmakta.

<img width="953" alt="Ekran Resmi 2023-12-13 16 46 02" src="https://github.com/yususus/PythonAI/assets/77053475/c289f1ff-4790-40d5-beb6-b60797c07906">
