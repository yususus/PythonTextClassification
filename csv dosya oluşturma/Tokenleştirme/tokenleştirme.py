import pandas as pd
from nltk.tokenize import word_tokenize


veri_seti = pd.read_csv('veri_seti_yeni_adlar.csv')

# Tokenleştirme fonksiyonu
def tokenleştir(metin):
    return word_tokenize(metin)


veri_seti['Yorumlar'] = veri_seti["yorum"]
veri_seti['Duygular'] = veri_seti["duygu"]

# farklı csv dosyasında kaydetme işlemi
veri_seti.to_csv('tokenleştirilmis.csv', index=False)
