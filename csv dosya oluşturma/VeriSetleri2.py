from textblob import TextBlob
import pandas as pd

# Veri setini yükle
data = pd.read_csv('Toplu.csv', header=None)

# Sütun adlarını belirle
data.columns = ['Yorum']

# Yorumları al
yorumlar = data['Yorum']

# Duygu durumlarını saklamak için bir liste oluştur
duygu_durumlari = []

# Her bir yorum için duygu durumunu hesapla
for yorum in yorumlar:
    blob = TextBlob(yorum)
    duygu_durumu = blob.sentiment.polarity
    duygu_durumlari.append(duygu_durumu)

# Duygu durumlarını veri setine ekle
data['Duygu Durumu'] = duygu_durumlari

# Veri setini CSV dosyasına yaz
data.to_csv('DuyguDurumlari.csv', index=False)
