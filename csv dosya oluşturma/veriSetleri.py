from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from textblob import TextBlob
import pandas as pd

# Veri setini yükle
data = pd.read_csv('test_tweets.csv', header=None)

# Sütun adlarını belirle
data.columns = ['Yorum', 'Duygu']

# Özellikler ve hedef değişkeni ayır
X = data['Yorum']
y = data['Duygu']

# Veriyi eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli oluştur ve eğit
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Tahminleri yap ve doğruluk skorunu hesapla
predictions = model.predict(X_test)
print('Eğitim Verisi Doğruluk Skoru:', accuracy_score(y_test, predictions))

# Veri setini yükle
data = pd.read_csv('Toplu.csv.csv', header=None)

# Sütun adlarını belirle
data.columns = ['Yorum']

# Yorumları al
yorumlar = data['Yorum']

# Her bir yorum için duygu durumunu hesapla
for yorum in yorumlar:
    blob = TextBlob(yorum)
    print(f"Yorum: {yorum}\nDuygu Durumu: {blob.sentiment.polarity}\n")
