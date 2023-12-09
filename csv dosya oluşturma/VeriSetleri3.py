# Gerekli modülleri içe aktar
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Veri setini yükle
data = pd.read_csv('veri_seti_yeni_adlar.csv', header=None)

# Sütun adlarını belirle
data.columns = ["Turkcell'e kızgınım. Ve bu kızgınlık sanırım ayrılıkla sonlanıcak gibi geliyor bana.Farklı bir operatörün %30'u fazla fiyat teklif ediyorlar", "olumsuz"]

X = data["Turkcell'e kızgınım. Ve bu kızgınlık sanırım ayrılıkla sonlanıcak gibi geliyor bana.Farklı bir operatörün %30'u fazla fiyat teklif ediyorlar"]
y = data["olumsuz"]

# Veriyi eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Modeli oluştur ve eğit
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Toplu veri setini yükle
data = pd.read_csv('Toplu.csv', header=None)

# Yorumları al
yorumlar = data['Yorum']

# Her bir yorum için duygu durumunu tahmin et
for yorum in yorumlar:
    duygu = model.predict([yorum])[0]
    print(f"Yorum: {yorum}\nDuygu Durumu: {duygu}\n")
