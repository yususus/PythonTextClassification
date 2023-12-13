from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from sklearn.metrics import accuracy_score


# Eğitim verilerini yükle
df_train = pd.read_csv('test_yeni.csv')
X_train = df_train['yorum'].values
y_train = df_train['duygu'].values

# Metin sınıflandırma modelini oluştur
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
clf = MultinomialNB()
clf.fit(X_train_vec, y_train)

df_new = pd.read_csv('Toplu9.csv')  # Yeni veri setini yükle

# NaN değerleri temizle
df_new = df_new.dropna(subset=['Yorum'])

X_new = df_new['Yorum'].values  # Yeni verilerin olduğu sütunu seç

# Yeni verileri sayısallaştır
X_new_vec = vectorizer.transform(X_new)

# Sınıflandırma yap
y_pred = clf.predict(X_new_vec)

# Gerçek etiketleri al
y_true = df_new['Yorum'].values  # Gerçek duygu etiketleri (örnek olarak 'Gercek_Duygu' sütunu kullanıldı, adınıza göre değiştirin)


# Doğruluk oranını hesapla
accuracy = accuracy_score(y_true, y_pred)
print("Doğruluk Oranı:", accuracy)

# Tahmin sonuçlarını göster
for comment, pred in zip(X_new, y_pred):
    print(f"Yorum: {comment} \nTahmin: {pred}\n")