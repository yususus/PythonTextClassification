# Gerekli modülleri içe aktar
import pandas as pd
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Veri setini yükle
data = pd.read_csv('test_tweets.csv')

# Özellikler ve hedef değişkeni belirle
X = data["Turkcell'e kızgınım. Ve bu kızgınlık sanırım ayrılıkla sonlanıcak gibi geliyor bana.Farklı bir operatörün %30'u fazla fiyat teklif ediyorlar"]
y = data["olumsuz"]

# Veriyi eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Word2Vec modelini yükle
model = Word2Vec.load('model.bin')

# TF-IDF vektörleştiriciyi oluştur
vectorizer = TfidfVectorizer()

# Eğitim verilerini vektörleştir
X_train_tfidf = vectorizer.fit_transform(X_train)

# Her bir kelimenin TF-IDF ağırlığını hesapla
weights = np.asarray(X_train_tfidf.mean(axis=0)).ravel().tolist()
weights_df = pd.DataFrame({'term': vectorizer.get_feature_names(), 'weight': weights})

# Her bir yorumu Word2Vec vektörlerine dönüştür
X_train_vec = np.array([np.mean([model[w] * weights_df.loc[weights_df.term == w, 'weight'].values[0] for w in words if w in model] or [np.zeros(300)], axis=0) for words in X_train])

# Modeli oluştur ve eğit
clf = RandomForestClassifier()
clf.fit(X_train_vec, y_train)

# Test verilerini vektörleştir
X_test_tfidf = vectorizer.transform(X_test)

# Her bir yorumu Word2Vec vektörlerine dönüştür
X_test_vec = np.array([np.mean([model[w] * weights_df.loc[weights_df.term == w, 'weight'].values[0] for w in words if w in model] or [np.zeros(300)], axis=0) for words in X_test])

# Modelin doğruluğunu kontrol et
accuracy = clf.score(X_test_vec, y_test)
print(f"Model accuracy: {accuracy}")
