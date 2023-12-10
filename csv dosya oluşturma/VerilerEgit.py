from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd

df = pd.read_csv('tokenleştirilmis2.csv')
# Tokenleştirilmiş ve sayısallaştırılmış veri setinizi yükle
X_train = df['yorum'].values  # Eğitim verileri
y_train = df['duygu'].values  # Eğitim etiketleri

# Metin sınıflandırma modelini oluştur
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
clf = MultinomialNB()
clf.fit(X_train_vec, y_train)

# Yeni verileri yorumla
X_new = ...  # Yeni veriler
X_new_vec = vectorizer.transform(X_new)
y_pred = clf.predict(X_new_vec)
