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
