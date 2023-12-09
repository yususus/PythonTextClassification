import pandas as pd
from gensim.models import Word2Vec

# CSV dosyasını oku
df = pd.read_csv('tokenleştirilmis.csv')  # 'dosya_yolu.csv' yerine CSV dosyanızın yolunu yazın

# Her iki sütunu birleştir
sentences = df['Yorumlar'].tolist() + df['Duygular'].tolist()  # 'sütun1' ve 'sütun2' yerine tokenlerin bulunduğu sütunların isimlerini yazın

# Word2Vec modelini eğit
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# Bir kelimenin vektörünü al
vector = model.wv["olumsuz"]  # 'kelime' yerine vektörünü almak istediğiniz kelimeyi yazın
