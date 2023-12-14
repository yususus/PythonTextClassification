import pandas as pd
from gensim.models import Word2Vec

df = pd.read_csv('tokenleştirilmis3.csv')

# 'Yorumlar' sütunu var mı diye kontrol ediyoruz
assert 'Yorumlar' in df.columns, "'Yorumlar' sütunu veri çerçevesinde bulunamadı."

# NaN değerleri temizle
df = df.dropna(subset=['Yorumlar'])

# model eğitme işlemi
sentences = df['Yorumlar'].tolist()
model = Word2Vec(sentences, vector_size=100, window=5, min_count=5, workers=4)

model.save("model2.bin")
