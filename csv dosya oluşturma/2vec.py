# Gerekli kütüphaneleri içe aktarma
import pandas as pd
from gensim.models import Word2Vec

# CSV dosyasını okuma
df = pd.read_csv('tokenleştirilmis2.csv')

# Metin verilerini temizleme ve önişleme (bu adımı verilerinize göre özelleştirebilirsiniz)

# Word2Vec modelini eğitme
sentences = df['Yorumlar'].tolist()
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

model.save("model.bin")
