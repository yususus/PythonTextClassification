# Gerekli kütüphaneleri içe aktarma
import pandas as pd
from gensim.models import Word2Vec

# CSV dosyasını okuma
df = pd.read_csv('Haber_token2.csv')

# 'Yorumlar' sütununun var olduğunu kontrol et
assert 'İçerikler' in df.columns, "'Yorumlar' sütunu veri çerçevesinde bulunamadı."

# NaN değerleri temizle
df = df.dropna(subset=['İçerikler'])

# Metin verilerini temizleme ve önişleme (bu adımı verilerinize göre özelleştirebilirsiniz)

# Word2Vec modelini eğitme
# min_count parametresi genellikle 1'den büyük bir değer olarak ayarlanmalıdır[^1^][3].
sentences = df['İçerikler'].tolist()
model = Word2Vec(sentences, vector_size=100, window=5, min_count=5, workers=4)

# Modeli kaydetme
# Eğitimi daha sonra sürdürmek istiyorsanız, tam modeli kaydetmeniz gerekir[^2^][5].
model.save("modelHaber.bin")
