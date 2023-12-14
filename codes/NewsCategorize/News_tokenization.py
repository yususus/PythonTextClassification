import pandas as pd
from nltk.tokenize import word_tokenize

# CSV dosyasını oku
df = pd.read_csv('Haber2.csv')

# 'yorum' ve 'duygu' sütunlarının var olduğunu kontrol et
assert 'Haber' in df.columns, "'Haber' sütunu veri çerçevesinde bulunamadı."
assert 'Yorum' in df.columns, "'Yorum' sütunu veri çerçevesinde bulunamadı."

# 'yorum' ve 'duygu' sütunlarını tokenleştir
df['Haberler'] = df['Haber'].apply(word_tokenize)
df['İçerikler'] = df['Yorum'].apply(word_tokenize)

# 'yorum' ve 'duygu' sütunlarını çıkar
df = df.drop(['Haber', 'Yorum'], axis=1)

# NaN değerleri temizle
df = df.dropna()

# Tokenleştirilmiş verileri 'tokenleştirilmis2.csv' adlı ayrı bir CSV dosyasına kaydet
df.to_csv('Haber_token2.csv', index=False)

