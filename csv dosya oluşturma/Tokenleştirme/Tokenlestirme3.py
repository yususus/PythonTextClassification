import pandas as pd
from nltk.tokenize import word_tokenize

# CSV dosyasını oku
df = pd.read_csv('veri_seti_yeni_adlar.csv')

# 'yorum' ve 'duygu' sütunlarının var olduğunu kontrol et
assert 'yorum' in df.columns, "'yorum' sütunu veri çerçevesinde bulunamadı."
assert 'duygu' in df.columns, "'duygu' sütunu veri çerçevesinde bulunamadı."

# 'yorum' ve 'duygu' sütunlarını tokenleştir
df['Yorumlar'] = df['yorum'].apply(word_tokenize)
df['Duygular'] = df['duygu'].apply(word_tokenize)

# 'yorum' ve 'duygu' sütunlarını çıkar
df = df.drop(['yorum', 'duygu'], axis=1)

# NaN değerleri temizle
df = df.dropna()

print(df[['Yorumlar', 'Duygular']])

# Tokenleştirilmiş verileri 'tokenleştirilmis2.csv' adlı ayrı bir CSV dosyasına kaydet
df.to_csv('tokenleştirilmis3.csv', index=False)
