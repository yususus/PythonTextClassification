import pandas as pd
from nltk.tokenize import word_tokenize

# hangi dosyası tokenleştirmek istiyorsak onu "Toplu.csv" yazan yere yazıyoruz
df = pd.read_csv('Toplu.csv')

# 'yorum' ve 'duygu' sütunları var mı diye bakıyoruz
assert 'Haber' in df.columns, "'Haber' sütunu bulunmadı"
assert 'Yorum' in df.columns, "'Yorum' sütunu bulunmadı"

# 'yorum' ve 'duygu' sütunlarını tokenleştir
df['Haber'] = df['Haber'].apply(word_tokenize)
df['Yorum'] = df['Yorum'].apply(word_tokenize)

# 'yorum' ve 'duygu' sütunlarını çıkar
df = df.drop(['Haber', 'Yorum'], axis=1)

# NaN değerleri temizle
df = df.dropna()

df.to_csv('Toplu_token.csv', index=False)

