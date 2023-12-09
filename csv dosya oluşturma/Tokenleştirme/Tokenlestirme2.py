import pandas as pd
from nltk.tokenize import word_tokenize

# CSV dosyasını oku
df = pd.read_csv('veri_seti_yeni_adlar.csv')

# 'yorum (2)' sütununu tokenleştir
df['Yorumlar'] = df['yorum'].apply(word_tokenize)
df['Duygular'] = df['duygu'].apply(word_tokenize)

print(df['Yorumlar', 'Duygular'])


df.to_csv('tokenleştirilmis2.csv', index=False)
