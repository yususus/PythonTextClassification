import pandas as pd

veri_seti = pd.read_csv('test_tweets.csv')

yeni_sütun_adları = {
"Turkcell'e kızgınım. Ve bu kızgınlık sanırım ayrılıkla sonlanıcak gibi geliyor bana.Farklı bir operatörün %30'u fazla fiyat teklif ediyorlar": "yorum",
"olumsuz": "duygu",
}

veri_seti = veri_seti.rename(columns=yeni_sütun_adları)

veri_seti.to_csv('veri_seti_yeni_sütun_adları.csv')

#internetten aldığım veri setininin sütun adlarını değiştirdim