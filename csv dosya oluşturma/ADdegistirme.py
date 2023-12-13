import pandas as pd

veri_seti = pd.read_csv('train_tweets.csv')

yeni_sütun_adları = {
"Ulan Wifi'ye bağlıyım ben. Ona bağlıyken Turkcell internet paketin bitti diye nasıl mesaj atabilir bana ya? Onu da mı ödeyelim": "yorum",
"olumsuz": "duygu",
}

veri_seti = veri_seti.rename(columns=yeni_sütun_adları)

veri_seti.to_csv('train_yeni.csv')

#internetten aldığım veri setininin sütun adlarını değiştirdim