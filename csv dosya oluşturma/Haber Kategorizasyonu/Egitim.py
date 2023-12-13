from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from gensim.models import Word2Vec

def convert_to_label(prediction):
    if prediction == 'egitim_Haber':
        return 'egitim'
    elif prediction == 'ekonomi_Haber':
        return 'ekonomi'
    elif prediction == 'magazin_Haber':
        return 'magazin'
    elif prediction == 'spor_Haber':
        return 'spor'
    elif prediction == 'dunya_Haber':
        return 'Dünya'
    elif prediction == 'kultur-sanat_Haber':
        return 'Kültür sanat'
    elif prediction == 'politika_Haber':
        return  'Politika'
    elif prediction == 'teknoloji_Haber':
        return 'teknoloji'
    else :
        return 'değerlendirilemedi'


word2vec_model = Word2Vec.load("modelHaber.bin")

# Load training data
df_train = pd.read_csv('Toplu9.csv', usecols=['Yorum', 'Haber'])

# Handle missing values and ensure 'Yorum' column is of type string
df_train = df_train.fillna('')
df_train['Haber'] = df_train['Haber'].astype(str)

X_train = df_train['Haber'].values
y_train = df_train['Yorum'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.9, random_state=42)

# Tokenize and convert text to word embeddings
X_train_vec = []
for sentence in X_train:
    word_embeddings = [word2vec_model.wv[word] for word in sentence.split() if word in word2vec_model.wv]
    if word_embeddings:
        sentence_embedding = np.mean(word_embeddings, axis=0)
        X_train_vec.append(sentence_embedding)
    else:
        # Handle the case where all words in the sentence are not in the Word2Vec model
        X_train_vec.append(np.zeros(word2vec_model.vector_size))

X_train_vec = np.vstack(X_train_vec)

# Train a classifier (using Support Vector Machine as an example)
clf = SVC()
clf.fit(X_train_vec, y_train)

# Tokenize and convert new text to word embeddings
X_test_vec = []
for sentence in X_test:
    word_embeddings = [word2vec_model.wv[word] for word in sentence.split() if word in word2vec_model.wv]
    if word_embeddings:
        sentence_embedding = np.mean(word_embeddings, axis=0)
        X_test_vec.append(sentence_embedding)
    else:
        # Handle the case where all words in the sentence are not in the Word2Vec model
        X_test_vec.append(np.zeros(word2vec_model.vector_size))

X_test_vec = np.vstack(X_test_vec)

# Make predictions
y_pred = clf.predict(X_test_vec)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Doğruluk Oranı:", accuracy)

# Print the comments and their corresponding predictions
for comment, prediction in zip(X_test, y_pred):
    label = convert_to_label(prediction)
    print(f"Haber: {comment} \n Tahmin: {label}\n")
