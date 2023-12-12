from gensim.models import Word2Vec
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB

# Load the Word2Vec model
word2vec_model = Word2Vec.load("model2.bin")

# Load training data
df_train = pd.read_csv('tokenleştirilmis3.csv')
X_train = df_train['Yorumlar'].values
y_train = df_train['Duygular'].values

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

# Sınıflandırıcı modeli eğit
clf = MultinomialNB()
clf.fit(X_train_vec, y_train)

# Load new data
df_new = pd.read_csv('Toplu.csv')
df_new = df_new.dropna(subset=['Yorum'])
X_new = df_new['Yorum'].values

# Tokenize and convert new text to word embeddings
X_new_vec = []
for sentence in X_new:
    word_embeddings = [word2vec_model.wv[word] for word in sentence.split() if word in word2vec_model.wv]
    if word_embeddings:
        sentence_embedding = np.mean(word_embeddings, axis=0)
        X_new_vec.append(sentence_embedding)
    else:
        # Handle the case where all words in the sentence are not in the Word2Vec model
        X_new_vec.append(np.zeros(word2vec_model.vector_size))

X_new_vec = np.vstack(X_new_vec)

# Make predictions
y_pred = clf.predict(X_new_vec)

# Get true labels
y_true = df_new['Yorum'].values

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)
print("Doğruluk Oranı:", accuracy)

# Display prediction results
for comment, pred in zip(X_new, y_pred):
    print(f"Yorum: {comment} \nTahmin: {pred}\n")
