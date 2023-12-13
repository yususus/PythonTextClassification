import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from gensim.models import Word2Vec
from sklearn.svm import SVC
import numpy as np

def convert_to_label(prediction):
    if prediction == 1:
        return 'pozitif'
    elif prediction == 0:
        return 'nötr'
    else:
        return 'negatif'

# Load the pre-trained Word2Vec model
word2vec_model = Word2Vec.load("model2.bin")

# Load training data
df_train = pd.read_csv('Toplu9.csv', usecols=['Yorum', 'Haber'])

# Handle missing values and ensure 'Yorum' column is of type string
df_train = df_train.fillna('')
df_train['Yorum'] = df_train['Yorum'].astype(str)

X_train = df_train['Yorum'].values
y_train = df_train['Haber'].values

# Load testing data
df_test = pd.read_csv('test_token.csv', usecols=['Duygular', 'Yorumlar'])

# Handle missing values and ensure 'Yorum' column is of type string
df_test = df_test.fillna('')
df_test['Duygular'] = df_test['Duygular'].astype(str)

X_test = df_test['Duygular'].values
y_test = df_test['Yorumlar'].values

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
    print(f"Yorum: {comment} \nTahmin: {label}\n")
