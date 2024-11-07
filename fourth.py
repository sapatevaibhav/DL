import numpy as np
from sklearn.preprocessing import OneHotEncoder

# Sample corpus and parameters
corpus = ["I like machine learning", "I enjoy deep", "Coding is fascinating"]
window_size, embedding_dim, lr, epochs = 2, 5, 0.01, 100

# Preprocessing: Tokenize and create vocabulary
words = [word for sentence in corpus for word in sentence.lower().split()]
vocab = sorted(set(words))
word_to_index = {word: idx for idx, word in enumerate(vocab)}
index_to_word = {idx: word for word, idx in word_to_index.items()}

# One-hot encoding
encoder = OneHotEncoder(sparse=False)
one_hot_matrix = encoder.fit_transform(np.array(vocab).reshape(-1, 1))

# Generate training data
def generate_training_data(words, window_size):
    X, y = [], []
    for i in range(window_size, len(words) - window_size):
        context = words[i - window_size:i] + words[i + 1:i + window_size + 1]
        target = words[i]
        X.append(sum(one_hot_matrix[word_to_index[w]] for w in context))
        y.append(one_hot_matrix[word_to_index[target]])
    return np.array(X), np.array(y)

X_train, y_train = generate_training_data(words, window_size)

# Model parameters and layers
W1, W2 = np.random.rand(len(vocab), embedding_dim), np.random.rand(embedding_dim, len(vocab))

# Training the CBOW model
for _ in range(epochs):
    for x, target in zip(X_train, y_train):
        h, u = np.dot(x, W1), np.dot(np.dot(x, W1), W2)
        y_pred = np.exp(u) / np.sum(np.exp(u))
        e = y_pred - target
        W2 -= lr * np.outer(h, e)
        W1 -= lr * np.outer(x, np.dot(W2, e))

# Word prediction function
def predict(context_words):
    context_vec = sum(one_hot_matrix[word_to_index[word]] for word in context_words)
    h = np.dot(context_vec, W1)
    u = np.dot(h, W2)
    y_pred = np.exp(u) / np.sum(np.exp(u))
    return index_to_word[np.argmax(y_pred)]

# Test prediction
print("Predicted word:", predict(["i","like","learning","i"]))
