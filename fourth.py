import numpy as np

# Prepare data with context weighting
def prepare_data(corpus, window_size):
    words = ' '.join(corpus).split()
    vocab = list(set(words))
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}

    X, y = [], []
    for i in range(window_size, len(words) - window_size):
        context = [word_to_idx[words[i+j]] for j in range(-window_size, window_size+1) if j != 0]
        target = word_to_idx[words[i]]
        X.append(context)
        y.append(target)
    return np.array(X), np.array(y), word_to_idx, idx_to_word

# CBOW Model with weighted context
class CBOW:
    def __init__(self, vocab_size, embed_dim):
        self.W1 = np.random.randn(vocab_size, embed_dim)
        self.W2 = np.random.randn(embed_dim, vocab_size)

    def forward(self, X, weights):
        # Compute the weighted average embedding for context words
        context_embeddings = self.W1[X] * weights[:, np.newaxis]
        hidden = np.sum(context_embeddings, axis=0) / np.sum(weights)  # Weighted average
        out = hidden @ self.W2
        softmax = np.exp(out) / np.sum(np.exp(out))
        return softmax, hidden

    def train(self, X, y, lr=0.01, epochs=5000):
        for _ in range(epochs):
            for i in range(len(y)):
                # Use proximity-based weights for context words
                context_len = len(X[i])
                weights = np.array([1 / abs(j) if j != 0 else 0 for j in range(-context_len//2, context_len//2+1) if j != 0])

                softmax, hidden = self.forward(X[i], weights)
                softmax[y[i]] -= 1  # Gradient of loss

                # Update weights
                self.W2 -= lr * np.outer(hidden, softmax)
                for idx, weight in zip(X[i], weights):  # Weighted update for each context word
                    self.W1[idx] -= lr * weight * softmax @ self.W2.T / len(X[i])

    def predict(self, context, word_to_idx, idx_to_word):
        idx = [word_to_idx[word] for word in context]
        weights = np.array([1 / abs(j) if j != 0 else 0 for j in range(-len(idx)//2, len(idx)//2+1) if j != 0])
        softmax, _ = self.forward(idx, weights)
        return idx_to_word[np.argmax(softmax)]

# Usage
corpus = ["open source is amazing", "I love open source projects"]
window_size = 2
X, y, word_to_idx, idx_to_word = prepare_data(corpus, window_size)
model = CBOW(len(word_to_idx), embed_dim=50)
model.train(X, y)

# Prediction test
print("Predicted word:", model.predict(["open", "source", "amazing", "I"], word_to_idx, idx_to_word))
