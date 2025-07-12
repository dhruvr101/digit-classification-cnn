import numpy as np
from PIL import Image
import os
from sklearn.model_selection import train_test_split

# Load dataset
X, y = [], []
for digit in range(10):
    folder = f"DigitsDataset/{digit}"
    for file in os.listdir(folder):
        if file.endswith(".jpg"):
            path = os.path.join(folder, file)
            img = Image.open(path).convert("L").resize((28, 28))
            X.append(np.array(img) / 255.0)
            y.append(digit)

X = np.array(X)
y = np.array(y)
y_onehot = np.zeros((len(y), 10))
y_onehot[np.arange(len(y)), y] = 1

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)
# Speed up training by limiting dataset size
X_train = X_train[:300]
y_train = y_train[:300]
X_test = X_test[:100]
y_test = y_test[:100]


# --- Helper Functions ---
def relu(x): return np.maximum(0, x)
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

def max_pooling(feature_maps, pool_size=2, stride=2):
    num_maps, h, w = feature_maps.shape
    pooled = np.zeros((num_maps, (h - pool_size)//stride + 1, (w - pool_size)//stride + 1))
    for f in range(num_maps):
        for i in range(0, h - pool_size + 1, stride):
            for j in range(0, w - pool_size + 1, stride):
                pooled[f, i//stride, j//stride] = np.max(feature_maps[f, i:i+pool_size, j:j+pool_size])
    return pooled
def max_pooling_backward(dout, feature_maps, pool_size=2, stride=2):
    num_filters, h, w = feature_maps.shape
    dx = np.zeros_like(feature_maps)
    for f in range(num_filters):
        for i in range(0, h - pool_size + 1, stride):
            for j in range(0, w - pool_size + 1, stride):
                region = feature_maps[f, i:i+pool_size, j:j+pool_size]
                max_idx = np.unravel_index(np.argmax(region), region.shape)
                dx[f, i + max_idx[0], j + max_idx[1]] += dout[f, i//stride, j//stride]
    return dx

# --- Initialize parameters ---
np.random.seed(42)
filters1 = np.random.randn(16, 3, 3) * 0.1
biases1 = np.zeros(16)
filters2 = np.random.randn(32, 16, 3, 3) * 0.1
biases2 = np.zeros(32)
W1 = np.random.randn(64, 32 * 5 * 5) * np.sqrt(2. / (32 * 5 * 5))
b1 = np.zeros((64, 1))
W2 = np.random.randn(10, 64) * np.sqrt(2. / 64)
b2 = np.zeros((10, 1))
lr = 0.01
epochs = 2

# --- Training loop ---
for epoch in range(epochs):
    loss_epoch = 0
    for idx in range(len(X_train)):
        if idx % 50 == 0:
            print(f"Epoch {epoch+1}, training image {idx}/{len(X_train)}")
        image = X_train[idx]
        label = y_train[idx].reshape(-1, 1)

        # Forward pass
        conv1 = np.array([
            [[np.sum(image[i:i+3, j:j+3] * filters1[f]) + biases1[f]
              for j in range(26)] for i in range(26)]
            for f in range(16)
        ])
        relu1 = relu(conv1)
        pool1 = max_pooling(relu1)

        conv2 = np.array([
            [[np.sum(pool1[:, i:i+3, j:j+3] * filters2[f]) + biases2[f]
              for j in range(11)] for i in range(11)]
            for f in range(32)
        ])
        relu2 = relu(conv2)
        pool2 = max_pooling(relu2)

        flat = pool2.flatten().reshape(-1, 1)
        Z1 = W1 @ flat + b1
        A1 = relu(Z1)
        Z2 = W2 @ A1 + b2
        probs = softmax(Z2)
        loss = -np.sum(label * np.log(probs + 1e-8))
        loss_epoch += loss

        # Backward pass
        dZ2 = probs - label
        dW2 = dZ2 @ A1.T
        db2 = dZ2
        dA1 = W2.T @ dZ2
        dZ1 = dA1 * (Z1 > 0)
        dW1 = dZ1 @ flat.T
        db1 = dZ1
        dflat = W1.T @ dZ1
        dpool2 = dflat.reshape(pool2.shape)
        drelu2 = max_pooling_backward(dpool2, relu2)
        dconv2 = drelu2 * (conv2 > 0)

        dpool1 = np.zeros_like(pool1)
        dfilters2 = np.zeros_like(filters2)
        dbiases2 = np.zeros_like(biases2)
        for f in range(32):
            for i in range(11):
                for j in range(11):
                    patch = pool1[:, i:i+3, j:j+3]
                    dval = dconv2[f, i, j]
                    dfilters2[f] += dval * patch
                    dbiases2[f] += dval
                    dpool1[:, i:i+3, j:j+3] += dval * filters2[f]

        drelu1 = max_pooling_backward(dpool1, relu1)
        dconv1 = drelu1 * (conv1 > 0)
        dfilters1 = np.zeros_like(filters1)
        dbiases1 = np.zeros_like(biases1)
        for f in range(16):
            for i in range(26):
                for j in range(26):
                    patch = image[i:i+3, j:j+3]
                    dval = dconv1[f, i, j]
                    dfilters1[f] += dval * patch
                    dbiases1[f] += dval

        # Update weights
        W2 -= lr * dW2
        b2 -= lr * db2
        W1 -= lr * dW1
        b1 -= lr * db1
        filters2 -= lr * dfilters2
        biases2 -= lr * dbiases2
        filters1 -= lr * dfilters1
        biases1 -= lr * dbiases1

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss_epoch:.4f}")



# --- Evaluate on test set ---
correct = 0
for idx in range(len(X_test)):
    image = X_test[idx]

    conv1 = np.array([
        [[np.sum(image[i:i+3, j:j+3] * filters1[f]) + biases1[f]
          for j in range(26)] for i in range(26)]
        for f in range(16)
    ])
    relu1 = relu(conv1)
    pool1 = max_pooling(relu1)

    conv2 = np.array([
        [[np.sum(pool1[:, i:i+3, j:j+3] * filters2[f]) + biases2[f]
          for j in range(11)] for i in range(11)]
        for f in range(32)
    ])
    relu2 = relu(conv2)
    pool2 = max_pooling(relu2)

    flat = pool2.flatten().reshape(-1, 1)
    Z1 = W1 @ flat + b1
    A1 = relu(Z1)
    Z2 = W2 @ A1 + b2
    probs = softmax(Z2)
    predicted = np.argmax(probs)
    actual = np.argmax(y_test[idx])
    if predicted == actual:
        correct += 1

print(f"Test Accuracy: {correct}/{len(X_test)} = {correct / len(X_test):.2%}")
