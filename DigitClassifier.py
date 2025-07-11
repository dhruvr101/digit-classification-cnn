import numpy as np
from PIL import Image
import os

X = []
y = []

for digit in range(10):
    folder = f"DigitsDataset/{digit}"
    for file in os.listdir(folder):
        if file.endswith(".jpg"):
            path = os.path.join(folder, file)
            img = Image.open(path).convert("L").resize((28, 28))
            img_array = np.array(img) / 255.0
            X.append(img_array)
            y.append(digit)

X = np.array(X)
y = np.array(y)
N = X.shape[0]

y_onehot = np.zeros((N, 10))
y_onehot[np.arange(N), y] = 1

np.random.seed(42)

num_filters = 16
filter_size = 3
image_size = 28
conv_output_size = image_size - filter_size + 1

#filters =  weights
filters = np.random.randn(num_filters, filter_size, filter_size) * 0.1
biases = np.zeros(num_filters)

image = X[0]
feature_maps = np.zeros((num_filters, conv_output_size, conv_output_size))

for f in range(num_filters):
    for i in range(conv_output_size):
        for j in range(conv_output_size):
            patch = image[i:i+filter_size, j:j+filter_size]
            feature_maps[f, i, j] = np.sum(patch * filters[f]) + biases[f]

def relu(x):
    return np.maximum(0, x)

def max_pooling(feature_maps, pool_size=2, stride=2):
    num_maps, h, w = feature_maps.shape
    pooled_h = (h - pool_size) // stride + 1
    pooled_w = (w - pool_size) // stride + 1
    pooled = np.zeros((num_maps, pooled_h, pooled_w))

    for f in range(num_maps):
        for i in range(0, h - pool_size + 1, stride):
            for j in range(0, w - pool_size + 1, stride):
                region = feature_maps[f, i:i+pool_size, j:j+pool_size]
                pooled[f, i//stride, j//stride] = np.max(region)
    return pooled

def softmax(z):
    exp_z = np.exp(z - np.max(z))  
    return exp_z / np.sum(exp_z)


feature_maps_relu = relu(feature_maps)
feature_maps_relu = max_pooling(feature_maps_relu)
#bias of filter layer two  = filterLayerTwo
filters = np.random.randn(32, 16, 3, 3) * 0.1
biasesLayerTwo = np.zeros(32)
feature_maps_LayerTwo = np.zeros((32, 22, 22))



# Input: feature_maps_relu → shape: (16, 13, 13)
input_maps = feature_maps_relu
num_input_maps, input_h, input_w = input_maps.shape


num_filters = 32
filter_size = 3
filters = np.random.randn(num_filters, num_input_maps, filter_size, filter_size) * 0.1
biasesLayerTwo = np.zeros(num_filters)

# Output shape after conv: 13 - 3 + 1 = 11
output_h = input_h - filter_size + 1
output_w = input_w - filter_size + 1
feature_maps_LayerTwo = np.zeros((num_filters, output_h, output_w))


for f in range(num_filters): 
    for i in range(output_h):
        for j in range(output_w):
    
            patch = input_maps[:, i:i+filter_size, j:j+filter_size]
            
            feature_maps_LayerTwo[f, i, j] = np.sum(patch * filters[f]) + biasesLayerTwo[f]



feature_maps_LayerTwo = relu(feature_maps_LayerTwo)
feature_maps_LayerTwo = max_pooling(feature_maps_LayerTwo)  # Output: (32, 5, 5)

feature_maps_LayerTwo.flatten()


flat = feature_maps_LayerTwo.flatten()
input_size = flat.shape[0]  # This is num_inputs to your MLP
hidden_size = 64
output_size = 10

# He Initialization for ReLU
W1 = np.random.randn(hidden_size, input_size) * np.sqrt(2. / input_size)
b1 = np.zeros((hidden_size, 1))

W2 = np.random.randn(output_size, hidden_size) * np.sqrt(2. / hidden_size)
b2 = np.zeros((output_size, 1))


x = flat.reshape(-1, 1)  # shape (input_size, 1)

# Hidden layer
Z1 = W1 @ x + b1
A1 = np.maximum(0, Z1)  # ReLU

# Output layer (logits)
Z2 = W2 @ A1 + b2


Z2 = W2 @ A1 + b2  


probs = softmax(Z2)  # shape: (10, 1)


predicted_digit = np.argmax(probs)

print("Predicted digit:", predicted_digit)
print("Probabilities:", probs.ravel()) 
