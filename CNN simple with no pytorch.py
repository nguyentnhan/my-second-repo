import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Cấu hình
DATA_DIR = '/kaggle/input/dogs-vs-cats/dataset/train'
TEST_DIR = '/kaggle/input/dogs-vs-cats/dataset/test'
IMG_SIZE = (64, 64)  # Resize nhỏ để huấn luyện nhanh

# Load ảnh
def load_images(data_dir, max_images=500):
    X = []
    y = []

    for label_name, label_value in [('dogs', 1), ('cats', 0)]:
        subdir = os.path.join(data_dir, label_name)
        count = 0

        for fname in os.listdir(subdir):
            if not fname.lower().endswith('.jpg') or count >= max_images:
                continue
            img_path = os.path.join(subdir, fname)
            img = Image.open(img_path).resize(IMG_SIZE).convert('L')  # grayscale
            img_array = np.array(img) / 255.0
            X.append(img_array)
            y.append(label_value)
            count += 1
            
    return np.array(X), np.array(y)
X, y = load_images(DATA_DIR, max_images=1000)
print("Dataset shape:", X.shape, y.shape)  # (1000, 64, 64), (1000,)
# Thêm channel dimension (1)
X = X.reshape(-1, 1, 64, 64)  # (batch, channel, height, width)

# Chia train/test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Khởi tạo tham số
np.random.seed(42)
kernel = np.random.randn(3, 3) * 0.1  # kernel 3x3
W = np.random.randn(62*62, 1) * 0.1   # fully connected (flatten từ ảnh 62x62)

def conv2d(X, kernel):
    kh, kw = kernel.shape
    h, w = X.shape
    output_h, output_w = h - kh + 1, w - kw + 1
    result = np.zeros((output_h, output_w))
    for i in range(output_h):
        for j in range(output_w):
            result[i, j] = np.sum(X[i:i+kh, j:j+kw] * kernel)
    return result
def relu(X):
    return np.maximum(0, X)

def flatten(X):
    return X.reshape(-1)

def forward(X,label):
    # Convolution
    conv = conv2d(X[0], kernel)
    act = relu(conv)
    flat = flatten(act).reshape(1, -1)  # (1, features)
    out = flat @ W  # (1, 1) , W là một vector cột
    out=np.clip(out,-500,500)
    prob = 1 / (1 + np.exp(-out))  # sigmoid
    loss = - (label * np.log(prob) + (1 - label) * np.log(1 - prob))
    return prob, loss, flat, act, conv
def conv2d_kernel_grad(input, grad_output_grad):
    kh, kw = kernel.shape
    grad = np.zeros((kh, kw))
    for i in range(grad_output_grad.shape[0]):
        for j in range(grad_output_grad.shape[1]):
            grad += input[i:i+kh, j:j+kw] * grad_output_grad[i, j]
    return grad
def relu_derivative(x):
    return (x > 0).astype(float)
lr = 0.001
losses = []

for epoch in range(10):
    total_loss = 0
    for i in range(len(X_train)):
        x = X_train[i]
        label = y_train[i]

        # Forward
        prob, loss, out, flat, act, conv = forward(x,label)
        total_loss += loss

        # Backprop
        dL_dout = prob - label  # derivative of loss w.r.t. output
        dL_dflat = dL_dout @ W.T  # sigmoid derivative
        dL_dW = flat.reshape(-1,1) @ dL_dout.reshape(1, 1) # (features, 1)
        dL_dact = dL_dflat.reshape(62, 62)   # reshape lại thành 2D
        dL_dconv = relu_derivative(conv) * dL_dact  # (62, 62)
        dL_dkernel = conv2d_kernel_grad(x[0], dL_dconv)  # (3, 3)
        # Update weights
        W -= lr * dL_dW
        kernel -= lr * dL_dkernel

    avg_loss = total_loss / len(X_train)
    print(f"Epoch {epoch+1} - Loss: {avg_loss.item():.4f}")
    losses.append(avg_loss.item())
# Vẽ đồ thị loss
def predict(X,label):
    prob, _, _, _, _ = forward(X,label)
    return 1 if prob > 0.5 else 0

correct = 0
for i in range(len(X_test)):
    pred = predict(X_test[i],y_test[i])
    if pred == y_test[i]:
        correct += 1

acc = correct / len(X_test)
print(f"Test Accuracy: {acc:.2f}")
plt.plot(losses)
plt.title("Loss over epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.show()
for i in range(len(X_test)):
    pred = predict(X_test[i],y_test[i])
    true_label = y_test[i]
    
    # Hiển thị ảnh
    img = X_test[i].reshape(64, 64)  # X_test có shape (1,64,64), bỏ kênh đi
    plt.imshow(img, cmap='gray')
    plt.title(f"True: {'Dog' if true_label==1 else 'Cat'} - Pred: {'Dog' if pred==1 else 'Cat'}")
    plt.axis('off')
    plt.show()
    
    # Nếu bạn chỉ muốn hiển thị một số ảnh đầu tiên, có thể thêm điều kiện dừng
    # if i == 9:
    #     break