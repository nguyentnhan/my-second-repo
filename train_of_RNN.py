import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import nltk
import utils.prepare as pre
import numpy as np
from RNN import RNN

# Giả sử bạn đã load dữ liệu vào DataFrame df
df = pd.read_csv('Data/rawdata.csv')
path = 'pretrained/glove.6B.50d.txt'
# nltk.download('punkt')
# nltk.download('punkt_tab')

df1 = pd.read_csv(path, sep=' ',index_col=0, header = None, quoting = 3)

le = LabelEncoder()
df['Label'] = le.fit_transform(df['Author']) # Human -> 0, AI -> 1
X_train, X_test, y_train, y_test = train_test_split(df.head(40)['Text'], df.head(40)['Label'], test_size=0.2)
y_train_np = y_train.values
y_test_np = y_test.values

maxlength = 100
embed_dim = 50
all_matrices = []
for batch in X_train:
    tokens = pre.preprocess_for_glove(batch)
    matrix = pre.text_to_matrix(tokens, df1, maxlength, embed_dim)
    all_matrices.append(matrix)

X_train_tensor = np.stack(all_matrices)
model = RNN(maxlength)

batch_size = 8
n_samples = len(X_train)
for _ in range(1000):
    for i in range(0, n_samples, batch_size):
        batch = X_train_tensor[i : i + batch_size].transpose(1, 0, 2)
        label = y_train_np[i : i + batch_size].reshape(-1, 1)
        model.train(batch, label,1, 0.001)
        print(f"Đang xét batch từ chỉ số {i} đến {i + batch_size}")
y, _, _ = model.forward(X_train_tensor[4:8].transpose(1, 0, 2), y_train_np[4:8].reshape(-1,1))

# 2. Chuyển về dạng nhãn 0 hoặc 1 (ngưỡng 0.5)
predictions = (y > 0.5).astype(int).flatten()
actuals = y_train_np[4:8]

# 3. In ra so sánh
print("Dự đoán (Xác suất):\n", y.flatten())
print("Dự đoán (Nhãn):     ", predictions)
print("Thực tế:            ", actuals)