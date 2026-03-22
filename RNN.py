import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
class RNN:
    # Biến lớp (Class variables) - Nếu bạn muốn tất cả các RNN dùng chung W
    # Thường thì nên để trong __init__ để mỗi thực thể có bộ trọng số riêng
    
    def __init__(self, maxlength: int):
        # 1. Lưu các tham số cơ bản
        self.maxlength = maxlength
        
        # 2. Xử lý Tensor: (n, 1000, 50) -> (1000, n, 50)
        
        # 3. Khởi tạo các biến trạng thái (Hidden state)
        
        # 4. Khởi tạo Trọng số (Weights) và Bias (b)
        # Lưu ý: Phải có 'self.' để các hàm khác trong class có thể dùng được
        self.b_h = np.random.randn(1,50).astype(np.float32)*0.1
        self.w_h = np.random.randn(50, 50).astype(np.float32)*0.1
        self.w_x = np.random.randn(50, 50).astype(np.float32)*0.1
        self.w_t = np.random.randn(50, 1).astype(np.float32)*0.1
        self.b_t = np.random.randn(1,1).astype(np.float32)*0.1
    def forward(self,test_tensor: np.ndarray ,label: np.ndarray):
        num = test_tensor.shape[1]
        h_lists=[]
        self.h = np.zeros((num, 50), dtype=np.float32)
        for i in range(self.maxlength):
            self.h = test_tensor[i] @ self.w_x + self.h @ self.w_h + self.b_h
            self.h = np.tanh(self.h)
            h_lists.append(self.h)
        y_raw = self.h @ self.w_t + self.b_t
        y = sigmoid(y_raw)
        epsilon = 1e-12
        loss = -np.mean(label * np.log(y + epsilon) + (1 - label) * np.log(1 - y + epsilon))
        return y, loss, h_lists

    def train(self, train_tensor: np.ndarray, label: np.ndarray, epoch=1000, lr=0.001):
        n = train_tensor.shape[1]
        for e in range(epoch):
            # 1. Forward Pass
            y_pred, loss, h_lists = self.forward(train_tensor, label)
            
            # 2. Gradient Layer Output
            dL_dyraw = (y_pred - label) / n
            dL_dwt = h_lists[-1].T @ dL_dyraw 
            dL_dbt = np.sum(dL_dyraw, axis=0, keepdims=True)
            
            # Đạo hàm truyền vào Hidden State cuối cùng
            dL_dh = dL_dyraw @ self.w_t.T
            
            # KHỞI TẠO GRADIENT TÍCH LŨY (Dùng np.zeros_like để khớp shape ma trận)
            grad_wx = np.zeros_like(self.w_x)
            grad_wh = np.zeros_like(self.w_h)
            grad_bh = np.zeros_like(self.b_h)
            
            # 3. Backpropagation Through Time (BPTT)
            for i in range(self.maxlength - 1, -1, -1):
                h = h_lists[i]
                # Đạo hàm qua sigmoid tại bước i
                dtanh = dL_dh * (1 - h**2)
                
                # Cộng dồn gradient
                grad_wx += train_tensor[i].T @ dtanh
                grad_bh += np.sum(dtanh, axis=0, keepdims=True)
                
                if i > 0:
                    k = h_lists[i-1]
                    grad_wh += k.T @ dtanh
                    # Truyền dL_dh về bước trước
                    dL_dh = dtanh @ self.w_h.T
                else:
                    # i = 0, không có h_{-1}
                    pass
            for grad in [dL_dwt, dL_dbt, grad_wh, grad_bh, grad_wx]:
                np.clip(grad, -5, 5, out=grad)
            # 4. Cập nhật tất cả trọng số sau khi tích lũy xong 1 vòng BPTT
            self.w_t -= lr * dL_dwt
            self.b_t -= lr * dL_dbt
            self.w_h -= lr * grad_wh
            self.b_h -= lr * grad_bh
            self.w_x -= lr * grad_wx
            
            if e % 100 == 0:
                print(f"Epoch {e} | Loss: {loss:.6f}")
            


        

    