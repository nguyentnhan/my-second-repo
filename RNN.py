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
        # forget gate w
        self.w_f = np.random.randn(100, 50).astype(np.float32)*0.1
        self.b_f = np.random.randn(1, 50).astype(np.float32)*0.1
        # input gate w
        self.w_i = np.random.randn(100, 50).astype(np.float32)*0.1
        self.b_i = np.random.randn(1, 50).astype(np.float32)*0.1
        self.w_c = np.random.randn(100, 50).astype(np.float32)*0.1
        self.b_c = np.random.randn(1, 50).astype(np.float32)*0.1
        # output gate w
        self.w_o = np.random.randn(100, 50).astype(np.float32)*0.1
        self.b_o = np.random.randn(1, 50).astype(np.float32)*0.1

        # w đầu ra cuối
        self.w_t = np.random.randn(50, 1).astype(np.float32)*0.1
        self.b_t = np.random.randn(1,1).astype(np.float32)*0.1
    def forward(self,test_tensor: np.ndarray ,label: np.ndarray):
        num = test_tensor.shape[1]
        h_lists=[]
        f_lists=[]
        o_lists=[]
        c_lists=[]
        ct_lists=[]
        i_lists=[]
        p_lists=[]
        self.h = np.zeros((num, 50), dtype=np.float32)
        ct = np.zeros((num, 50), dtype=np.float32)
        for t in range(self.maxlength):
            p = np.concatenate((self.h,test_tensor[t]),axis=1)
            f = sigmoid(p @ self.w_f + self.b_f)
            i = sigmoid(p @ self.w_i + self.b_i)
            o = sigmoid(p @ self.w_o + self.b_o)
            c = np.tanh(p @ self.w_c + self.b_c)
            ct = f * ct + i * c
            self.h = o * np.tanh(ct)
            
            f_lists.append(f)
            o_lists.append(o)
            c_lists.append(c)
            ct_lists.append(ct)
            i_lists.append(i)
            h_lists.append(self.h)
            p_lists.append(p)
        y_raw = self.h @ self.w_t + self.b_t
        y = sigmoid(y_raw)
        epsilon = 1e-12
        loss = -np.mean(label * np.log(y + epsilon) + (1 - label) * np.log(1 - y + epsilon))
        return y, loss, h_lists, f_lists, o_lists, c_lists, ct_lists,i_lists, p_lists

    def train(self, train_tensor: np.ndarray, label: np.ndarray, epoch=1000, lr=0.001):
        n = train_tensor.shape[1]
        for e in range(epoch):
            # 1. Forward Pass
            y_pred, loss, h_lists, f_lists, o_lists, c_lists, ct_lists, i_lists, p_lists = self.forward(train_tensor, label)
            
            # 2. Gradient Layer Output (Lớp Dense cuối cùng)
            dL_dyraw = (y_pred - label) / n
            dL_dwt = h_lists[-1].T @ dL_dyraw 
            dL_dbt = np.sum(dL_dyraw, axis=0, keepdims=True)
            
            # Khởi tạo gradient cho các cổng LSTM
            grad_wf, grad_bf = np.zeros_like(self.w_f), np.zeros_like(self.b_f)
            grad_wi, grad_bi = np.zeros_like(self.w_i), np.zeros_like(self.b_i)
            grad_wc, grad_bc = np.zeros_like(self.w_c), np.zeros_like(self.b_c)
            grad_wo, grad_bo = np.zeros_like(self.w_o), np.zeros_like(self.b_o)

            # Lỗi truyền ngược từ bước sau (t+1)
            dh_next = dL_dyraw @ self.w_t.T
            dc_next = np.zeros((n, 50))
            
            # 3. Backpropagation Through Time (BPTT)
            for t in range(self.maxlength - 1, -1, -1):
                # Lấy dữ liệu đã lưu tại bước t
                p = p_lists[t]
                f = f_lists[t]
                i = i_lists[t]
                o = o_lists[t]
                c_tilde = c_lists[t]
                ct = ct_lists[t]
                ct_prev = ct_lists[t-1] if t > 0 else np.zeros((n, 50))

                # --- Đạo hàm qua Cell State (Ct) ---
                # Lỗi tại Ct = (Lỗi từ Ht truyền vào) + (Lỗi từ Ct+1 truyền về qua Forget gate)
                dc = dh_next * o * (1 - np.tanh(ct)**2) + dc_next
                
                # --- Đạo hàm các cổng (Gates) ---
                # Đạo hàm qua cổng Output (o)
                do_raw = dh_next * np.tanh(ct) * o * (1 - o)
                
                # Đạo hàm qua cổng Forget (f)
                df_raw = dc * ct_prev * f * (1 - f)
                
                # Đạo hàm qua cổng Input (i)
                di_raw = dc * c_tilde * i * (1 - i)
                
                # Đạo hàm qua Candidate (c_tilde)
                dc_tilde_raw = dc * i * (1 - c_tilde**2)

                # --- Tích lũy Gradient cho trọng số ---
                grad_wf += p.T @ df_raw
                grad_bf += np.sum(df_raw, axis=0, keepdims=True)
                
                grad_wi += p.T @ di_raw
                grad_bi += np.sum(di_raw, axis=0, keepdims=True)
                
                grad_wc += p.T @ dc_tilde_raw
                grad_bc += np.sum(dc_tilde_raw, axis=0, keepdims=True)
                
                grad_wo += p.T @ do_raw
                grad_bo += np.sum(do_raw, axis=0, keepdims=True)

                # --- Tính dh_next và dc_next cho bước t-1 ---
                # dh_next = Tổng đạo hàm từ các cổng truyền về phần h trong p=[h, x]
                # Vì p có 100 cột (50 cho h, 50 cho x), ta lấy 50 hàng đầu của W
                dh_next = (df_raw @ self.w_f[:50, :].T + 
                           di_raw @ self.w_i[:50, :].T + 
                           dc_tilde_raw @ self.w_c[:50, :].T + 
                           do_raw @ self.w_o[:50, :].T)
                
                # Lỗi truyền về Cell state trước đó
                dc_next = dc * f

            # 4. Gradient Clipping & Cập nhật trọng số
            grads = [dL_dwt, dL_dbt, grad_wi, grad_bi, grad_wo, grad_bo, grad_wf, grad_bf, grad_wc, grad_bc]
            for g in grads:
                np.clip(g, -1, 1, out=g)

            self.w_t -= lr * dL_dwt; self.b_t -= lr * dL_dbt
            self.w_i -= lr * grad_wi; self.b_i -= lr * grad_bi
            self.w_o -= lr * grad_wo; self.b_o -= lr * grad_bo
            self.w_f -= lr * grad_wf; self.b_f -= lr * grad_bf
            self.w_c -= lr * grad_wc; self.b_c -= lr * grad_bc
            
            if e % 100 == 0:
                print(f"Epoch {e} | Loss: {loss:.6f}")
            


        

    