import numpy as np
import re
import pandas as pd
def clean_and_tokenize(text):
    # Chuyển về chữ thường và xóa ký tự đặc biệt
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Tách từ bằng khoảng trắng
    tokens = text.split()
    return tokens

def build_vocab(all_tokens, max_vocab_size=10000):
    # Đếm tần suất xuất hiện (tùy chọn để lọc từ hiếm)
    from collections import Counter
    word_counts = Counter(all_tokens)
    
    # Lấy các từ phổ biến nhất
    common_words = word_counts.most_common(max_vocab_size)
    
    # Tạo dict: 0 cho PAD (bù dòng), 1 cho UNK (từ lạ)
    word_to_id = {"<PAD>": 0, "<UNK>": 1}
    for i, (word, _) in enumerate(common_words):
        word_to_id[word] = i + 2 # Bắt đầu từ 2
        
    return word_to_id

def text_to_sequence(tokens, word_to_id):
    # Nếu từ không có trong từ điển, dùng ID của <UNK> (là 1)
    sequence = [word_to_id.get(word, 1) for word in tokens]
    return np.array(sequence)

def pad_sequence(sequence, max_length):
    padded = np.zeros(max_length, dtype=int)
    # Cắt bớt nếu quá dài, hoặc giữ nguyên nếu ngắn hơn
    actual_len = min(len(sequence), max_length)
    padded[:actual_len] = sequence[:actual_len]
    return padded

file_path = 'pretrained/glove.6B.50d.txt' 

# Đọc file
# - sep=' ': dùng dấu cách làm phân tách
# - index_col=0: đưa cột từ vựng đầu tiên làm chỉ số (index)
# - quoting=3: (csv.QUOTE_NONE) giúp tránh lỗi nếu có từ chứa dấu ngoặc kép
df = pd.read_csv(file_path, sep=' ', index_col=0, header=None, quoting=3)
import nltk
from nltk.tokenize import RegexpTokenizer

# Regex này ưu tiên giữ lại các cụm chữ-chấm-chữ (u.n.) 
# và các từ có dấu nháy (don't)
pattern = r'\w+(?:\.\w+)+|\w+(?:\'\w+)?|\w+|[^\w\s]'
tokenizer = RegexpTokenizer(pattern)

text = "The u.n. meeting in the U.S. didn't start yet."
tokens = tokenizer.tokenize(text.lower())

print(tokens)
# Kết quả mong đợi: ['the', 'u.n.', 'meeting', 'in', 'the', 'u.s.', 'did', "n't", 'start', 'yet', '.']
def check_coverage(tokens, glove_df):
    matched = [t for t in tokens if t in glove_df.index]
    coverage = len(matched) / len(tokens) * 100
    oov = [t for t in tokens if t not in glove_df.index] # Out of vocabulary
    return coverage, oov

# Sử dụng
coverage, missing = check_coverage(tokens, df)
print(f"Độ khớp: {coverage}%")
print(f"Từ bị thiếu: {missing}")
print(df.loc["n't"])