import numpy as np
import re
import pandas as pd
import nltk
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

def preprocess_for_glove(sentence):
    # 1. Lowercase (Bắt buộc)
    # 2. Dùng bộ não tự động của NLTK
    tokens = nltk.word_tokenize(sentence.lower())
    return tokens

def text_to_matrix(tokens, glove_df, maxlength=1000, embed_dim=50):
    # Lấy vector cho từng token, nếu không có thì dùng vector 0
    # Chuyển token về lowercase vì GloVe dùng chữ thường
    matrix = [glove_df.loc[t.lower()].values if t.lower() in glove_df.index 
              else np.zeros(embed_dim) for t in tokens]
    
    # 1. Truncating (Cắt bớt nếu quá dài)
    if len(matrix) > maxlength:
        matrix = matrix[:maxlength]
    
    # 2. Padding (Thêm vector 0 nếu quá ngắn)
    else:
        padding_len = maxlength - len(matrix)
        padding = [np.zeros(embed_dim)] * padding_len
        matrix.extend(padding)
        
    return np.array(matrix)