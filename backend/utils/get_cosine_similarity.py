import numpy as np
def cosine_similarity(vec_a, vec_b):
    """2つのベクトルのコサイン類似度を計算する関数"""
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    return dot_product / (norm_a * norm_b)