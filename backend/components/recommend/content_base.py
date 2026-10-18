import pandas as pd
import numpy as np
from utils.Embedding import Embeddings
from sklearn.metrics.pairwise import cosine_similarity
class ContentBasePredictor:
    def __init__(self, parquet_path, embedding_client : Embeddings):
        df = pd.read_parquet(parquet_path)
        # IDの配列を保持
        self.movie_ids = df['movie_id'].values
        
        # df['embedding'] がリストや配列で保存されている前提で、2次元のNumPy配列に変換
        # shape: (映画の数, ベクトルの次元数)
        self.embeddings = np.stack(df['embedding'].values)
        
        # 入力テキストをベクトル化するためのクライアント（外部APIやローカルモデル）
        self.embedding_client = embedding_client
        
        print(f"ロード完了！ 映画数: {self.embeddings.shape[0]}, 次元数: {self.embeddings.shape[1]}")

    def get_all_raw_scores(
            self, 
            like_str: str, 
            dislike_str: str, 
            dislike_weight=0.5
            ):
        """
        like/dislikeテキストを受け取り、全映画に対するスコア(コサイン類似度)を返す
        """
        # 1. テキストをベクトル化
        if like_str.strip():
            # 例: self.embedding_client.get_embedding("SF 宇宙 アクション")
            like_vec = np.array(self.embedding_client.get_embedding_byText(like_str))
        else:
            like_vec = np.zeros(self.embeddings.shape[1])
            
        if dislike_str.strip():
            dislike_vec = np.array(self.embedding_client.get_embedding_byText(dislike_str))
        else:
            dislike_vec = np.zeros(self.embeddings.shape[1])
            
        # 2. 検索用ベクトルの合成 (Like - α * Dislike)
        query_vec = like_vec - (dislike_weight * dislike_vec)
        
        # ゼロベクトルの場合は計算できないため、適当にスコア0を返す
        if np.all(query_vec == 0):
            return {int(m_id): 0.0 for m_id in self.movie_ids}

        # 3. コサイン類似度の計算
        # query_vecを (1, 次元数) の形に変形
        query_vec_2d = query_vec.reshape(1, -1)
        
        # cosine_similarity は (1, 映画数) の配列を返すので [0] で1次元に戻す
        sim_scores = cosine_similarity(query_vec_2d, self.embeddings)[0]
        
        # 4. movie_id をキー、スコアを値とする辞書を作成
        raw_scores = {int(m_id): float(score) for m_id, score in zip(self.movie_ids, sim_scores)}
        
        return raw_scores