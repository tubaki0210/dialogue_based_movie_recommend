import pandas as pd
import numpy as np
from utils.Embedding import Embeddings
from sklearn.metrics.pairwise import cosine_similarity
class ContentBasePredictor:
    DISLIKE_WEIGHT = 0.3
    DISLIKE_THRESHOLD = 0.45
    
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
    ):
        """
        like/dislikeテキストを受け取り、ハイブリッド方式（ハード除外 + スコア減算）で
        全映画に対する統合スコアを算出して返す
        """
        has_like = bool(like_str.strip())
        has_dislike = bool(dislike_str.strip())

        # どちらも指定されていない場合は全映画スコア 0.0 を返却
        if not has_like and not has_dislike:
            return {int(m_id): 0.0 for m_id in self.movie_ids}

        # 1. 好む要素のコサイン類似度計算 (sim_like)
        if has_like:
            formatted_like = f"ユーザーが好む映画の特徴：{like_str.strip()}"
            like_vec = np.array(self.embedding_client.get_embedding_byText(formatted_like)).reshape(1, -1)
            sim_like = cosine_similarity(like_vec, self.embeddings)[0]
        else:
            sim_like = np.zeros(self.embeddings.shape[0])

        # 2. 避ける要素のコサイン類似度計算 (sim_dislike)
        if has_dislike:
            formatted_dislike = f"ユーザーが避ける映画の特徴：{dislike_str.strip()}"
            dislike_vec = np.array(self.embedding_client.get_embedding_byText(formatted_dislike)).reshape(1, -1)
            sim_dislike = cosine_similarity(dislike_vec, self.embeddings)[0]
        else:
            sim_dislike = np.zeros(self.embeddings.shape[0])


        # 3. 基本スコア算出： (Like類似度) - α * (Dislike類似度)
        final_scores_arr = sim_like - (self.DISLIKE_WEIGHT * sim_dislike)

        # 4. ハードフィルター（地雷除外）
        # Dislikeとの類似度が閾値（デフォルト0.45）以上の作品はスコアを -1.0 に強制上書きして確実に除外する
        if has_dislike:
            dislike_mask = sim_dislike >= self.DISLIKE_THRESHOLD
            final_scores_arr[dislike_mask] = -1.0

        # 5. movie_id をキー、最終スコアを値とする辞書を作成して返却
        raw_scores = {int(m_id): float(score) for m_id, score in zip(self.movie_ids, final_scores_arr)}

        return raw_scores