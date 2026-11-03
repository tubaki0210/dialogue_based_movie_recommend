import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasePredictor:
    DISLIKE_THRESHOLD = 0.60
    
    def __init__(self, parquet_path, embedding_client):
        self.parquet_path = parquet_path
        self.embedding_client = embedding_client
        
        # Parquetを1回だけ読み込んで、Predictor用とDict用の両方を効率よく生成する
        self._load_and_prepare_data()

    def _load_and_prepare_data(self):
        """
        iterrowsを使わず、ベクトル演算ベースで高速にデータを展開する
        """
        df = pd.read_parquet(self.parquet_path)
        
        flat_movie_ids = []
        flat_vectors = []
        feature_embedding_dict = {}

        # iterrows を避け、values や to_dict でループを高速化
        for m_id, f_texts, f_vecs in zip(df["movie_id"], df["feature_texts"], df["feature_vectors"]):
            m_id_int = int(m_id)
            
            if f_vecs is not None and len(f_vecs) > 0:
                f_arr = np.vstack(f_vecs).astype(np.float32)
                
                # ContentBasePredictor 用のフラットデータ蓄積
                for vec in f_arr:
                    flat_movie_ids.append(m_id_int)
                    flat_vectors.append(vec)
                
                # FeatureHintExtractor 用の辞書も同時に構築（1回のループで済ませる）
                if isinstance(f_texts, (list, np.ndarray)):
                    for text, vec in zip(f_texts, f_arr):
                        if text not in feature_embedding_dict:
                            feature_embedding_dict[text] = vec
                else:
                    if f_texts not in feature_embedding_dict:
                        feature_embedding_dict[f_texts] = f_arr[0] if len(f_arr) > 0 else np.zeros(768, dtype=np.float32)

        # 保持データのセット
        self.movie_ids = [int(m_id) for m_id in df["movie_id"].unique()]
        self.flat_movie_ids = np.array(flat_movie_ids)
        self.all_feature_embeddings = np.vstack(flat_vectors).astype(np.float32)
        
        # 外部で必要となる辞書をインスタンス変数または外部参照用に保持
        self.feature_embedding_dict = feature_embedding_dict

    def get_all_raw_scores(self, like_list: list[str], dislike_list: list[str]):
        # (※ 既存のロジックはそのまま維持)
        clean_likes = [s.strip() for s in like_list if s and s.strip()]
        clean_dislikes = [s.strip() for s in dislike_list if s and s.strip()]

        if not clean_likes and not clean_dislikes:
            return {int(m_id): 0.0 for m_id in self.movie_ids}, None

        if clean_likes:
            raw_like_vecs = self.embedding_client.get_embedding_byList(clean_likes)
            like_vecs = np.vstack(raw_like_vecs).astype(np.float32)
            sim_like_matrix = cosine_similarity(like_vecs, self.all_feature_embeddings)
            df_like = pd.DataFrame(sim_like_matrix.T, index=self.flat_movie_ids)
            like_scores_series = df_like.groupby(level=0).max().mean(axis=1)
        else:
            like_scores_series = pd.Series(0.0, index=self.movie_ids)
            like_vecs = None

        if clean_dislikes:
            raw_dislike_v = self.embedding_client.get_embedding_byList(clean_dislikes)
            dislike_vecs = np.vstack(raw_dislike_v).astype(np.float32)
            sim_dislike_matrix = cosine_similarity(dislike_vecs, self.all_feature_embeddings)
            df_dislike = pd.DataFrame(sim_dislike_matrix.T, index=self.flat_movie_ids)
            dislike_max_series = df_dislike.groupby(level=0).max().max(axis=1)
        else:
            dislike_max_series = pd.Series(0.0, index=self.movie_ids)

        raw_scores = {}
        for m_id in self.movie_ids:
            m_id_int = int(m_id)
            max_sim_dislike = float(dislike_max_series.get(m_id_int, 0.0))
            if clean_dislikes and max_sim_dislike >= self.DISLIKE_THRESHOLD:
                raw_scores[m_id_int] = -1.0
            else:
                like_score = float(like_scores_series.get(m_id_int, 0.0))
                raw_scores[m_id_int] = like_score

        return raw_scores, like_vecs