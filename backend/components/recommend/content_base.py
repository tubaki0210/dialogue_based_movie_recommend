import pandas as pd
import numpy as np
from utils.Embedding import Embeddings
from sklearn.metrics.pairwise import cosine_similarity
class ContentBasePredictor:
    DISLIKE_THRESHOLD = 0.60
    def __init__(self, parquet_path, embedding_client : Embeddings):
        self.parquet_path = parquet_path
        # 入力テキストをベクトル化するためのクライアント（外部APIやローカルモデル）
        self.embedding_client = embedding_client
        self._load_parquet_data()


    def _load_parquet_data(self):
        """
        Parquetファイルから映画IDと特徴ベクトルをフラットに展開し、
        一括計算用の2次元NumPy配列を生成する
        """
        df = pd.read_parquet(self.parquet_path)
        
        flat_movie_ids = []
        flat_vectors = []
        
        for _, row in df.iterrows():
            m_id = int(row["movie_id"])
            f_vecs = row["feature_vectors"]  # shape: (features要素数, 768)
            
            if f_vecs is not None and len(f_vecs) > 0:
                # リストのリストやobject配列を (N_features, 768) の 2次元 float32 配列に成形
                f_arr = np.vstack(f_vecs).astype(np.float32)
                for vec in f_arr:
                    flat_movie_ids.append(m_id)
                    flat_vectors.append(vec)

        # 存在している全movie_id（特徴のない映画も漏れなく保持）
        self.movie_ids = [int(m_id) for m_id in df["movie_id"].unique()]
        # 各特徴ベクトルがどの movie_id に属するかを示す配列 shape: (全特徴数,)
        self.flat_movie_ids = np.array(flat_movie_ids)
        # 全映画の全特徴ベクトルをまとめた2次元行列 shape: (全特徴数, 768)
        self.all_feature_embeddings = np.vstack(flat_vectors).astype(np.float32)

    def get_all_raw_scores(
        self, 
        like_list: list[str], 
        dislike_list: list[str], 
    ) -> dict[int, float]:
        """
        like/dislike の要素リストを受け取り、要素単位のハイブリッド方式（ハード除外 + スコア算出）で
        全映画に対する統合スコアを算出して返す
        """
        # 空文字列や空白のみの要素を除外したクリーンなリストを作成
        clean_likes = [s.strip() for s in like_list if s and s.strip()]
        clean_dislikes = [s.strip() for s in dislike_list if s and s.strip()]

        # どちらも指定されていない場合は全映画スコア 0.0 を返却
        if not clean_likes and not clean_dislikes:
            return {int(m_id): 0.0 for m_id in self.movie_ids}

        print(f"Like List: {clean_likes} | Dislike List: {clean_dislikes}")

        # 1. 好む要素（Like）のコサイン類似度計算
        if clean_likes:
            # get_embedding_byList で一括ベクトル取得 (shape: [N_likes, dim])
            raw_like_vecs = self.embedding_client.get_embedding_byList(clean_likes)
            like_vecs = np.vstack(raw_like_vecs).astype(np.float32)
            
            # 全特徴量との類似度行列を一括計算 (shape: [N_likes, N_all_features])
            sim_like_matrix = cosine_similarity(like_vecs, self.all_feature_embeddings)
            
            # Pandas を用いた映画単位での高速集約
            # 各映画に属する特徴量の中で、各 Like アイテムに対する「最大類似度」を取得し、その「平均値」を Like スコアとする
            df_like = pd.DataFrame(sim_like_matrix.T, index=self.flat_movie_ids)
            like_scores_series = df_like.groupby(level=0).max().mean(axis=1)
        else:
            like_scores_series = pd.Series(0.0, index=self.movie_ids)

        # 2. 避ける要素（Dislike）のコサイン類似度計算
        if clean_dislikes:
            # get_embedding_byList で一括ベクトル取得 (shape: [N_dislikes, dim])
            raw_dislike_vecs = self.embedding_client.get_embedding_byList(clean_dislikes)
            dislike_vecs = np.vstack(raw_dislike_vecs).astype(np.float32)
            
            # 全特徴量との類似度行列を一括計算 (shape: [N_dislikes, N_all_features])
            sim_dislike_matrix = cosine_similarity(dislike_vecs, self.all_feature_embeddings)
            
            # 映画ごとに、全 Dislike キーワード × 全特徴量の中での「最大類似度」を取得（地雷判定用）
            df_dislike = pd.DataFrame(sim_dislike_matrix.T, index=self.flat_movie_ids)
            dislike_max_series = df_dislike.groupby(level=0).max().max(axis=1)
        else:
            dislike_max_series = pd.Series(0.0, index=self.movie_ids)

        # 3. 映画ごとの統合スコア決定およびハードフィルター（地雷除外）の適用
        raw_scores = {}
        for m_id in self.movie_ids:
            m_id_int = int(m_id)
            
            # この映画に対する Dislike の最大類似度を取得
            max_sim_dislike = float(dislike_max_series.get(m_id_int, 0.0))

            # ハードフィルター：1つでも Dislike 類似度が閾値（0.45）以上であれば -1.0 で確実に弾く
            if clean_dislikes and max_sim_dislike >= self.DISLIKE_THRESHOLD:
                raw_scores[m_id_int] = -1.0
            else:
                like_score = float(like_scores_series.get(m_id_int, 0.0))
                raw_scores[m_id_int] = like_score

        return raw_scores