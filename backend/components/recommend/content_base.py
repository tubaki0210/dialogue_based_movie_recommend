import gc
import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics.pairwise import cosine_similarity
from utils.Embedding import Embeddings


class ContentBasePredictor:

  DISLIKE_THRESHOLD = 0.60

  def __init__(self, parquet_path, embedding_client: Embeddings):
    self.parquet_path = parquet_path
    self.embedding_client = embedding_client
    self.feature_embedding_dict = {}
    self._load_parquet_data()

  def _load_parquet_data(self):
    """pyarrow を使用して低メモリ・高速に Parquet データを展開する"""
    # 1. pyarrow でテーブルとして直接ロード (Pandasのオーバーヘッドを回避)
    table = pq.read_table(
        self.parquet_path,
        columns=["movie_id", "feature_vectors", "feature_texts"],
    )

    movie_id_col = table["movie_id"].to_pylist()
    feature_vecs_col = table["feature_vectors"].to_pylist()
    features_col = (
        table["feature_texts"].to_pylist() if "feature_texts" in table.column_names else []
    )

    # 全映画ID
    self.movie_ids = list(dict.fromkeys(movie_id_col))

    flat_movie_ids = []
    all_vecs_list = []

    # 2. フラット展開処理
    for i in range(len(movie_id_col)):
      m_id = int(movie_id_col[i])
      f_vecs = feature_vecs_col[i]
      feats = features_col[i] if i < len(features_col) else None

      if f_vecs is not None and len(f_vecs) > 0:
        f_arr = np.array(f_vecs, dtype=np.float16)
        all_vecs_list.append(f_arr)
        flat_movie_ids.extend([m_id] * len(f_arr))

        # 辞書の作成
        if feats is not None and len(feats) == len(f_arr):
          for text, vec in zip(feats, f_arr):
            if text:
              clean_text = str(text).strip()
              if clean_text not in self.feature_embedding_dict:
                self.feature_embedding_dict[clean_text] = vec

    # 3. 単一の巨大なNumPy行列へ結合
    self.flat_movie_ids = np.array(flat_movie_ids, dtype=np.int32)
    self.all_feature_embeddings = np.vstack(all_vecs_list)

    # 不要になったPyArrowテーブルと中間リストを即座に破棄
    del table, movie_id_col, feature_vecs_col, features_col, all_vecs_list
    gc.collect()

  def get_all_raw_scores(
      self,
      like_list: list[str],
      dislike_list: list[str],
  ) -> tuple[dict[int, float], np.ndarray]:
    clean_likes = [s.strip() for s in like_list if s and s.strip()]
    clean_dislikes = [s.strip() for s in dislike_list if s and s.strip()]

    embedding_dim = self.all_feature_embeddings.shape[1]
    like_vecs = np.empty((0, embedding_dim), dtype=np.float16)

    if not clean_likes and not clean_dislikes:
      return {int(m_id): 0.0 for m_id in self.movie_ids}, like_vecs

    if clean_likes:
      raw_like_vecs = self.embedding_client.get_embedding_byList(clean_likes)
      like_vecs = np.vstack(raw_like_vecs).astype(np.float16)

      sim_like_matrix = cosine_similarity(
          like_vecs, self.all_feature_embeddings
      )

      # pandasを使わずNumPyで直接グループ最大値を計算し、メモリ・計算量を削減
      like_scores_series = {}
      for u_idx in range(len(clean_likes)):
        sims = sim_like_matrix[u_idx]
        # movie_id ごとの max を高速集計
        unq_ids, unq_inv = np.unique(self.flat_movie_ids, return_inverse=True)
        max_sims = np.full(len(unq_ids), -np.inf, dtype=np.float16)
        np.maximum.at(max_sims, unq_inv, sims)

        for m_id, score in zip(unq_ids, max_sims):
          like_scores_series[m_id] = (
              like_scores_series.get(m_id, 0.0) + score / len(clean_likes)
          )
    else:
      like_scores_series = {m_id: 0.0 for m_id in self.movie_ids}

    if clean_dislikes:
      raw_dislike_vecs = self.embedding_client.get_embedding_byList(
          clean_dislikes
      )
      dislike_vecs = np.vstack(raw_dislike_vecs).astype(np.float16)

      sim_dislike_matrix = cosine_similarity(
          dislike_vecs, self.all_feature_embeddings
      )

      unq_ids, unq_inv = np.unique(self.flat_movie_ids, return_inverse=True)
      max_dislike_sims = np.full(len(unq_ids), -np.inf, dtype=np.float16)
      np.maximum.at(max_dislike_sims, unq_inv, sim_dislike_matrix.max(axis=0))
      dislike_max_series = dict(zip(unq_ids, max_dislike_sims))
    else:
      dislike_max_series = {m_id: 0.0 for m_id in self.movie_ids}

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