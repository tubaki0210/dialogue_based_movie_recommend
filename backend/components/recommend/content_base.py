import gc
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from utils.Embedding import Embeddings


class ContentBasePredictor:

  DISLIKE_THRESHOLD = 0.60

  def __init__(self, parquet_path, embedding_client: Embeddings):
    self.parquet_path = parquet_path
    self.embedding_client = embedding_client
    self.feature_embedding_dict = {}  # float16 のまま辞書を保持
    self._load_parquet_data()

  def _load_parquet_data(self):
    """Parquetファイルを float16 でメモリ効率よくロードし、辞書も正確に構築する"""
    df = pd.read_parquet(self.parquet_path)

    self.movie_ids = [int(m_id) for m_id in df["movie_id"].unique()]

    flat_movie_ids = []
    flat_vectors = []

    for _, row in df.iterrows():
      m_id = int(row["movie_id"])
      f_vecs = row.get("feature_vectors")
      features = row.get("feature_texts")

      if f_vecs is not None and len(f_vecs) > 0:
        f_arr = np.vstack(f_vecs).astype(np.float16)

        for vec in f_arr:
          flat_movie_ids.append(m_id)
          flat_vectors.append(vec)

        # ★ 修正: features（テキスト）と feature_vectors（ベクトル）を正確に1対1マッピング
        if features is not None and len(features) == len(f_vecs):
          for text, vec in zip(features, f_arr):
            if text:
              clean_text = str(text).strip()
              if clean_text not in self.feature_embedding_dict:
                self.feature_embedding_dict[clean_text] = vec

    self.flat_movie_ids = np.array(flat_movie_ids, dtype=np.int32)
    self.all_feature_embeddings = np.vstack(flat_vectors).astype(np.float16)

    print(
        "【ContentBasePredictor】 辞書登録完了数:"
        f" {len(self.feature_embedding_dict)} 件"
    )

    del df, flat_vectors, flat_movie_ids
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

      df_like = pd.DataFrame(sim_like_matrix.T, index=self.flat_movie_ids)
      like_scores_series = df_like.groupby(level=0).max().mean(axis=1)
    else:
      like_scores_series = pd.Series(0.0, index=self.movie_ids)

    if clean_dislikes:
      raw_dislike_vecs = self.embedding_client.get_embedding_byList(
          clean_dislikes
      )
      dislike_vecs = np.vstack(raw_dislike_vecs).astype(np.float16)

      sim_dislike_matrix = cosine_similarity(
          dislike_vecs, self.all_feature_embeddings
      )

      df_dislike = pd.DataFrame(
          sim_dislike_matrix.T, index=self.flat_movie_ids
      )
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