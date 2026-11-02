import numpy as np
from supabase_client import supabase


class FeatureHintExtractor:

  def __init__(self, feature_embedding_dict: dict):
    """初期化: 事前ロード済みの特徴ベクトル辞書のみを保持

    :param feature_embedding_dict: {feature_text: np.ndarray(1536,)} の辞書
    """
    self.feature_embedding_dict = feature_embedding_dict

  def _get_candidate_features_and_vectors(self, movie_ids: list):
    """指定された20作品のIDリストから、Supabaseを介して特徴テキストとベクトル行列を取得"""
    if not movie_ids:
      return [], np.empty((0, 1536), dtype=np.float16)

    # Supabaseの id(bigint) に合わせて int 型のリストに変換
    int_movie_ids = []
    for m_id in movie_ids:
      try:
        int_movie_ids.append(int(m_id))
      except (ValueError, TypeError):
        continue

    # 1. 上位20作品の features(jsonb) を Supabase から一括取得
    res = (
        supabase.table("movies")
        .select("id, features")
        .in_("id", int_movie_ids)
        .execute()
    )

    unique_features = set()
    for row in res.data or []:
      features = row.get("features") or []
      if isinstance(features, list):
        for f in features:
          if f:
            unique_features.add(str(f).strip())

    # 2. 特徴テキストに対応するベクトルを抽出
    valid_texts = []
    valid_vectors = []

    for text in unique_features:
      if text in self.feature_embedding_dict:
        valid_texts.append(text)
        valid_vectors.append(self.feature_embedding_dict[text])

    if valid_vectors:
      feature_matrix = np.vstack(valid_vectors).astype(np.float16)
    else:
      feature_matrix = np.empty((0, 1536), dtype=np.float16)

    return valid_texts, feature_matrix

  def extract_hints(
      self,
      movie_ids,
      short_term_like_vecs,
      preference_labels=None,
      top_n_per_pref=20,
      min_similarity=0.45,
      max_similarity=0.80,
      intra_similarity_threshold=0.75,
  ):
    """20作品の特徴から、各短期的嗜好に合致するヒントを嗜好タグ付きで抽出"""
    valid_texts, F_matrix = self._get_candidate_features_and_vectors(
        movie_ids
    )

    if F_matrix.shape[0] == 0 or len(short_term_like_vecs) == 0:
      return {}

    # 2. コサイン類似度の計算（float32に安全変換して計算誤差・オーバーフローを防止）
    U_like = np.array(short_term_like_vecs, dtype=np.float32)
    if U_like.ndim == 1:
      U_like = U_like.reshape(1, -1)

    F_mat_32 = F_matrix.astype(np.float32)

    U_norm = U_like / (np.linalg.norm(U_like, axis=1, keepdims=True) + 1e-9)
    F_norm = F_mat_32 / (
        np.linalg.norm(F_mat_32, axis=1, keepdims=True) + 1e-9
    )

    sim_matrix = np.dot(U_norm, F_norm.T)

    # 3. 嗜好ラベルの準備
    num_prefs = sim_matrix.shape[0]
    if not preference_labels or len(preference_labels) < num_prefs:
      preference_labels = [f"短期的嗜好 {i + 1}" for i in range(num_prefs)]

    # 4. 嗜好ごとに条件を満たす上位特徴を取り出してタグ付け
    hints_by_pref = {}
    seen_features_global = set()

    for pref_idx in range(num_prefs):
      label = preference_labels[pref_idx]
      sims = sim_matrix[pref_idx]

      sorted_indices = np.argsort(sims)[::-1]

      extracted_for_this_pref = []
      extracted_vecs_for_this_pref = []

      for feat_idx in sorted_indices:
        score = sims[feat_idx]

        # 類似度フィルター
        if score < min_similarity or score > max_similarity:
          continue

        feat_text = valid_texts[feat_idx]
        feat_vec = F_norm[feat_idx]

        if feat_text in seen_features_global:
          continue

        if len(extracted_vecs_for_this_pref) > 0:
          vec_stack = np.vstack(extracted_vecs_for_this_pref)
          intra_sims = np.dot(vec_stack, feat_vec)
          if np.max(intra_sims) > intra_similarity_threshold:
            continue

        seen_features_global.add(feat_text)
        extracted_for_this_pref.append(feat_text)
        extracted_vecs_for_this_pref.append(feat_vec)

        if len(extracted_for_this_pref) >= top_n_per_pref:
          break

      hints_by_pref[label] = extracted_for_this_pref

    return hints_by_pref