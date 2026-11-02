import numpy as np


class FeatureHintExtractor:
  def __init__(self, movie_data_json, feature_embedding_dict):
    """初期化: 映画メタデータと事前ロード済みのベクトル辞書を保持

    :param movie_data_json: 映画データ辞書
    :param feature_embedding_dict: {feature_text: np.ndarray(1536,)} の辞書
    """
    self.movie_data_json = movie_data_json
    self.feature_embedding_dict = feature_embedding_dict

  def _get_candidate_features_and_vectors(self, movie_ids):
    """[ステップ2相当] 指定された映画IDリストからユニークな特徴とベクトル行列を取得"""
    unique_features = set()
    for m_id in movie_ids:
      features = self.movie_data_json.get(str(m_id), {}).get("features", [])
      if isinstance(features, list):
        unique_features.update(features)

    valid_texts = []
    valid_vectors = []

    for text in unique_features:
      if text in self.feature_embedding_dict:
        valid_texts.append(text)
        valid_vectors.append(self.feature_embedding_dict[text])

    if valid_vectors:
      feature_matrix = np.vstack(valid_vectors).astype(np.float32)
    else:
      feature_matrix = np.empty((0, 1536), dtype=np.float32)

    return valid_texts, feature_matrix

  def extract_hints(
      self,
      movie_ids,
      short_term_like_vecs,
      preference_labels=None,
      top_n_per_pref=3,
      min_similarity = 0.45,
      max_similarity=0.80,
      intra_similarity_threshold=0.75,  # 採用済み特徴同士の類似度上限
  ):
    """[ステップ2+3統合] 20作品の特徴から、各短期的嗜好に合致するヒントを嗜好タグ付きで抽出

    :param movie_ids: 1段階目で絞り込んだ20作品のIDリスト
    :param short_term_like_vecs: ユーザーの短期的嗜好ベクトル行列 (K, 1536)
    :param preference_labels: 各嗜好のラベル名リスト (例: ["嗜好A: SF", "嗜好B: ドラマ"])
    :param top_n_per_pref: 各嗜好ごとに抽出する特徴の数
    :param output_format: 'text' (LLMプロンプト用文字列) または 'dict' (辞書型)
    :return: 嗜好ごとにタグ付けされたヒントデータ
    """
    # 1. 20作品から特徴テキストとベクトル行列を取得
    valid_texts, F_matrix = self._get_candidate_features_and_vectors(
        movie_ids
    )

    # 2. コサイン類似度の一括計算 (K, M)
    U_like = np.array(short_term_like_vecs, dtype=np.float32)
    U_norm = U_like / (np.linalg.norm(U_like, axis=1, keepdims=True) + 1e-9)
    F_norm = F_matrix / (
        np.linalg.norm(F_matrix, axis=1, keepdims=True) + 1e-9
    )

    # 行列積: (K, 1536) x (1536, M) -> (K, M)
    sim_matrix = np.dot(U_norm, F_norm.T)

    # 3. 嗜好ラベルの準備
    num_prefs = sim_matrix.shape[0]
    if not preference_labels or len(preference_labels) < num_prefs:
      preference_labels = [
          f"短期的嗜好 {i + 1}" for i in range(num_prefs)
      ]

    # 4. 嗜好ごとに条件を満たす上位特徴を取り出してタグ付け
    hints_by_pref = {}
    seen_features_global = set()  # 他の嗜好と重複させたくない場合に使用

    for pref_idx in range(num_prefs):
        label = preference_labels[pref_idx]
        sims = sim_matrix[pref_idx]

        # 類似度が高い順にインデックスをソート
        sorted_indices = np.argsort(sims)[::-1]

        extracted_for_this_pref = []
        extracted_vecs_for_this_pref = []

        for feat_idx in sorted_indices:
            score = sims[feat_idx]

            # 嗜好ベクトルとの類似度が max_similarity (0.80) を超えるものは無視してスキップ
            if score < min_similarity or score > max_similarity:
                continue

            feat_text = valid_texts[feat_idx]
            feat_vec = F_norm[feat_idx]

            # 既に他の嗜好で使われた特徴をスキップ（被りを防ぎ多様性を確保）
            if feat_text in seen_features_global:
                continue

        # 条件3: 同じ嗜好内で採択済みの特徴と似すぎていないか（多様性の確保）
            if len(extracted_vecs_for_this_pref) > 0:
                intra_sims = np.dot(extracted_vecs_for_this_pref, feat_vec)
                if np.max(intra_sims) > intra_similarity_threshold:
                    continue

            seen_features_global.add(feat_text)
            extracted_for_this_pref.append(feat_text)
            extracted_vecs_for_this_pref.append(feat_vec)

            if len(extracted_for_this_pref) >= top_n_per_pref:
                break

        hints_by_pref[label] = extracted_for_this_pref

    # 5. 出力フォーマットの整形
    return hints_by_pref