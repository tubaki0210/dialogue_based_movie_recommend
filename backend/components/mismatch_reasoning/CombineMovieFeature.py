import numpy as np
from utils.Embedding import Embeddings


class CommonFeatureExtractor:
    """映画の特徴リストからコサイン類似度を用いて共通要素グループ/似ていない要素グループを抽出するクラス"""

    # クラス変数
    MODEL = "text-embedding-3-small"
    SIMILARITY_THRESHOLD = 0.45

    def __init__(self, embedding_client: Embeddings):
        self.embedding_client = embedding_client

    def extract_common_features(self, features_dict: dict[str, list[str]]) -> list[list[str]]:
        """
        1つ目の映画の特徴を基準とし、他の映画の特徴との総当たりコサイン類似度を計算。
        閾値(0.40)を超えた最大の類似度を持つ特徴同士を1つのグループとして
        [[特徴1-1, 特徴2-1, 特徴3-1], [特徴1-2, 特徴2-2, 特徴3-2], ...] の二次元配列で返す。
        """
        titles = list(features_dict.keys())
        if len(titles) < 2:
            return []
        # 1. API呼び出し削減のため、全映画の特徴テキストを1つのリストに集約
        all_texts = []
        movie_slices = []
        start_idx = 0

        for title in titles:
            texts = features_dict[title]
            all_texts.extend(texts)
            end_idx = start_idx + len(texts)
            movie_slices.append((start_idx, end_idx))
            start_idx = end_idx

        # 2. 一括でEmbeddingを取得
        all_embeddings = self.embedding_client.get_embedding_byList(all_texts, model=self.MODEL)

        # 3. コサイン類似度の高速計算用に L2 正規化
        norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
        normalized_embeddings = all_embeddings / np.maximum(norms, 1e-9)

        # 基準映画（1つ目）のデータを準備
        base_slice = movie_slices[0]
        base_vecs = normalized_embeddings[base_slice[0]:base_slice[1]]
        base_texts = features_dict[titles[0]]

        common_groups = []

        # 4. 基準映画の特徴ごとに、他の全映画との総当たり計算
        for base_idx, base_text in enumerate(base_texts):
            base_vec = base_vecs[base_idx]
            current_group = [base_text]  # グループの先頭に基準の特徴を追加
            is_common_in_all_other_movies = True

            for other_movie_idx in range(1, len(titles)):
                other_title = titles[other_movie_idx]
                other_slice = movie_slices[other_movie_idx]
                other_vecs = normalized_embeddings[other_slice[0]:other_slice[1]]
                other_texts = features_dict[other_title]

                # 行列積による一括コサイン類似度計算
                similarities = np.dot(other_vecs, base_vec)

                max_idx = int(np.argmax(similarities))
                max_sim = float(similarities[max_idx])

                # 閾値 0.40 を超えている場合はグループに追加
                if max_sim >= self.SIMILARITY_THRESHOLD:
                    current_group.append(other_texts[max_idx])

            # 比較対象の「すべての映画」で条件を満たした場合のみ結果リストに追加
            if len(current_group) >= 2:
                common_groups.append(current_group)

        return common_groups
