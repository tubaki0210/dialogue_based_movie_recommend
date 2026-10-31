import pandas as pd
import numpy as np
def load_feature_embedding_dict(parquet_path):
  """Parquetファイルから {特徴テキスト: 1536次元Vector} の検索辞書を作成"""
  df = pd.read_parquet(parquet_path)
  feature_embedding_dict = {}

  for _, row in df.iterrows():
    texts = row["feature_texts"]
    vectors = row["feature_vectors"]  # shape: (N, 1536)

    # feature_texts がリストの場合、対応するベクトルを1つずつ辞書に格納
    if isinstance(texts, (list, np.ndarray)):
      for text, vec in zip(texts, vectors):
        if text not in feature_embedding_dict:
          feature_embedding_dict[text] = np.array(vec, dtype=np.float32)
    else:
      # 1つのテキストに対して1つのベクトルが格納されている場合
      if texts not in feature_embedding_dict:
        feature_embedding_dict[texts] = np.array(vectors, dtype=np.float32)

  return feature_embedding_dict