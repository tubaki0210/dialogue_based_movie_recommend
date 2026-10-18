import numpy as np
from utils.Embedding import Embeddings
class ExtractUnSimCluster:

    MODEL = "text-embedding-3-small"
    SIMILARITY_THRESHOLD = 0.40
    
    def __init__(self, embedding_client : Embeddings):
        self.embedding_client = embedding_client
    
    def _cosine_similarity(self, a, b):
        """コサイン類似度を計算するヘルパーメソッド"""
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def extract(self, cluster_list, short_term_like_pre):
        # プレフィックスの埋め込みベクトルを取得
        pre_emb = self.embedding_client.get_embedding_byText(" ".join(short_term_like_pre))
        
        un_sim_clusters = []
        
        for cluster in cluster_list:
            join_cluster = " ".join(cluster)
            cluster_emb = self.embedding_client.get_embedding_byText(join_cluster)
            
            # コサイン類似度を計算
            similarity = self._cosine_similarity(pre_emb, cluster_emb)
            
            # SIMILARITY_THRESHOLD 未満のクラスターを抽出
            if similarity < self.SIMILARITY_THRESHOLD:
                un_sim_clusters.append(cluster)
                
        return un_sim_clusters