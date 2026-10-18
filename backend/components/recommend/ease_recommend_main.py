import numpy as np
import pickle
class EASEPredictor:
    def __init__(self, model_path='model/ease_model_v1.pkl'):
        """
        システムの起動時などに1回だけ呼ばれ、モデルをメモリに保持する
        """
        print("モデルをメモリにロードしています...")
        with open(model_path, 'rb') as f:
            model_package = pickle.load(f)
            
        self.W = model_package['W']
        self.movie_to_idx = model_package['movie_to_idx']
        self.idx_to_movie = model_package['idx_to_movie']
        self.movie_id_to_title = model_package['movie_id_to_title']
        self.num_movies = self.W.shape[0]
        print("ロード完了！")

    def recommend(
            self, 
            user_history, 
            top_k=10
            ):
        """
        予測時にはメモリ上のモデル（self.Wなど）をそのまま使うため、爆速で処理できる
        """
        x_new = np.zeros(self.num_movies)
        
        for item in user_history:
            m_id = item['movie_id']
            rating = item['rating']
            if m_id in self.movie_to_idx:
                internal_idx = self.movie_to_idx[m_id]
                x_new[internal_idx] = float(rating) - 3.0
                
        # メモリ上の行列と計算
        predictions = x_new.dot(self.W)
        
        for item in user_history:
            m_id = item['movie_id']
            if m_id in self.movie_to_idx:
                internal_idx = self.movie_to_idx[m_id]
                predictions[internal_idx] = -np.inf
                
        top_indices = np.argsort(predictions)[::-1][:self.num_movies]
        
        recommendations = []
        for rank, idx in enumerate(top_indices):
            m_id = self.idx_to_movie[idx]
            recommendations.append({
                "rank": rank + 1,
                "movie_id": m_id,
                "title": self.movie_id_to_title.get(m_id, "不明なタイトル"),
                "cf_score": round(float(predictions[idx]), 4)
            })
            
        return recommendations