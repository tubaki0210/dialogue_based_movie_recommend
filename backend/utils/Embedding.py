from openai import OpenAI
import numpy as np

class Embeddings:
    def __init__(self, key):
        self.client = OpenAI(api_key=key)
    
    def get_embedding_byText(self, text: str, model: str = "text-embedding-3-small"):
        """指定したテキストとモデルで埋め込みベクトルを取得する関数"""
        text = text.replace("\n", " ")
        response = self.client.embeddings.create(
            input=[text], 
            model=model
            # 1536次元
            # model="text-embedding-3-small"
            # 3072次元
            # model="text-embedding-3-large",
            # dimensions=256 # ここで次元数を指定
            )
        return response.data[0].embedding
    

    def get_embeddings_by_text_list(self, texts: list[str], model: str = "text-embedding-3-small"):
        """指定したテキストのリストとモデルで埋め込みベクトルを一括取得する関数"""
        
        # リスト内のすべてのテキストに対して改行コードの削除を行う
        cleaned_texts = [text.replace("\n", " ") for text in texts]
        
        # inputにテキストのリストをそのまま渡す
        response = self.client.embeddings.create(
            input=cleaned_texts, 
            model=model
        )
        
        # レスポンスのdata配列から、すべてのベクトルを取り出してリストとして返す
        return [data.embedding for data in response.data]
    
    def get_embedding_byList(self, text: list, model: str = "text-embedding-3-small"):
        """指定したテキストとモデルで埋め込みベクトルを取得する関数"""
        # text = text.replace("\n", " ")
        response = self.client.embeddings.create(
            input=text, 
            model=model
            # 1536次元
            # model="text-embedding-3-small"
            # 3072次元
            # model="text-embedding-3-large",
            # dimensions=256 # ここで次元数を指定
            )
        embeddings_list = [data.embedding for data in response.data]
    
    # NumPy配列に変換して返すのがおすすめ
        return np.array(embeddings_list)
        # return response.data[0].embedding

# モデル名	料金（100万トークンあたり）
# text-embedding-3-small	$0.02
# text-embedding-3-large	$0.13


# def cosine_similarity(vec_a, vec_b):
#     """2つのベクトルのコサイン類似度を計算する関数"""
#     dot_product = np.dot(vec_a, vec_b)
#     norm_a = np.linalg.norm(vec_a)
#     norm_b = np.linalg.norm(vec_b)
#     return dot_product / (norm_a * norm_b)

# EmbeddingsInstace = Embeddings(api_key)
# text1 = "ラストは泣ける"
# text2 = "最後は感動した"
# emb1 = EmbeddingsInstace.get_embedding_byText(text1)
# emb2 = EmbeddingsInstace.get_embedding_byText(text2)
# emb1_np = np.array(emb1)
# emb2_np = np.array(emb2)
# print(emb1_np - emb2_np)
# emb3 = emb1 - emb2
# text3 = "泣ける"
# text_list = []
#     # モデルを選択
# MODEL = "text-embedding-3-small"

#     # 各テキストの埋め込みベクトルを取得
# embedding1 = EmbeddingsInstace.get_embedding(text1, model=MODEL)
# embedding2 = EmbeddingsInstace.get_embedding(text2, model=MODEL)
# embedding3 = EmbeddingsInstace.get_embedding_byText(text3, model=MODEL)
# print(embedding3)
#     # 類似度を計算して表示
# similarity_1_2 = cosine_similarity(embedding1, embedding2)
# similarity_1_3 = cosine_similarity(embedding1, embedding3)

# print(f"「{text1}」「{text2}」")
# print(f"類似度: {similarity_1_2:.4f}\n") # 結果は1.0に近いほど類似

# print(f"「{text1}」「{text3}」")
# print(f"類似度: {similarity_1_3:.4f}") 

