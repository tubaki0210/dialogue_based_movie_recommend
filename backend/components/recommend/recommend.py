import numpy as np
from .ease_recommend_main import EASEPredictor
from .content_base import ContentBasePredictor
from .llm_recommend import LLMReranker
class RecommendManager:
    TOP_NUM_FIRST = 20
    TOP_NUM_FINAL = 3
    CF_PARAM = 0.2
    CB_PARAM = 0.8
    def __init__(self, ease_predictor : EASEPredictor, content_base_predictor : ContentBasePredictor, llm_rerank_predictor : LLMReranker):
        self.ease_predictor = ease_predictor
        self.content_base_predictor = content_base_predictor
        self.llm_rerank_predictor = llm_rerank_predictor

    def recommend(
            self, 
            user_watch_history, 
            already_recommend_set, 
            short_term_like_list, 
            short_term_dislike_list, 
            long_term_like_list, 
            long_term_dislike_list
            ):
        # 嗜好をリスト形式から文字列に変更
        short_term_like_str = self._format_preference(short_term_like_list)
        short_term_dislike_str = self._format_preference(short_term_dislike_list)
        long_term_like_str = self._format_preference(long_term_like_list)
        long_term_dislike_str = self._format_preference(long_term_dislike_list)
        
        # 1段階目(協調フィルタリング＋コンテンツベースフィルタリング)
        state1_candidate = self._state1_ensemble(
            user_watch_history, 
            already_recommend_set, 
            short_term_like_str, 
            short_term_dislike_str
            )
        # return state1_candidate
        # print(state1_candidate)

        # 2段階目(LLM)
        reponse = self.llm_rerank_predictor.rerank(
            state1_candidate, 
            short_term_like_str, 
            short_term_dislike_str, 
            long_term_like_str, 
            long_term_dislike_str, 
            self.TOP_NUM_FINAL
            )
        return reponse
    
    def _state1_ensemble(
            self, 
            user_watch_history, 
            already_recommend_set, 
            short_term_like_str, 
            short_term_dislike_str
            ):
        # 協調フィルタリング
        cf_recommend_result = self.ease_predictor.recommend(user_watch_history, self.TOP_NUM_FINAL)
        # コンテンツベースフィルタリング
        cb_recommend_result = self.content_base_predictor.get_all_raw_scores(short_term_like_str, short_term_dislike_str)
        # 協調フィルタリングとコンテンツベースフィルタリングの結果を足して、ソート
        filtered_result = [
            m for m in cf_recommend_result 
            if m.get("title") not in already_recommend_set and not np.isneginf(m["cf_score"])
        ]

        # -inf を除外した有限なスコアのみを抽出
        valid_cf_scores = [m["cf_score"] for m in filtered_result if np.isfinite(m["cf_score"])]
        cf_min, cf_max = min(valid_cf_scores), max(valid_cf_scores)

        for rec_movie in filtered_result:
            cf_score = rec_movie["cf_score"]
            # 2. CFスコアをそのユーザー内で 0.0 〜 1.0 に正規化（相対評価に変換）
            if cf_max == cf_min:
                norm_cf = 0.5  # 全スコア同値の回避
            else:
                norm_cf = (cf_score - cf_min) / (cf_max - cf_min)
            rec_movie["cf_score_normalized"] = norm_cf
            
            movie_id = rec_movie.get("movie_id")
            cb_score = cb_recommend_result.get(movie_id, 0.0)
            rec_movie["cb_score"] = cb_score

            cf_multiplier = 0.95 + (norm_cf * 0.1)
            
            rec_movie["total_score"] = cb_score * cf_multiplier

        stage1_final_result = sorted(filtered_result, key=lambda x : x["total_score"], reverse=True)
        # print(stage1_final_result)
        # 順位を振りなおす
        for rank, rec_movie in enumerate(stage1_final_result):
            rec_movie["rank"] = rank + 1

        return stage1_final_result[:self.TOP_NUM_FIRST]

    def _format_preference(self, pref):
        return "、".join(pref)