from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
@dataclass
class QuestionContext:
    """対話の「現在の状態」をすべて保持するオブジェクト"""
    user_profile: Dict[str, Any] = field(default_factory=dict)                   # ユーザーの嗜好プロファイル
    conversation_history: List[Dict[str, str]] = field(default_factory=list)      # これまでの対話履歴
    mismatch_results: List[Dict[str, str | List]] = field(default_factory=list) # ミスマッチ推論の結果
    un_mismatch_results : List[Dict[str, str | List ]]  = field(default_factory=list)
    already_recommended_titles: Set[str] = field(default_factory=set)

    @property
    def watch_record(self) -> list:
        return self.user_profile.get("watch_record", [])

    @property
    def long_term_like(self) -> list:
        return self.user_profile.get("like_long_preference", [])

    @property
    def long_term_dislike(self) -> list:
        return self.user_profile.get("dislike_long_preference", [])

    @property
    def short_term_like(self) -> list:
        return self.user_profile.get("short_term", {}).get("like", [])

    @property
    def short_term_dislike(self) -> list:
        return self.user_profile.get("short_term", {}).get("dislike", [])    

    def update_short_term_preference(self, extracted_prefs: Dict[str, List[str]]):
        """抽出された短期的嗜好を context の user_profile["short_term"] に追加する"""
        if "short_term" not in self.user_profile:
            self.user_profile["short_term"] = {"like": [], "dislike": []}

        # 重複を防ぎながら追加
        for pref in extracted_prefs.get("like", []):
            if pref not in self.user_profile["short_term"]["like"]:
                self.user_profile["short_term"]["like"].append(pref)

        for pref in extracted_prefs.get("dislike", []):
            if pref not in self.user_profile["short_term"]["dislike"]:
                self.user_profile["short_term"]["dislike"].append(pref)

    def update_mismatch_results(self, results):
        """ミスマッチ推論結果を更新するメソッド"""
        self.mismatch_results = results