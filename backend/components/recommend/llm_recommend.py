import sys
import os
from glob import glob
import json
# このファイルのディレクトリの絶対パスを取得
current_dir = os.path.dirname(os.path.abspath(__file__))
# 親ディレクトリのパスを取得
parent_dir = os.path.dirname(current_dir)
# sys.pathに親ディレクトリを追加
sys.path.append(parent_dir)
from utils.LLM import LLM

init = "あなたはユーザと対話を行い、ユーザの好みを引き出し、ユーザに合った映画を推薦するアシスタントです。"
class LLMReranker:
    SYSTEM_ROLE = "あなたは、ユーザーの嗜好に合った映画を推薦するアシスタントです。"
    TEMPERATURE = 0.7
    MODEL = "gpt-6-luna"
    REASONING_LEVEL = "low"

    def __init__(self, movie_metadata, llm_client : LLM):
        self.movie_metadata = movie_metadata
        self.llm_client = llm_client

    def rerank(
            self, 
            stage1_candidates, 
            short_term_like_str, 
            short_term_dislike_str, 
            long_term_like_str, 
            long_term_dislike_str, 
            top_num
            ):
        
        prompt = self._build_recommend_prompt(
            stage1_candidates, 
            short_term_like_str, 
            short_term_dislike_str, 
            long_term_like_str, 
            long_term_dislike_str, 
            top_num
            )
        
        response = self.llm_client.generate_response_stream(
            self.SYSTEM_ROLE, 
            prompt, 
            self.MODEL, 
            self.TEMPERATURE,
            self.REASONING_LEVEL
            )
        
        return response

    def _build_recommend_prompt(self, stage1_candidates, short_term_like_str, short_term_dislike_str, long_term_like_str, long_term_dislike_str, top_num):
        task_definition = (f"""
            ユーザーの短期的嗜好（今日の気分）と長期的嗜好（普段の好み）、および提供された映画の候補情報から、ユーザーが最も好むと思われる順番で{top_num}つの映画を厳選して推薦してください。
            必ず、提供された映画情報（あらすじ・特徴フレーズ）のみを用いてタスクを実行してください。

            【評価・選定のルール（内部プロセス）】
            以下の優先順位とルールに従って、候補作品を分析・スコアリングしてください。

            1. 除外ルール（最優先）
            - 短期的嗜好（好まない要素）が映画の特徴に含まれている場合、その作品は絶対に推薦候補から除外してください。

            2. 加点・ペナルティのルール
            - 短期的嗜好（好む要素）に合致する映画は、最も高く評価（大幅加点）してください。
            - 長期的嗜好（好む要素）に合致する要素も加点し、複数の要素を満たすほど高評価とします。
            - 長期的嗜好（好まない要素）が含まれる場合は原則ペナルティ（減点）とします。
            - 【例外】「今日観たいもの（短期的嗜好）」と「普段苦手なもの（長期的嗜好）」が矛盾する場合は、短期的嗜好を最優先し、ペナルティを無効化してください。

            3. 順位決定
            - ルール1と2の分析結果に基づき、最もスコアが高い{top_num}作品を決定してください。

            【推薦文の出力ルール（ユーザーへの応答）】
            選定した映画をユーザーに推薦する文章を作成してください。出力時は以下のルールを厳守してください。

            - 導入文:
            推薦の冒頭に「〇〇という今の気分（好み）に合わせて、こちらの作品を選びました」と自然な導入を入れてください。短期的に避けたい要素が指定されていた場合は「〇〇の要素は避けて選んでいます」と添えてください。
            ※長期的嗜好の嫌う要素については言及不要です。

            - 作品ごとの紹介文の構成とトーン:
            ロボットのような箇条書きや定型文ではなく、映画の魅力が伝わる自然で熱量のある文章で紹介してください。以下の3つの要素を滑らかに織り交ぜて1つの紹介文にしてください。
            各映画の説明は十分な情報量を持った説明をしてください
            1. 世界観と導入: あらすじから読み取れる舞台設定、登場人物、物語の始まり。その際、作品の映画の説明の冒頭「『映画ID』は～」と始めてください。
            2. 嗜好とのマッチ: 短期的嗜好（好む要素）と映画の特徴がどのように合致し、どう楽しめるのかの具体的な説明。
            3. さらなる魅力: ユーザーの指定要件にはなくても、興味を惹きつける映画のその他の特徴（提供情報に基づくもの）。

            - 出力時の禁止事項:
            - 順位付けの表現（「1位」「第1位」など）は使用しないこと。
            - あらすじ情報がない場合でも「あらすじはありませんが」などの謝罪や言い訳は書かないこと。
            - 提供された映画情報（あらすじ・特徴）にない情報を勝手に捏造（ハルシネーション）して語らないこと。
            - 「*」などのマークダウンや余計な記号は一切使用禁止です
        """)

        precautions = ("""
            - あらすじと映画の特徴をベースに推薦する映画を考えてください。例えば、ジャンルが記載されていますが、嗜好がジャンルにマッチするからといって安易にスコアを高くするのはやめてください
            - 推薦する映画IDは『』で囲むようにしてください。
            - 「長期的嗜好である」や「短期的嗜好である」といった表現はつけないでください。
            - 映画の公開年は映画IDの後に(例: 『○○』(2020年)のように含めてください。
            - 対話の中では「ユーザ」ではなく「あなた」と言ってください。
            - あなたが参照できるのは与えられた情報のみです。
        """)

        candidates_movie_informaiton_str = self._bulid_prompt_of_movie_information(stage1_candidates)  
        prompt = (f"""
        ## タスク: {task_definition}

        ## 制約事項: {precautions}

        ## ユーザーの短期的嗜好:
            好む要素: {short_term_like_str}
            嫌う要素: {short_term_dislike_str}

        ## ユーザーの長期的嗜好:
            好む要素：{long_term_like_str}
            嫌う要素: {long_term_dislike_str}

        ## 推薦候補の映画情報:
        {candidates_movie_informaiton_str}
        """)
        return prompt

    def _bulid_prompt_of_movie_information(self, stage1_candidates):
        movie_information_list = []
        for movie in stage1_candidates:
            movie_id = str(movie.get("movie_id"))     
            target_movie_metadata = self.movie_metadata.get(movie_id, None)
            if target_movie_metadata is None:
                continue
            genre = "、".join(target_movie_metadata.get("genre"))
            published_year = target_movie_metadata.get("year")
            synopsis = target_movie_metadata.get("summary")
            features = target_movie_metadata.get("features", [])
            join_features_list = [f"・{feature}" for feature in features]
            join_features_str = "\n".join(join_features_list)
            movie_information_text = f"""
            映画ID: {movie_id}
            ジャンル: {genre}
            公開年: {published_year}
            あらすじ: {synopsis}
            映画の特徴:\n {join_features_str}
            """
            movie_information_list.append(movie_information_text)

        return "\n".join(movie_information_list)