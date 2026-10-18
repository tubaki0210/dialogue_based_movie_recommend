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
    # MODEL = "gpt-4.1-mini-2025-04-14"
    MODEL = "gpt-6-luna"
    def __init__(self, movie_json_path, llm_client : LLM):
        with open(movie_json_path, "r", encoding='utf-8') as f:
            self.movie_metadata = json.load(f)
            f.close()
            
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
        # print(prompt)
        response = self.llm_client.generate_response_stream(
            self.SYSTEM_ROLE, 
            prompt, 
            self.MODEL, 
            self.TEMPERATURE
            )
        return response

    def _build_recommend_prompt(self, stage1_candidates, short_term_like_str, short_term_dislike_str, long_term_like_str, long_term_dislike_str, top_num):
        task_definition = (f"""
            ユーザの短期的嗜好と長期的嗜好及び、映画の情報から、ユーザが好むと思われる順番で{top_num}つの映画を選んで推薦してください
            必ず、映画情報内の候補の中から推薦してください。
            タスクを行うときは以下の手順に沿ってください:
            1. 各映画について、ユーザーの好む要素との合致度を分析します。
               - その際は、短期的嗜好を重要視してほしいですが、長期的嗜好も考慮してください
            2. 好まない要素に合致する場合は、ペナルティとして映画の評価を下げます。
               - 短期的嗜好の好む要素が、長期的嗜好の嫌う要素に含まれている場合は、短期的嗜好を重視しペナルティをつけないでください。
            3. ステップ1とステップ2での分析に基づき、上位3つの映画を選びます。
               - 順位決定では、合致度の高さを最優先し、ペナルティ要素がないことをその次に重要視してください。
               - 複数の求めている要素に深く合致している映画を優先してください。
            4. その映画をユーザーに推薦してください。
               - ユーザが映画に興味を持てるように、それぞれの映画のあらすじや特徴を利用して、積極的に映画の説明を行ってください。
               - ユーザーの好む要素に合致する映画の特徴は、その映画をユーザーに推薦する理由に直結するため積極的に説明に含めてください。
               - 各映画について、あらすじからわかる、映画の舞台、登場人物の設定、物語の始まりがあれば含めてください。
               - 各映画の説明は、あらすじからわかる情報＋嗜好にマッチしている特徴の説明＋嗜好とは関係ない映画の特徴を複数説明という形にしてください
               - ユーザが求めている要素とは関係がなくても、興味を持たせるために映画の特徴を説明に含めててください。
               - おすすめ順に推薦しますが、「1位」などは不要です。
               - 「*」などの余計な記号はつけないでください。
               - 推薦時に、嫌う要素に関して「〇〇の要素は含まれていません」のような表現は不必要です
        """
        )
        taskHints = ("""
            - ユーザが映画に興味を持てるように、それぞれの映画のあらすじや特徴を利用して、積極的に映画の説明を行ってください。
            - ユーザーの好む要素に合致する映画の特徴は、その映画をユーザーに推薦する理由に直結するため積極的に説明に含めてください。
            - 各映画の説明は読みやすさを考慮しつつ、情報量を持った文章にしてください。
            - 各映画について、あらすじからわかる、映画の舞台、登場人物の設定、物語の始まりがあれば含めてください。
            - ユーザが求めている要素とは関係がなくても、興味を持たせるために映画の特徴を説明に含めててください。
            """
        )
        precautions = ("""
            - 推薦する映画のタイトルは『』で囲むようにしてください。
            - 「長期的嗜好である」や「短期的嗜好である」といった表現はつけないでください。
            - 映画の公開年はタイトルの後に(例: 『○○』(2020年)のように含めてください。
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
        # prompt = (f"""
        #             ## タスク: {task_definition}
        
        #             ## タスク詳細: {taskHints}
        
        #             ## 制約事項: {precautions}
        
        #             ## ユーザーの短期的嗜好:
        #                 好む要素: {short_term_like_str}
        #                 嫌う要素: {short_term_dislike_str}
        
        #             ## 推薦候補の映画情報:
        #             {candidates_movie_informaiton_str}
        #         """)
        return prompt

    def _bulid_prompt_of_movie_information(self, stage1_candidates):
        movie_information_list = []
        for movie in stage1_candidates:
            title = movie.get("title")
            target_movie_metadata = self.movie_metadata.get(title, None)
            if target_movie_metadata is None:
                continue
            # title = target_movie_metadata.get("title")
            genre = target_movie_metadata.get("genre")
            published_year = target_movie_metadata.get("year")
            synopsis = target_movie_metadata.get("summary")
            features = target_movie_metadata.get("features", [])
            join_features_str = [f"・{feature}" for feature in features]
            movie_information_text = f"""
            タイトル: {title}
            ジャンル: {genre}
            公開年: {published_year}
            あらすじ: {synopsis}
            映画の特徴: {"\n".join(join_features_str)}
            """
            movie_information_list.append(movie_information_text)

        return "\n".join(movie_information_list)