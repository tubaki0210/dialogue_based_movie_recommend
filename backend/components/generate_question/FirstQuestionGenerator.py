from .BaseQuestionGenerator import BaseQuestionGenerator
from utils.LLM import LLM
class FirstQuestionGenerator(BaseQuestionGenerator):
    SYSTEM_ROLE = "あなたは対話型映画推薦システムにおいて、ユーザーの好みを引き出すための質問をするアシスタントです。"
    TEMPERATURE = 1.0
    # MODEL = "gpt-4.1-mini-2025-04-14"
    MODEL = "gpt-6-luna"


    def __init__(self, llm_client : LLM):
        self.llm_client = llm_client

    def generate(self, context):
        long_like_preferences_str = "、".join(context.long_term_like)
        long_dislike_preferences_str = "、".join(context.long_term_dislike)
        prompt = self._build_prompt(long_like_preferences_str, long_dislike_preferences_str)
        # print(prompt)
        response = self.llm_client.generate_response_stream(self.SYSTEM_ROLE, prompt, self.MODEL, self.TEMPERATURE)
        return response

    def _build_prompt(
            self, 
            long_like_preferences_str, 
            long_dislike_preferences_str
            ):
        prompt = (f"""
            ## タスク
            ユーザーが今観たい映画を自由に入力できるよう、考えるヒントを含めた最初の質問を生成してください。

            ## タスク詳細
            ・ユーザーの「好み」や「苦手」を入力時のヒントとして自然に触れてください（「データによると」などの機械的な表現は不可）
            ・ユーザーが回答しやすいように、簡単に嗜好に触れて考えるヒントを提示していください
              - その際は、ユーザーの嗜好情報を参照してほしいのですが、好みを決めつけて見透かした発言にならないようにソフトな応答にしてください
              - ユーザーが新たな発見をできるように、避ける要素に含まれない要素で好む要素とも異なる観点からの質問を「例えば、～」のように含めてください
              - 「〇〇ですよね」のような決めつけた表現はやめてください
              - その際は、ユーザーの嗜好情報を参照して、普段から避ける要素は薦めないようにしてください
            ・最後は、「どんな映画が観たいか自由に教えてください」と締めくくってください

            ## ユーザーの嗜好情報
            - 普段から好む要素（ポジティブ）: {long_like_preferences_str}
            - 普段から避ける要素（ネガティブ）: {long_dislike_preferences_str}

            ## 制約事項
            - 質問文のみを出力してください。解説などは不要です。
            - 200~250文字程度で簡潔に記述してください。
            """)
        prompt = f"""\
        ## タスク
        ユーザーが今観たい映画を自由に入力できるよう、過去の嗜好に配慮しつつ新たな視点のヒントを含めた最初の質問文を生成してください。

        ## タスクを行う際の手順
        1. ユーザーの嗜好情報（好み・苦手）を確認します。
        2. 普段の「好む要素」に触れつつ、決めつけにならないソフトな問いかけの導入文を作成します。
        3. 新たな発見を促すため、「好む要素」の単なる言い換えではなく、全く別の切り口（例：「アクション」、「感動」、「ホラー」、「ファンタジー」、「冒険」といったテーマ）からの選択肢を2~3つ示してください。
        4. 最後は「どんな映画が観たいか自由に教えてください」と締めくくってください。

        ## タスク実行例
        【入力嗜好】
        ・好む要素: テンポの良いアクション、爽快なストーリー
        ・避ける要素: 暗く重い展開、ホラー

        【出力例】
        軽快でスカッとするアクション作品などが好きですかね。今日はどのような作品を観たいですか？、例えば「魔法が登場するファンタジー」や「異世界の冒険」など、どんな映画が観たいか自由に教えてください。

        ## 制約事項
        ・「普段から避ける要素」は提案や例えに含めないでください。
        ・質問文のみを出力してください（解説や余計な文字は不要）。

        ## ユーザーの嗜好情報
        ・普段から好む要素: {long_like_preferences_str}
        ・普段から避ける要素: {long_dislike_preferences_str}
        """
        return prompt