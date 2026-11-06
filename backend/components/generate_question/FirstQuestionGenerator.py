from .BaseQuestionGenerator import BaseQuestionGenerator
from utils.LLM import LLM
class FirstQuestionGenerator(BaseQuestionGenerator):
    SYSTEM_ROLE = "あなたは対話型映画推薦システムにおいて、ユーザーの好みを引き出すための質問をするアシスタントです。"
    TEMPERATURE = 1.0
    MODEL = "gpt-6-luna"
    REASONING_LEVEL = "low"



    def __init__(self, llm_client : LLM):
        self.llm_client = llm_client

    def generate(self, context):
        long_like_preferences_str = ", ".join(context.long_term_like)
        long_dislike_preferences_str = ", ".join(context.long_term_dislike)
        prompt = self._build_prompt(long_like_preferences_str, long_dislike_preferences_str)
        response = self.llm_client.generate_response_stream(self.SYSTEM_ROLE, prompt, self.MODEL, self.TEMPERATURE, self.REASONING_LEVEL)
        return response

    def _build_prompt(
            self, 
            long_like_preferences_str, 
            long_dislike_preferences_str
            ):
        prompt = f"""\
            ## タスク
            ユーザーの長期的嗜好（過去の好み）を理解していることを示しつつ、今日の気分に寄り添い、新しい発見につながるような選択肢を含めた最初の質問文（会話の導入）を生成してください。

            ## タスクを行う際の手順
            1. ユーザーの「普段から好む要素」と「普段から避ける要素」を確認します。
            2. 【理解と共感の導入】
             - 普段の好む要素に触れ、「〇〇な作品をお楽しみいただくことが多いですね」のように、ユーザーの好みを理解していることが伝わるソフトな問いかけを作成します。
             - この時、すべての普段から好む要素について言及せずに、2つまでにしてください。
            3. 【パーソナライズされた選択肢の提示】
            今日の気分を選びやすくするため、以下の2つの視点から簡潔な選択肢を提示してください。

            - 【いつものお好みに近い方向性】
                「好む要素」から、固有名詞などを含む特定の要素以外の、わかりやすい要素を2〜3個取り出し、それぞれのスタイルを表す簡潔なフレーズ（20文字以内）を2~3個提示してください。
                この時、取り出した要素をそのまま提示するのはやめてください。必ず汎用的なフレーズにしてください
                ※ ✕ 悪い例: 「テンポの良い掛け合いや意外性のある展開が楽しい、笑いどころ満載のコメディ」（長文）
                ※ 〇 良い例: 「テンポの良い掛け合いで笑えるコメディ」

            - 【いつもと少し視点を変えた方向性】
                「避ける要素」を避けつつ、好む要素とは異なるジャンルやテーマを2個程度提示してください。

            4. 【自由入力の促し】
            最後は「もちろんこれ以外でも構いませんので、今どんな映画が観たい気分か自由に教えてください！」と締めくくってください。

            ## タスク実行例
            【入力嗜好】
            ・普段から好む要素: アクション、コメディ、家族の絆
            ・普段から避ける要素: 暗く重い展開、ホラー

            【出力例】
            普段は派手なアクションや笑えるコメディなどをよくご覧になっていますね。今日はどんな映画をお探しですか？

            たとえば、このような方向性はいかがでしょうか。

            【いつものお好みに近い方向性】
            ・ハイスピードなカーチェイスや派手な銃撃戦
            ・テンポの良い掛け合いやシュールな笑い
            ・不器用な親子の絆や成長

            【いつもと少し視点を変えた方向性】
            ・未知の世界を旅するファンタジー
            ・チームで困難を乗り越えて勝利を目指す
            ・鮮やかなトリックと軽妙なチームワーク

            もちろんこれら以外でも構いませんので、今観たい映画のイメージや気分を自由に教えてください！

            ## 制約事項
            - 「普段から好む要素」の深掘りでは、複数のジャンルや要素を1つの選択肢に混ぜ合わせないでください。
            - 「普段から避ける要素」は提案や例えに一切含めないでください。
            - 「避ける要素」そのものを言葉に出して言及すること（例：「暗い作品は除いて〜」等）も禁止します。
            - 出力はユーザーへの問いかけ文章のみとし、解説や思考プロセスなどの余計なテキストは含めないでください。

            ## ユーザーの嗜好情報
            ・普段から好む要素: {long_like_preferences_str}
            ・普段から避ける要素: {long_dislike_preferences_str}
"""
        return prompt