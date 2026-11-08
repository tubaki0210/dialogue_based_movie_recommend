import sys
import os
# このファイルのディレクトリの絶対パスを取得
current_dir = os.path.dirname(os.path.abspath(__file__))
# 親ディレクトリのパスを取得
parent_dir = os.path.dirname(current_dir)
# sys.pathに親ディレクトリを追加
sys.path.append(parent_dir)
from utils.LLM import LLM

init = "あなたはユーザと対話を行い、ユーザの好みを引き出し、ユーザに合った映画を推薦するアシスタントです。"
class BaseLineRecommend:
    SYSTEM_ROLE = "あなたは、ユーザーの嗜好に合った映画を推薦するアシスタントです。"
    TEMPERATURE = 0.7
    MODEL = "gpt-6-luna"
    REASONING_LEVEL = "low"

    def __init__(self, llm_client : LLM):
        self.llm_client = llm_client

    def generate_response(self, current_turn_input_str ,dialogue):
        format_dialogue = [f"[{utt['role']}]: {utt['content']}" for utt in dialogue]
        join_dialogue = "\n".join(format_dialogue)
        prompt = f"""
            あなたのタスクはユーザーとの対話を通して、相手の現在の気分や嗜好を聞き出し、最適な映画を推薦することです。

            ### タスク
            以下の「対話履歴」と「最新のユーザー発話」を確認し、映画を推薦してしてください。

            1. 嗜好のヒアリング（情報が不十分な場合）
            - ユーザーの好きなジャンル、最近観て面白かった作品、今日の気分などの情報が少ない場合は、質問を1〜2個投げかけてください。
            - 一度にたくさんの質問をしてユーザーを疲れさせないよう、自然な会話のキャッチボールを心がけてください。

            2. 映画の推薦（情報が揃った、または推薦を求められた場合）
            - ユーザーの嗜好が十分に把握できたタイミングで、条件に合う映画を3作品提案してください。
            - 推薦する際は、「作品タイトル」「公開年」「簡潔な見どころ」「なぜユーザーにおすすめなのか（ヒアリングした内容との関連性）」を伝えてください。

            3. 注意事項
            - 結末などの重大なネタバレは避けてください。
            - 「*」などの余計な記号は付けないでください。

            ### 対話履歴
            {join_dialogue}

            ### 最新のユーザー発話
            {current_turn_input_str}

        """
        response = self.llm_client.generate_response_stream(self.SYSTEM_ROLE, prompt, self.MODEL, self.TEMPERATURE, self.REASONING_LEVEL)
        return response
