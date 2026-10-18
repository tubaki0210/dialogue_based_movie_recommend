import json, re
from utils.LLM import LLM
class ExtractUserPreferences:
    SYSTEM_ROLE = "あなたは対話型映画推薦システムにおいて、対話履歴と現在のユーザーの嗜好情報をもとにユーザーの今の好み/避ける要素を考えるアシスタントです"
    TEMPERATURE = 0.3
    # MODEL = "gpt-4.1-mini-2025-04-14"
    MODEL = "gpt-6-luna"

    def __init__(self, llm_client : LLM):
        self.llm_client = llm_client

    def extract_user_preferences(self, current_turn_input_str, dialogue, current_short_term_like_list, current_short_term_dislike_list):
        prompt = self._build_extract_prompt(current_turn_input_str, dialogue, current_short_term_like_list, current_short_term_dislike_list)
        # print(prompt)
        response = self.llm_client.generate_response(self.SYSTEM_ROLE, prompt, self.MODEL, self.TEMPERATURE)
        print(response)
        return json.loads(response)
        return self._parse_json_response(response)

    def _build_extract_prompt(self, current_turn_input_str, dialogue, current_short_term_like_list, current_short_term_dislike_list):
        taskDefinition = (
        "あなたのタスクは、「対話履歴」と「新たなユーザーの発話」と「映画に対する現在のユーザの好む要素と好まない要素」を基に、ユーザの映画に対する好む要素と好まない要素を推測することです。\n"
        "- 対話履歴からユーザーの好む要素と好まない要素が、より**具体的になったことが確認できる場合は必ず情報を更新してください。**\n"
        "- 新たなユーザの発話の中で**言及されていな要素を勝手に好みと好まない要素として捉えないでください。**\n"
        "- 映画のタイトルは含めないでください。\n"
        "出力形式：\n"
        "必ず下記の**JSON形式**で出力してください。その他のテキストは一切含めないでください。\n"
        # "```json\n"
        "{\n"
        '  "like": ["キーフレーズ１", "キーフレーズ２", ...],\n'
        '  "dislike": ["キーフレーズ３", "キーフレーズ４", ...]\n'
        "}\n"
        "出力例：\n"
        "{\n"
        '   "like": ["派手なアクション", "銃撃戦", "キャラクターの成長"],\n'
        '   "dislike: ["感動", "コメディ"]\n'
        "}\n"
        # "```\n"
    )
        thinkingHints = (
        "1. フレーズの塊: 関連性の高い嗜好要素（例:「スリリングなアクション」）は分割せず、1つの嗜好フレーズとして抽出してください。独立した要素（例:「不思議な世界観」と「感動する」）は分けてください。\n"
        "2. 各要素は短く簡潔にしてください。"
        "3. 具体化: 既存の嗜好が最新の対話で具体的になった（例:「感動する」から「温かさや希望を感じる感動」へ）場合は、より具体的な表現に**上書き**してください。\n"
        "4. 「やっぱり銃撃戦よりカーチェイス」などと変更があった場合には、該当要素を正確に書き換えてください。"
        "5. 曖昧な表現の除外: 「面白い」映画といった表現は嗜好として抽出しないでください。\n"
        "6. 「映画」や「要素」という不要な言葉はつけないでください。例えば、「派手なアクション映画が観たい」なら「派手なアクション」と判断してください。「感動要素」なら「感動」と判断してください。"
    )
        current_short_term_like_str = "、".join(current_short_term_like_list)
        current_short_term_dislike_str = "、".join(current_short_term_dislike_list)
        format_dialogue = [f"[{utt['role']}]: {utt['content']}" for utt in dialogue]
        join_dialogue = "\n".join(format_dialogue)
        prompt = (
                f"## タスク\n{taskDefinition}\n\n"
                f"## タスク詳細\n{thinkingHints}\n\n"
                "## 制約事項：出力に「映画」は付けないでください。\n"
                "--- \n"
                "以下の情報に基づいて、タスクを実行してください。\n\n"
                f"対話履歴：{join_dialogue}\n" 
                # f"現在のターンのユーザ発話：{current_turn_input_str}\n"
                f"ユーザが好む要素：{current_short_term_like_str}\n"
                f"ユーザが避ける要素：{current_short_term_dislike_str}\n"
                )
        return prompt


    def _parse_json_response(self, response_text: str):
        """LLMの返答テキストを安全にパースする"""
        cleaned_text = response_text.strip()

        # コードブロック（```json ... ```）のクレンジング
        if cleaned_text.startswith("```"):
            lines = cleaned_text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned_text = "\n".join(lines).strip()

        try:
            return json.loads(cleaned_text)
        except json.JSONDecodeError:
            return [{
            "like": "解析エラー",
            "dislike": "解析エラー"
        }]