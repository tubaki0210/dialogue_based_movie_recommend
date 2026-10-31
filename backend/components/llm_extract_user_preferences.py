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
        current_short_term_like_str = (
            "、".join(current_short_term_like_list)
            if current_short_term_like_list
            else "なし"
        )
        current_short_term_dislike_str = (
            "、".join(current_short_term_dislike_list)
            if current_short_term_dislike_list
            else "なし"
        )
        format_dialogue = [f"[{utt['role']}]: {utt['content']}" for utt in dialogue]
        join_dialogue = "\n".join(format_dialogue)

        prompt = f"""# タスク
        あなたは映画推薦システムのユーザー嗜好抽出エンジンです。
        「対話履歴」「最新の発話」「現在の嗜好リスト」を分析し、ユーザーの最新の「好む要素(like)」と「好まない要素(dislike)」を抽出・更新してJSON形式で出力してください。

        # 抽出ルール
        1. **ユーザー発話のみを対象にする**
        - [system]の発話から嗜好を抽出してはいけません。
        - ユーザーが発言していない要素を推測で補わないでください。

        2. **フレーズの「分割」と「統合」の判断基準（最重要）**
        - **【統合する（1つのフレーズにする）ケース】**
            - 「修飾語 ＋ 要素」の関係であり、要素の性質や特徴を表している場合。
            - 例: 「スリリングなアクション」「近未来のSF」「ダークな世界観」「重厚なミステリー」
        - **【分割する（別々のフレーズにする）ケース】**
            - 「ジャンル/感情/トーン」など、独立した異なる概念が組み合わさっている場合。
            - 例: 「感動するアクション」 → 「感動」「アクション」に分割
            - 例: 「笑えるホラー」 → 「コメディ」「ホラー」に分割
            - 例: 「恋愛とサスペンス」 → 「恋愛」「サスペンス」に分割

        3. **具体化と更新（上書き）**
        - 既存の嗜好より具体的な発言があった場合は、具体的な表現に上書き更新してください。
            - 例: 「アクション」がある状態で「派手なカーチェイスが好き」と言った場合 → 「派手なカーチェイス」に更新
        - 前言撤回や変更があった場合は、該当項目を書き換えてください。
            - 例: 「やっぱりホラーは苦手」と言った場合 → likeから「ホラー」を削り、dislikeに追加

        4. **不要な語句の除外**
        - 「映画」「要素」「作品」などの言葉は除去してください。（例: 「アクション映画」→「アクション」）
        - 「面白い」「おすすめ」などの抽象的で意味を持たない評価表現は除外してください。
        - 映画の固有タイトルは含めないでください。

        # 分割・統合の具体例
        - 入力: 「スリリングなアクションが見たい」
        - 判定: 「スリリング」はアクションの修飾語
        - like: ["スリリングなアクション"]
        - 入力: 「感動できるアクション映画が良いな」
        - 判定: 「感動（感情）」と「アクション（ジャンル）」は独立した概念
        - like: ["感動", "アクション"]
        - 入力: 「怖いのは嫌だけど、ゾンビが出てくるパニック系は好き」
        - like: ["ゾンビ", "パニック"]
        - dislike: ["ホラー"]

        # 出力フォーマット
        余計な解説や挨拶、Markdown記法（```json など）は一切含めず、純粋なJSON文字列のみを出力してください。

        {{
        "like": ["要素1", "要素2"],
        "dislike": ["要素3", "要素4"]
        }}

        # 入力データ
        ---
        [対話履歴]
        {join_dialogue}

        [最新のユーザー発話]
        {current_turn_input_str}

        [現在の好む要素(like)]
        {current_short_term_like_str}

        [現在の避ける要素(dislike)]
        {current_short_term_dislike_str}
        ---
        """
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