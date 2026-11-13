from typing import Any, Dict
from utils.LLM import LLM
from .BaseQuestionGenerator import BaseQuestionGenerator
from .QuestionContext import QuestionContext

class FallbackQuestionGenerator(BaseQuestionGenerator):
  """ミスマッチがなかった場合に、嗜好と遠いクラスタを利用して質問を生成するクラス"""

  SYSTEM_ROLE = "あなたは対話型映画推薦システムにおいて、ユーザーの意図や好みを丁寧に聞き出す対話専門のアシスタントです。"
  # MODEL = "gpt-4.1-mini-2025-04-14"
  MODEL = "gpt-6-luna"
  TEMPERATURE = 0.5
  REASONING_LEVEL = "medium"

  def __init__(self, llm_client : LLM):
    self.llm_client = llm_client

  def generate(self, context: QuestionContext) -> Dict[str, Any]:
    """contextから嗜好情報とミスマッチ推論結果を取り出して質問文を生成する"""
    prompt = self._build_prompt(context)
    # LLMの呼び出し（ストリーミングレスポンスの集約）
    response = self.llm_client.generate_response_stream(
        role=self.SYSTEM_ROLE,
        prompt=prompt,
        model=self.MODEL,
        temperature=self.TEMPERATURE,
        reasoning_level=self.REASONING_LEVEL
    )
    return response

  def _build_prompt(self, context: QuestionContext):
    short_term_like_str = self._format_preference(context.short_term_like)
    short_term_dislike_str = self._format_preference(context.short_term_dislike)
    long_term_dislike_str = self._format_preference(context.long_term_dislike)
    
    # ----------------------------------------------------
    # long_term_dislike が空かどうかを判定
    # ----------------------------------------------------
    is_long_term_dislike_empty = not context.long_term_dislike or long_term_dislike_str.strip() in ["", "なし", "None", "[]"]

    if is_long_term_dislike_empty:
        # ==========================================
        # パターンA: 長期的な苦手データがない場合
        # ==========================================
        task_details = """
        ・選択肢(ヒント)を3～4つ提示して、「あなたの現在の好みはどれに近いですか？」のように聞いてください。
        """
        
        user_preference_section = f"""
        ## ユーザーの嗜好情報
        短期的嗜好(好み): {short_term_like_str}
        短期的嗜好(苦手): {short_term_dislike_str}
        """
    else:
        # ==========================================
        # パターンB: 長期的な苦手データがある場合（元のバージョン）
        # ==========================================
        task_details = """
        ・選択肢(ヒント)を3～4つ提示して、「あなたの現在の好みはどれに近いですか？」のように聞いてください。
        ・長期的嗜好(苦手)に含まれる内容を選択肢に含めないようにしてください。
        """
        
        user_preference_section = f"""
        ## ユーザーの嗜好情報
        短期的嗜好(好み): {short_term_like_str}
        短期的嗜好(苦手): {short_term_dislike_str}
        長期的嗜好(苦手): {long_term_dislike_str}
        """

    # ----------------------------------------------------
    # プロンプトの構築
    # ----------------------------------------------------
    prompt = f"""## タスク
与えられた嗜好情報を基に、ユーザーの現在の短期的嗜好(好み)を具体化してください。

## タスク詳細
{task_details.strip()}

## 制約事項
・「短期的嗜好によると」、「長期的嗜好によると」などの表現は不必要です。
・現在の嗜好の具体化なので、嗜好に関係ない要素を付け加えないでください。

{user_preference_section.strip()}
"""

    return prompt

  def _format_preference(self, pref):
    if isinstance(pref, list):
      return ", ".join(pref) if pref else ""
    return pref if pref else ""