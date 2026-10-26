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
    short_term_like = context.short_term_like
    short_term_dislike = context.short_term_dislike
    long_term_dislike = context.long_term_dislike
    prompt = f"""\
        ## タスク
        与えられた嗜好情報を基に、ユーザーの現在の短期的嗜好(好み)を具体化してください

        ## タスク詳細
        ・選択肢(ヒント)を3～4つ提示して、「あなたの現在の好みはどれに近いですか？」のように聞いてください。
        ・長期的嗜好(苦手)に含まれる内容を選択肢に含めないようにしてください
        ・最後には、「具体的にどこが微妙だったかを自由に教えてください！」と締めくくってください。

        ## 制約事項
        ・「長期的嗜好」によるとなどの表現は不必要です

        ## ユーザーの嗜好情報
        短期的嗜好(好み): {short_term_like}
        短期的嗜好(苦手): {short_term_dislike}
        長期的嗜好(苦手): {long_term_dislike}
    """
    return prompt