from typing import Any, Dict
from utils.LLM import LLM
from .BaseQuestionGenerator import BaseQuestionGenerator
from .QuestionContext import QuestionContext

class MismatchQuestionGenerator(BaseQuestionGenerator):
  """ミスマッチ推論結果を基に、具体的にユーザーの嗜好を引き出す質問を生成するクラス"""

  SYSTEM_ROLE = "あなたは対話型映画推薦システムにおいて、ユーザーの意図や好みを丁寧に聞き出す対話専門のアシスタントです。"
  # MODEL = "gpt-4.1-mini-2025-04-14"
  MODEL = "gpt-6-luna"
  TEMPERATURE = 0.5

  def __init__(self, llm_client : LLM):
    self.llm_client = llm_client

  def generate(self, context: QuestionContext) -> Dict[str, Any]:
    """contextから嗜好情報とミスマッチ推論結果を取り出して質問文を生成する"""
    prompt = self._build_prompt(context)
    # print(prompt)
    # LLMの呼び出し（ストリーミングレスポンスの集約）
    response = self.llm_client.generate_response_stream(
        role=self.SYSTEM_ROLE,
        prompt=prompt,
        model=self.MODEL,
        temperature=self.TEMPERATURE,
    )
    return response

  def _build_prompt(self, context: QuestionContext) -> str:
    """画像仕様に基づいたプロンプト作成"""
    # 嗜好情報の整形
    positive_preferences = f"短期的嗜好: {', '.join(context.short_term_like)}\n長期的嗜好: {', '.join(context.long_term_like)}"

    # ミスマッチ情報の整形
    mismatch_str = self._format_mismatch_result(context.mismatch_results)

    prompt = f"""\
## タスク
与えられた情報を基に、それぞれのミスマッチの観点からユーザの嗜好を引き出すための質問を生成してください。

## タスク詳細
・「〇〇という要素はお好みではないですか？それより△△という方が好きですか？」のテンプレートを使用してください。
  - 〇〇: ミスマッチ要素
  - △△: 対照的な要素
・対照的な要素の具体例は、質問文の後に選択肢やイメージとして「（例えば、「～」や「～」など）」のように続けてください。

## 出力形式
1. 最初に、ユーザのフィードバックを受け止めて、「なるほど、〜〜かもしれません。」のように導入文を生成してください。
2. 次に、各観点に対してテンプレに従った質問文を生成し、リスト形式で出力してください。
3. 最後に、ユーザの回答を促す締め文を生成してください。（例: 「具体的にどんな〜〜が好きか教えていただけますか？」）

## ミスマッチ要素を推論した情報
{mismatch_str}

## ユーザのポジティブ嗜好
{positive_preferences}

"""
    return prompt

  def _format_mismatch_result(self, mismatch_result):
    result = []
    for i, mismatch_ele in enumerate(mismatch_result):
      mismatch_element = mismatch_ele.get("mismatch")
      positive = mismatch_ele.get("positive")
      positive_example = mismatch_ele.get("positive_example")
      positive_example_str = "、".join(positive_example)
      mismatch_ele_str = f"ミスマッチ{i + 1}:\nミスマッチ要素:{mismatch_element}\n対照的な要素:{positive}\n対照的な要素の具体例:{positive_example_str}\n"
      result.append(mismatch_ele_str)

    return "\n".join(result)