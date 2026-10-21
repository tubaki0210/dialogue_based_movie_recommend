from typing import Any, Dict
import time
from utils.LLM import LLM
from .BaseQuestionGenerator import BaseQuestionGenerator
from .QuestionContext import QuestionContext

class UnMismatchQuestionGenerator(BaseQuestionGenerator):
  """ミスマッチがなかった場合に、嗜好と遠いクラスタを利用して質問を生成するクラス"""

  SYSTEM_ROLE = "あなたは対話型映画推薦システムにおいて、ユーザーの意図や好みを丁寧に聞き出す対話専門のアシスタントです。"
  # MODEL = "gpt-4.1-mini-2025-04-14"
  MODEL = "gpt-6-luna"
  TEMPERATURE = 0.5

  def __init__(self, llm_client : LLM):
    self.llm_client = llm_client

  def generate(self, context: QuestionContext) -> Dict[str, Any]:
    """contextから嗜好情報とミスマッチ推論結果を取り出して質問文を生成する"""
    prompt = self._build_prompt(context)
    print(prompt)
    # LLMの呼び出し（ストリーミングレスポンスの集約）
    response = self.llm_client.generate_response_stream(
        role=self.SYSTEM_ROLE,
        prompt=prompt,
        model=self.MODEL,
        temperature=self.TEMPERATURE,
    )
    return response


  def generate_unmismatch_questions(self, context: QuestionContext):
      """mismatch_resultを基に、ルールベースでユーザへの質問文を生成する"""
      mismatch_result = context.un_mismatch_results

      # 1. 導入文（最初のミスマッチ要素を軽く引用して自然な形にする例）
      intro = "なるほど、もしかしたら推薦した映画は少しお好みとは違ったかもしれません。"

      # 2. 各観点に対する質問文の生成（リスト形式）
      question_list = []
      for item in mismatch_result:
        mismatch_elem = item.get("mismatch", "")
        positive_elem = item.get("positive", "")
        positive_examples = item.get("positive_example", [])

        # 「〇〇という要素はお好みではないですか？それより△△という方が好きですか？」
        question = (
            f"・{mismatch_elem}はお好みではないですか？それより"
            f"{positive_elem}の方が好みですか？"
        )
        # 具体例の追加
        if positive_examples:
          question += "\n 例えば、"
          for pos_example in positive_examples:
            question += f"\n ・{pos_example}"
        question += "\n"
        question_list.append(question)

      # 3. 締め文
      closing = "具体的にどんなストーリーや要素が好きか、自由に教えてください！"

      # すべてを結合して出力形式に合わせる
      output_parts = [intro, ""]
      output_parts.extend(question_list)
      output_parts.append("")
      output_parts.append(closing)
      full_text = "\n".join(output_parts)
      yield full_text


  def _build_prompt(self, context: QuestionContext) -> str:
    """画像仕様に基づいたプロンプト作成"""
    # 嗜好情報の整形
    # positive_preferences = f"長期的嗜好(好み): {', '.join(context.long_term_like)}\n長期的嗜好(苦手):{', '.join(context.long_term_dislike)}"

    # ミスマッチ情報の整形
    unliked_element_str = self._format_mismatch_result(context.un_mismatch_results)

    prompt = f"""\
## タスク
与えられた情報を基に、それぞれの好まなかったと思われる観点からユーザの嗜好を引き出すための質問を生成してください。

## タスク詳細
・「〇〇という要素はお好みではないですか？それより△△という方が好きですか？」のテンプレートを使用してください。
  - 〇〇: 好まないと思われる要素
  - △△: 対照的な要素
・対照的な要素の具体例は、質問文の後に選択肢やイメージとして「（例えば、「～」や「～」など）」のように続けてください。

## 出力形式
1. 最初に、ユーザのフィードバックを受け止めて、「なるほど、〜〜かもしれません。」のように導入文を生成してください。
2. 次に、各観点に対してテンプレに従った質問文を生成し、リスト形式で出力してください。
3. 最後に、ユーザの回答を促す締め文を生成してください。（例: 「具体的にどんな〜〜が好きか教えていただけますか？」）

## ミスマッチ要素を推論した情報
{unliked_element_str}

"""

    return prompt


  def _format_unliked_result(self, mismatch_result):
    result = []
    for i, un_mismatch_ele in enumerate(mismatch_result):
      unliked_element = un_mismatch_ele.get("unliked_element")
      positive = un_mismatch_ele.get("positive")
      positive_example = un_mismatch_ele.get("positive_example")
      positive_example_str = "、".join(positive_example)
      unliked_ele_str = f"{i + 1}:\n好まないと思われる要素:{unliked_element}\n対照的な要素:{positive}\n対照的な要素の具体例:{positive_example_str}\n"
      result.append(unliked_ele_str)

    return "\n".join(result)