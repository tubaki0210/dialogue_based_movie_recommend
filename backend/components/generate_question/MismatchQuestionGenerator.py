from typing import Any, Dict
import time
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

  def generate_mismatch_questions(self, context: QuestionContext):
      """mismatch_resultを基に、ルールベースでユーザへの質問文を生成する"""
      mismatch_result = context.mismatch_results

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
            f"{mismatch_elem}はお好みではないですか？それより"
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


  def generate_mismatch_questions_with_title(self, context: QuestionContext):
        """mismatch_resultを基に、ルールベースでユーザへの質問文を生成する"""
        mismatch_result = context.mismatch_results

        # 1. 導入文
        intro = "なるほど、もしかしたら推薦した映画は少しお好みとは違ったかもしれません。"

        # 2. 各観点に対する質問文の生成（リスト形式）
        question_list = []
        for item in mismatch_result:
            mismatch_elem = item.get("mismatch", "")
            positive_elem = item.get("positive", "")
            positive_examples = item.get("positive_example", [])

            # 「〇〇はお好みではないですか？それより△△の方が好みですか？」
            question = (
                f"「{mismatch_elem}」はお好みではないですか？ それより"
                f"「{positive_elem}」の方が好みですか？"
            )

            # 具体例の追加（辞書型構造に対応）
            if positive_examples:
                question += "\n例えば、"
                for pos_example in positive_examples:
                    # 辞書から element と title を取得
                    element = pos_example.get("element", "")
                    title = pos_example.get("title")

                    # タイトルが存在する場合は「特徴（例えば『映画名』）」の形式にする
                    if title:
                        question += f"\n・{element}（『{title}』など）"
                    else:
                        question += f"\n・{element}"

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
    positive_preferences = f"短期的嗜好: {', '.join(context.short_term_like)}\n長期的嗜好: {', '.join(context.long_term_like)}"

    # ミスマッチ情報の整形
    mismatch_str = self._format_mismatch_result_with_title(context.mismatch_results)

    prompt = f"""\
## タスク
与えられた情報を基に、それぞれのミスマッチの観点からユーザの嗜好を引き出すための質問を生成してください。

## タスク詳細
・「〇〇」より「△△」の方が今の気分には合っていますか？のテンプレートを使用してください。
  - 〇〇: ミスマッチ要素
  - △△: 別方向の要素
・別方向の要素の具体例は、質問文の後に選択肢やイメージとして、出力例のように見やすいようにリスト形式で「（例えば、「～」や「～」など）」のように続けてください。
・関連映画は別方向の要素の具体例の後に、それぞれ(映画タイトルなど)のように添えてください。関連映画が空の場合は勝手につけないでください。

## 出力形式
1. 最初に、ユーザのフィードバックを受け止めて、「なるほど、〜〜かもしれません。」のように導入文を生成してください。
    - 対象の嗜好にミスマッチがあったことを受け止めてください。
    - 対象の嗜好が長期的嗜好だった場合は、長期的嗜好は相手には明示的に見えていないため、「[短期手嗜好]の中でも、～～だったかもしれません」と補足してください
2. 次に、各観点に対してテンプレに従った質問文を生成し、リスト形式で出力してください。
3. 最後に、ユーザの回答を促す締め文を生成してください。（例: 「具体的にどんな〜〜が好きか教えていただけますか？」）

## 出力例
なるほど、「感動」でも「切なさや悲しさ」はお好みと違ったかもしれません。

「妖精童話的で情緒重視のファンタジー」より「戦闘主導のファンタジー」の方が今の気分には合っていますか？
例えば、
・ファンタジー戦闘シーン（『クレヨンしんちゃん ブリブリ王国の秘宝』など）
・魔法を交えた戦闘
・剣と魔法の対決

具体的にどんなストーリーや要素が好きか、自由に教えてください！

## 制約事項
- 「短期的嗜好である」「長期的嗜好である」という表現は付けないでください。ユーザーは長期的嗜好が何か理解していないため、自然な導入を意識してください。 

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
      mismatch_ele_str = f"ミスマッチ{i + 1}:\nミスマッチ要素:{mismatch_element}\n別方向の要素:{positive}\n別方向の要素の具体例:{positive_example_str}\n"
      result.append(mismatch_ele_str)

    return "\n".join(result)

  def _format_mismatch_result_with_title(self, mismatch_result):
        result = []
        for i, mismatch_ele in enumerate(mismatch_result):
            target_preference = mismatch_ele.get("target_preference", "指定なし")
            mismatch_element = mismatch_ele.get("mismatch", "")
            positive = mismatch_ele.get("positive", "")
            
            # positive_example が辞書型のリストか、単なる文字列のリストかに応じて柔軟に処理
            positive_examples = mismatch_ele.get("positive_example", [])
            formatted_examples = []
            for ex in positive_examples:
                if isinstance(ex, dict):
                    # {'element': '...', 'title': '...'} のような構造の場合
                    elem = ex.get("element", "")
                    title = ex.get("title", "")
                    if title:
                        formatted_examples.append(f"{elem}（関連映画: {title}）")
                    else:
                        formatted_examples.append(elem)
                else:
                    # 単純な文字列の場合
                    formatted_examples.append(str(ex))
            
            positive_example_str = "、".join(formatted_examples)
            
            mismatch_ele_str = (
                f"ミスマッチ{i + 1}:\n"
                f"対象の嗜好:{target_preference}\n"
                f"ミスマッチ要素:{mismatch_element}\n"
                f"別方向の要素:{positive}\n"
                f"別方向の要素の具体例:{positive_example_str}\n"
            )
            result.append(mismatch_ele_str)

        return "\n".join(result)