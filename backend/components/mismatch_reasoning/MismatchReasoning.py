import json
from typing import Any, Dict, List, Union
from utils.LLM import LLM


class MismatchReasoning:
  """共通要素とユーザーの嗜好情報から、解釈のズレ（ミスマッチ）と対照的な要素を推論するクラス"""

  # クラス変数
  SYSTEM_ROLE = "あなたは対話型映画推薦システムにおいて、ユーザーの嗜好と映画特徴の『解釈のズレ（ミスマッチ）』を精密に推論・分析する専門アナリストです。"
  # MODEL = "gpt-4.1-mini-2025-04-14"
  MODEL = "gpt-6-luna"

  TEMPERATURE = 0.3  # 推論の論理的整合性を保つため低めに設定

  def __init__(self, llm_client: LLM):
    self.llm_client = llm_client

  def infer_mismatch(
      self,
      common_features: List[List[str]],
      short_term_like: Union[List[str], str],
      short_term_dislike: Union[List[str], str],
      long_term_like: Union[List[str], str],
      long_term_dislike: Union[List[str], str],
      recent_recommend_res
  ) -> List[Dict[str, Any]]:
    """共通要素の二次元リストとユーザー嗜好（短期・長期の好み/嫌い）を受け取り、

    ミスマッチ要素、対照的要素、その具体例（最大3つ）のリスト（JSON構造）を返す。
    """
    # 引数がリストの場合は文字列に整形
    st_like_str = self._format_preference(short_term_like)
    st_dislike_str = self._format_preference(short_term_dislike)
    lt_like_str = self._format_preference(long_term_like)
    lt_dislike_str = self._format_preference(long_term_dislike)

    prompt = self._build_prompt(
        common_features, 
        st_like_str, 
        st_dislike_str, 
        lt_like_str, 
        lt_dislike_str,
        recent_recommend_res
    )

    # LLMの呼び出し（ストリーミング結果を集約）
    response = self.llm_client.generate_response(
        role=self.SYSTEM_ROLE,
        prompt=prompt,
        model=self.MODEL,
        temperature=self.TEMPERATURE,
    )
    print(response)
    if response == "not_mismatch":
      return { "status" : "not_mismatch" }
    
    # JSON形式にパースして返す
    # return json.loads(response)
    return { "status" : "mismatch", "content" : self._parse_json_response(response) }

  def _build_prompt(
      self,
      common_features: List[List[str]],
      short_term_like: str,
      short_term_dislike: str,
      long_term_like: str,
      long_term_dislike: str,
      recent_recommend_res
  ) -> str:
    """プロンプトの組み立てを行うプライベートメソッド"""
    # 二次元リストを読みやすい文字列に整形
    formatted_clusters = ""
    for idx, cluster in enumerate(common_features, 1):
      items_str = " / ".join([f"「{item}」" for item in cluster])
      formatted_clusters += f"・クラスター{idx}: {items_str}\n"

    prompt = f"""\
## ミスマッチの定義
・ユーザーの好みの大枠（短期的嗜好）には一致しているが、その具体的内容や解釈がユーザーの期待（長期的嗜好や避ける要素など）とズレている要素。

## タスク
提示された複数のクラスター（共通要素群）の分析を行った上で、短期的嗜好（好み）の大枠には合致するものの、ユーザーの期待と異なる要素（ミスマッチ）を特定し、**「大枠のテーマを保持したまま」**ユーザーが真に求めている対照的なテイストと具体例を推論してください。

## タスクを行う際の手順
1. 【クラスターの分析】提示された各クラスターが、それぞれどのような特徴でまとめられているかを分析します。
2. 【大枠の選定】ユーザーの短期的嗜好（好み）の「大枠（ジャンルやテーマ）」に合致するクラスターを特定します。
3. 【ミスマッチの判定】 
   ・【重要】`mismatch` に指定する要素は、必ず【抽出された共通要素（クラスター情報）】に含まれている特徴やトーンをベースに作成してください。**
   ・クラスター情報の中に「長期的嗜好（苦手）」と直接一致するものがない場合は、クラスター情報の中から「長期的嗜好（好み）」の期待（例: 温かい雰囲気など）とズレているトーンをミスマッチとして特定してください。

## タスク実行例
### 【入力例】
・クラスター情報:
  - クラスター1: ["涙を誘う悲恋", "切ない別れ", "儚く残る余韻"]
  - クラスター2: ["日常の温かさ", "仲間との絆", "成長と達成感"]
・短期的嗜好（好み）: 感動できる映画
・短期的嗜好（苦手/嫌い）: 特になし
・長期的嗜好（好み）: 明るい感動、前向きになれる展開
・長期的嗜好（苦手/嫌い）: 切ない感動、後味の悪い悲劇

### 【推論プロセスと出力例】
1. 【クラスター分析】
   - クラスター1: 切なさや悲恋を中心としたしんみりする感動要素の集まり。
   - クラスター2: 成長や絆を中心とした温かく前向きな要素の集まり。
2. 【大枠の選定】
   - 短期的嗜好（好み）は「感動できる映画」であり、クラスター1・2ともに感動という大枠に合致。
3. 【ミスマッチの判定】
   - クラスター1（切ない感動）は、「感動」の大枠には合致するが、長期的嗜好（苦手）にある「切ない感動」と直接一致するため、期待と異なるミスマッチ要素として選定。
4. 【対照的要素の導出】
   - 「感動」という大枠を維持したまま、長期的嗜好（好み）にある「明るい感動」に沿った対照的なテイスト（明るく爽やかな感動）を推論。

[
  {{
    "mismatch": "切なく悲しい感動",
    "positive": "明るく前向きになれる感動",
    "positive_example": [
      "困難を乗り越えて絆が深まる展開",
      "努力が報われて勇気をもらえるハッピーな結末",
      "爽快感と温かさがあるストーリー"
    ]
  }}
]

## 制約事項
・映画のタイトルは含めないでください。
・対照的な要素を考える際、大枠のジャンルを変更しないでください（例：感動ドラマからアクションへ変更するのは禁止）。
・ミスマッチ要素と対照的な要素はできるだけ簡潔にしてください。
・ユーザーの「長期的嗜好（苦手）」に記載されている文字列をそのまま `mismatch` の値として出力することは厳禁です。** 必ず【抽出された共通要素】に存在するクラスター表現を元に記述してください。

## 【抽出された共通要素（クラスター情報）】
{formatted_clusters}

## ユーザーの嗜好情報
・短期的嗜好（好み）: {short_term_like}
・短期的嗜好（苦手/嫌い）: {short_term_dislike}
・長期的嗜好（好み）: {long_term_like}
・長期的嗜好（苦手/嫌い）: {long_term_dislike}

## クラスターの元となった直前の推薦内容
{recent_recommend_res}

## 出力フォーマット
出力は必ず以下の純粋なJSON配列形式（コードブロック記法を含まない）で出力してください。
`positive_example` は最大3つまでの文字列リストとしてください。
また、ミスマッチがない場合は「not_mismatch」とだけ出力してください

[
  {{
    "mismatch": "ミスマッチ要素",
    "positive": "ミスマッチと対照的な要素",
    "positive_example": [
      "対照的要素の具体例1",
      "対照的要素の具体例2",
      "対照的要素の具体例3"
    ]
  }}
]
"""
    return prompt

  def _format_preference(self, pref: Union[List[str], str]) -> str:
    """嗜好データをプロンプト用の文字列に整える"""
    if isinstance(pref, list):
      return "、".join(pref) if pref else "特になし"
    return pref if pref else "特になし"

  def _parse_json_response(self, response_text: str) -> List[Dict[str, Any]]:
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
          "mismatch": "解析エラー",
          "positive": "解析エラー",
          "positive_example": [cleaned_text],
      }]