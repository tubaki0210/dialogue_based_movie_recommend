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
・ユーザーの短期的嗜好(好み)または長期的嗜好(好み)の大枠には一致しているが、その具体的内容や解釈がユーザーの期待（長期的嗜好や避ける要素など）とズレている要素。

## タスク
【抽出された共通要素（クラスター情報）】および【直前の推薦内容】の分析を行った上で、、短期的嗜好(好み)または長期的嗜好(好み)の大枠には合致するものの、ユーザーの期待と異なる要素（ミスマッチ）を推論してください。

## タスクを行う際の手順
1. 【クラスターの分析】提示された各クラスターが、それぞれどのような特徴でまとめられているかを分析します。
2. 【ミスマッチだと考えられるクラスターの選択】ユーザーの短期的嗜好(好み)または長期的嗜好(好み)に対してミスマッチであると推測される共通要素集合を最大3つ選びます
    - 短期的嗜好(好み)または長期的嗜好(好み)の両方にミスマッチでなくとも、片方だけでもミスマッチと判断できれば選んでください
3. 【ミスマッチ要素の推論と対照的な要素の推論】 ステップ2で選定した各集合について、ミスマッチの具体的な原因を推論してください。その推論に基づき、ミスマッチ要素と対照的な要素と具体例を推論してください。
    - `mismatch` に指定する要素は、必ず【抽出された共通要素（クラスター情報）】に含まれている特徴をベースに作成してください。
    - 【最重要：対照的要素（positive）の導出規則】
     対照的な要素を考える際は、長期的嗜好にある単語（例：家族愛、癒やし、日常など）をそのまま足し合わせるのではなく、「その大枠ジャンル（例：ファンタジー、アクション等）が本来持っている世界観の中で、どのような別の視点がるか」を推論してください

## タスク実行例1
### 【入力例】
・クラスター情報:
  - クラスター1: ["涙を誘う悲恋", "切ない別れ", "儚く残る余韻"]
  - クラスター2: ["派手なアクション", "カーチェイス", "銃撃アクション"]
・短期的嗜好（好み）: 感動
・短期的嗜好（苦手/嫌い）: 特になし
・長期的嗜好（好み）: コメディ、SF
・長期的嗜好（苦手/嫌い）: ホラー、恋愛

### 【推論例】
1. 【クラスター分析】
   - クラスター1: 切なさや悲恋を中心とした感動要素の集まり。
   - クラスター2: アクション要素の集まり。
2. 【ミスマッチと考えられるクラスターの選択】
   - クラスター1（切ない感動）は、短期的嗜好（好み）の「感動」の大枠には合致するが、感動は「切なさ」だけでなく、「明るい感動」もあるため期待と異なるミスマッチの可能性がある
   - クラスター2はアクションについての集合であり、短期的嗜好（好み）の感動とは関係ないため、ミスマッチではない
3. 【ミスマッチ要素の推論と対照的な要素の推論】
   - クラスター1はミスマッチの可能性があり、それは「切ない感動」である
   - 対照的な要素は、感動という大枠を維持したままミスマッチを要素である「切なく感動」と異なる視点である「明るい感動」である。

## タスク実行例2
### 【入力例】
・クラスター情報:
  - クラスター1: ["高度な心理戦", "犯人との頭脳戦", "複雑なトリック"]
  - クラスター2: ["日常の温かさ", "ほのぼのとした会話", "心温まる関係性"]
・短期的嗜好（好み）: テンポの早いストーリー
・短期的嗜好（苦手/嫌い）: 特になし
・長期的嗜好（好み）: サスペンス、ミステリー
・長期的嗜好（苦手/嫌い）: 特になし

### 【推論例】
1. 【クラスター分析】
   - クラスター1: 緻密なロジックや頭脳戦を中心としたサスペンス・ミステリー要素の集まり。
   - クラスター2: 平穏で心温まる日常的な人間関係の集まり。
2. 【ミスマッチと考えられるクラスターの選択】
   - クラスター1（頭脳戦・トリック）は、長期的嗜好（好み）の「サスペンス」の大枠には合致するが、サスペンスには「複雑な頭脳戦」だけでなく「警察組織の裏側を描く社会派ドラマ」や「スピード感ある逃走劇」などの異なる視点・テイストもあるため、ユーザーの今の期待とはズレているミスマッチの可能性がある。
   - クラスター2は日常描写についての集合であり、短期的嗜好・長期的嗜好の「サスペンス」「テンポの早さ」とは関係がないため、ミスマッチではない。
3. 【ミスマッチ要素の推論と対照的な要素の推論】
   - クラスター1はミスマッチの可能性があり、それは「複雑な論理展開や頭脳戦に特化したサスペンス」である。
   - 対照的な要素は、サスペンスという大枠を維持したまま「複雑な頭脳戦」とは異なる視点である「テンポ良く展開するスリリングなサスペンス」や「人間の業を描く社会的サスペンス」である。

## 制約事項
・ユーザーの好む要素(短期的or長期的)と重複したミスマッチ要素を推論しないでください
・大枠のジャンルに、無関係な別ジャンルや要素を安易に合体させないでください。**
  - ✕ 悪い例: 「盗賊の冒険ファンタジー」 ➔ 「家族愛と笑いのあるホームドラマ風ファンタジー」（外から家族愛を足している）
  - 〇 良い例: 「戦闘や緊迫感を中心とした重厚なファンタジー」 ➔ 「世界の美しさや魔法のワクワク感を中心とした明るく幻想的なファンタジー」（ファンタジーという枠の中での別の視点）
・映画のタイトルは含めないでください。
・対照的な要素を考える際、大枠のジャンルを変更しないでください（例：感動ドラマからアクションへ変更するのは禁止）。
・ミスマッチ要素と対照的な要素はできるだけ簡潔に長くなりすぎないようにしてください。
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
    "cluster" : 選んだクラスターを記述,
    "cluster_feature" : "クラスターの特徴",
    "reason" : そのクラスターがミスマッチだと考えられる選んだ理由を記述,
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