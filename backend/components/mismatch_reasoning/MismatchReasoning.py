import json
from typing import Any, Dict, List, Union
from utils.LLM import LLM


class MismatchReasoning:
  """共通要素とユーザーの嗜好情報から、解釈のズレ（ミスマッチ）と対照的な要素を推論するクラス"""

  # クラス変数
  SYSTEM_ROLE = "あなたは対話型映画推薦システムにおいて、ユーザーの嗜好と映画特徴の『解釈のズレ（ミスマッチ）』を精密に推論・分析する専門アナリストです。"
  MODEL = "gpt-6-luna"
  REASONING_LEVEL = "medium"

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
      mismatch_hints,
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
        mismatch_hints,
        recent_recommend_res
    )
    print(prompt)

    # LLMの呼び出し（ストリーミング結果を集約）
    response = self.llm_client.generate_response(
        role=self.SYSTEM_ROLE,
        prompt=prompt,
        model=self.MODEL,
        temperature=self.TEMPERATURE,
        reasoning_level=self.REASONING_LEVEL
    )
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
      mismatch_hints,
      recent_recommend_res
  ) -> str:
    """プロンプトの組み立てを行うプライベートメソッド"""
    # 二次元リストを読みやすい文字列に整形
    formatted_clusters = ""
    for idx, cluster in enumerate(common_features, 1):
      items_str = " / ".join([f"「{item}」" for item in cluster])
      formatted_clusters += f"・クラスター{idx}: {items_str}\n"

    formated_feature_hints = self._format_feature_hints(mismatch_hints)

    prompt = f"""\
    ## ミスマッチの定義
    ・ユーザーの短期的嗜好(好み)または長期的嗜好(好み)の大枠には一致しているが、その具体的内容や解釈がユーザーの期待（長期的嗜好や避ける要素など）とズレている要素。

    ## タスク
    【抽出された共通要素（クラスター情報）】および【直前の推薦内容】の分析を行った上で、短期的嗜好(好み)または長期的嗜好(好み)の大枠には合致するものの、ユーザーの期待と異なる要素（ミスマッチ）を推論してください。
    また、【推薦候補（17作品）に実在する嗜好別特徴リスト】を参考・ガイドラインとして活用し、ミスマッチとは異なる方向性を持つ要素（positive）とその具体例（positive_example）を推論してください。

    ## タスクを行う際の手順
    1. 【クラスターの分析】提示された各クラスターが、それぞれどのような特徴でまとめられているかを分析します。
    2. 【ミスマッチだと考えられるクラスターの選択】ユーザーの短期的嗜好(好み)または長期的嗜好(好み)に対してミスマッチであると推測される共通要素集合を最大3つ選びます。
        - 短期的嗜好(好み)または長期的嗜好(好み)の両方にミスマッチでなくとも、片方だけでもミスマッチと判断できれば選んでください。
    3. 【ミスマッチ要素の推論と異なる方向性を持つ要素の推論】
        - ステップ2で選定した各集合について、ミスマッチの具体的な原因（mismatch）を推論してください。
        - `mismatch` に指定する要素は、原則【抽出された共通要素（クラスター情報）】に含まれている特徴をベースに作成してください。
        - **`positive` および `positive_example` を作成する際は、【推薦候補（17作品）に実在する嗜好別特徴リスト】に存在するフレーズを参考にし、方向性に適した具体例を導出してください。**

    ## 【最重要：「嗜好の大枠には合致するがミスマッチ要素とは異なる方向性を持つ要素」の導出規則】
    ・嗜好の大枠には合致するがミスマッチ要素とは異なる方向性を持つ要素を考える際は、長期的嗜好にある単語（例：家族愛、癒やし、日常など）をそのまま足し合わせるのではなく、「その大枠ジャンル（例：ファンタジー、アクション等）が本来持っている世界観の中で、どのような別の視点があるか」を推論してください。
    ・**`positive_example` には、【推薦候補（17作品）に実在する嗜好別特徴リスト】を参考に、`positive` の方向性を表す具体的なフレーズを設定してください。**

    ## タスク実行例1
    ### 【入力例】
    ・クラスター情報:
      - クラスター1: ["涙を誘う悲恋", "切ない別れ", "儚く残る余韻"]
      - クラスター2: ["派手なアクション", "カーチェイス", "銃撃アクション"]
    ・短期的嗜好（好み）: 感動
    ・短期的嗜好（苦手/嫌い）: 特になし
    ・長期的嗜好（好み）: コメディ、SF
    ・長期的嗜好（苦手/嫌い）: ホラー、恋愛
    ・推薦候補に実在する嗜好別特徴リスト:
      - 感動: ["心温まるストーリー", "ハッピーエンド", "笑いを含む演出", "主人公の成長物語"]

    ### 【推論例】
    1. 【クラスター分析】
      - クラスター1: 切なさや悲恋を中心とした感動要素の集まり。
      - クラスター2: アクション要素の集まり。
    2. 【ミスマッチと考えられるクラスターの選択】
      - クラスター1（切ない感動）は、短期的嗜好（好み）の「感動」の大枠には合致するが、感動は「切なさ」だけでなく「明るい感動」もあるため期待と異なるミスマッチの可能性がある。
      - クラスター2はアクションについての集合であり、短期的嗜好の「感動」とは関係ないため、ミスマッチではない。
    3. 【ミスマッチ要素の推論と嗜好の大枠には合致するがミスマッチ要素とは異なる方向性を持つ要素の推論】
      - ミスマッチ要素（mismatch）: 「切なく悲しい結末や展開に特化した感動」
      - 異なる方向性の要素（positive）: 「悲恋や切なさではなく、心が明るく温かくなる前向きな感動」
      - 具体例（positive_example）: ["心温まるストーリー", "ハッピーエンド", "笑いを含む演出"] (ヒントリストを参考に選出)

    ## 制約事項
    ・ユーザーの好む要素(短期的or長期的)と重複したミスマッチ要素を推論しないでください。
    ・大枠のジャンルに、無関係な別ジャンルや要素を安易に合体させないでください。
      - ✕ 悪い例: 「盗賊の冒険ファンタジー」 ➔ 「家族愛と笑いのあるホームドラマ風ファンタジー」（外から家族愛を足している）
      - 〇 良い例: 「戦闘や緊迫感を中心とした重厚なファンタジー」 ➔ 「世界の美しさや魔法のワクワク感を中心とした明るく幻想的なファンタジー」（ファンタジーという枠の中での別の視点）
    ・映画のタイトルは含めないでください。
    ・嗜好の大枠には合致するがミスマッチ要素とは異なる方向性を持つ要素を考える際、大枠のジャンルを変更しないでください（例：感動ドラマからアクションへ変更するのは禁止）。
    ・ミスマッチ要素と嗜好の大枠には合致するがミスマッチ要素とは異なる方向性を持つ要素はできるだけ簡潔に長くなりすぎないようにしてください。
    ・ユーザーの「長期的嗜好（苦手）」に記載されている文字列をそのまま `mismatch` の値として出力することは厳禁です。必ず【抽出された共通要素】に存在するクラスター表現を元に記述してください。
    ・**`positive_example` を作成する際は、【推薦候補（17作品）に実在する嗜好別特徴リスト】に記載されているフレーズを参考にして、具体的な選択肢にしてください。**

    ## 【抽出された共通要素（クラスター情報）】
    {formatted_clusters}

    ## 【推薦候補（17作品）に実在する嗜好別特徴リスト（positive導出の参考情報）】
    {formated_feature_hints}

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
    また、ミスマッチがない場合は「not_mismatch」とだけ出力してください。

    [
      {{
        "cluster" : 選んだクラスターを記述,
        "cluster_feature" : "クラスターの特徴",
        "reason" : そのクラスターがミスマッチだと考えられる選んだ理由を記述,
        "mismatch": "ミスマッチ要素",
        "positive": "嗜好の大枠には合致するがミスマッチ要素とは異なる方向性を持つ要素",
        "positive_example": [
          "具体的な特徴1",
          "具体的な特徴2",
          "具体的な特徴3"
        ]
      }}
    ]
    """
    return prompt

# ミスマッチ推論v2(共通要素集合なし)
  def infer_mismatch_v2(
    self,
    recent_movies_features_data,
    short_term_like,
    short_term_dislike,
    long_term_like,
    long_term_dislike,
    feature_hints_text,  # 17作品のヒント
):
    prompt = self._build_prompt_v2(
      recent_movies_features_data,
      short_term_like,
      short_term_dislike,
      long_term_like,
      long_term_dislike,
      feature_hints_text
    )
        # LLMの呼び出し（ストリーミング結果を集約）
    print(prompt)
    response = self.llm_client.generate_response(
        role=self.SYSTEM_ROLE,
        prompt=prompt,
        model=self.MODEL,
        temperature=self.TEMPERATURE,
        reasoning_level=self.REASONING_LEVEL
    )
    if response == "not_mismatch":
      return { "status" : "not_mismatch" }
    
    # JSON形式にパースして返す
    # return json.loads(response)
    return { "status" : "mismatch", "content" : self._parse_json_response(response) }

  def _build_prompt_v2(
      self,
      recent_movies_features_data,
      short_term_like,
      short_term_dislike,
      long_term_like,
      long_term_dislike,
      feature_hints_text,  # 17作品のヒント
  ):
    recent_movies_features_text = self._format_features_list(recent_movies_features_data)
    prompt = f"""\
    
  ## ミスマッチの定義
    ・ユーザーの短期的嗜好(好み)または長期的嗜好(好み)の大枠には一致しているが、その具体的内容や解釈がユーザーの期待（長期的嗜好や避ける要素など）とズレている要素。


  ## タスク概要
  ユーザーは直前に推薦された3作品に対して「微妙・何か違う」という否定的なフィードバックを示しています。
  提示された【直前推薦3作品の生の特徴フレーズ】と【ユーザーの嗜好情報】を分析し、**なぜユーザーが微妙だと感じたのか、どの特徴のニュアンスや方向性がユーザーの期待とズレていたのか（ミスマッチ要素）**を推論してください。
  一つの観点だけでなく、複数の観点から考えても構いません。

  ## 推論の前提条件（最重要）
  1. ユーザーは「微妙」と言っているため、3作品の特徴の中には**一見良く見えてもユーザーの今の気分や期待に対して裏目に出ている要素が必ず存在します**。
  2. 単に「苦手なジャンルが含まれていたか」ではなく、**「好みのジャンル（例：感動、アクション）として提示されたが、その解釈やテイスト（例：重すぎる、テンポが遅い、演出が派手すぎる）がズレていたのではないか？」**という視点で特徴フレーズ（mismatch）を特定してください。

  ## タスク手順
  1. 【3作品の特徴分析】直前推薦3作品の特徴フレーズを観察し、作品間に共通する傾向や、特定の作品で目立つ特徴を整理します。
  2. 【ミスマッチ原因の特定】ユーザーの短期的・長期的嗜好と照らし合わせ、不満の原因となっている具体的特徴フレーズを【直前推薦3作品の生の特徴フレーズ】の中から特定します（`mismatch`）。
  3. 【方向性の異なる要素の選定】ミスマッチ要素は避けて、【推薦候補（17作品）の参考特徴リスト】を参考に、嗜好の大枠を維持した別方向の要素（`positive`）と、その具体的な特徴フレーズ（`positive_example`）を選定してください。
    - `positive_example` は長文のあらすじ説明（〜する物語）にするのではなく、**映画の特徴を表す簡潔なフレーズ（20文字前後）**として記述してください。の異なる要素の選定】ミスマッチ要素（例：重苦しい感動）は避けて、【推薦候補（17作品）の参考特徴リスト】を参考にして、ユーザーの嗜好の大枠を維持したまま、ミスマッチ要素とは別の方向性の要素（`positive`）およびその具体例（`positive_example`）を考えてください。

  ## ミスマッチ推論の例
  ・ユーザー嗜好: [短期的嗜好: 感動]
  ・直前3作品の特徴: ["泣ける親子の別れ", "重厚な人間ドラマ", "切ないラスト"]
  ・推論:
    - mismatch: 「重苦しく切ない展開を中心とした泣かせる感動」
    - reason: ユーザーは「感動」を求めているが、「切なさ・重さ」に偏っており、これがユーザーの期待する感動体験とズレて「微妙」と感じさせた可能性が高い。
    - positive: 「切なさや重さではなく、読後に心が明るく前向きになる感動」
    - positive_example: ["主人公が困難を乗り越えて成長, "家族の絆や友情"]

  ## 制約事項
  ・ユーザーの好む要素(短期的or長期的)と重複したミスマッチ要素を推論しないでください。
  ・`positive_example` は絶対に「〜する物語」「〜なファンタジー」といった説明文・あらすじ文章にしないでください。
  ・**必ず「体言止め（名詞句）」の短くシャープな特徴タグ（15文字以内）として出力してください。**
    - ✕ 悪い例: 「魔法の世界に包まれた心温まる物語」（要素が合体した説明文）
    - ✕ 悪い例: 「異世界の不思議さをじっくり味わえる」（述語になっている）
    - 〇 良い例: 「心温まるストーリー」
    - 〇 良い例: 「幻想的な世界観」
    - 〇 良い例: 「魔法世界の冒険」
    - 〇 良い例: 「丁寧な人物描写」
・「長期的嗜好」にある単語（例：家族愛など）を安易にジャンルへ合体させないでください。そのジャンルの世界観の中でどのように表現されるかの「視点・切り口」として記述してください。
  ・映画のタイトルは含めないでください。
  ・嗜好の大枠には合致するがミスマッチ要素とは異なる方向性を持つ要素を考える際、大枠のジャンルを変更しないでください（例：感動ドラマからアクションへ変更するのは禁止）。
  ・ミスマッチ要素と嗜好の大枠には合致するがミスマッチ要素とは異なる方向性を持つ要素はできるだけ簡潔に長くなりすぎないようにしてください。
  ・ユーザーの「長期的嗜好（苦手）」に記載されている文字列をそのまま `mismatch` の値として出力することは厳禁です。必ず【抽出された共通要素】に存在するクラスター表現を元に記述してください。
  ・**`positive_example` を作成する際は、【推薦候補（17作品）に実在する嗜好別特徴リスト】に記載されているフレーズを参考にしてほしいですが、そのフレーズをそのまま使うのではなく、具体的にしてください。**
  


  ## 【直前推薦3作品の生の特徴フレーズ】
  {recent_movies_features_text}

  ## 【推薦候補（17作品）の参考特徴リスト（positive導出用）】
  {feature_hints_text}

  ## ユーザーの嗜好情報
  ・短期的嗜好（好み）: {short_term_like}
  ・短期的嗜好（苦手/嫌い）: {short_term_dislike}
  ・長期的嗜好（好み）: {long_term_like}
  ・長期的嗜好（苦手/嫌い）: {long_term_dislike}

  ## 出力フォーマット
  必ず以下のJSON配列形式で出力してください。真に根拠がなくミスマッチが推論できない場合のみ "not_mismatch" と出力してください。

  [
    {{
      "mismatch_target_features": ["直前3作品から特定した特徴フレーズ"],
      "reason": "なぜこの特徴がユーザーにとってミスマッチ（微妙の理由）だと考えられるかの考察",
      "mismatch": "ミスマッチ要素",
      "positive": "嗜好の大枠を維持したままミスマッチと方向性を変えた要素",
      "positive_example": [
        "具体例1",
        "具体例2",
        "具体例3
      ]
    }},
    {{
      ...
    }}
  ]
  """
    return prompt
    

  def _format_features_list(self, recent_movies_features):
    result = []
    for index, (title, fetures) in enumerate(recent_movies_features.items()):
      feature_str = ",".join(fetures)
      result.append(f"作品{index + 1}: {feature_str}")
    return "\n".join(result)

  def _format_feature_hints(self, hints_dict):
    """辞書形式のヒントをプロンプト埋め込み用テキストに整形"""
    lines = []
    for pref_label, features in hints_dict.items():
      formatted_feats = ", ".join([f'"{f}"' for f in features])
      lines.append(f"・{pref_label}: [{formatted_feats}]")
    return "\n".join(lines)

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