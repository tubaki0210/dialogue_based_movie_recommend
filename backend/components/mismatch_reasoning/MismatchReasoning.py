import json
from typing import Any, Dict, List, Union
from utils.LLM import LLM
from utils.Embedding import Embeddings
from utils.supabase_function import extract_movie_features_by_ids
import numpy as np

class MismatchReasoning:
  """共通要素とユーザーの嗜好情報から、解釈のズレ（ミスマッチ）と対照的な要素を推論するクラス"""

  # クラス変数
  SYSTEM_ROLE = "あなたは対話型映画推薦システムにおいて、ユーザーの嗜好と映画特徴の『解釈のズレ（ミスマッチ）』を精密に推論・分析する専門アナリストです。"
  MODEL = "gpt-6-luna"
  REASONING_LEVEL = "medium"
  THEREHOLD = 0.65

  TEMPERATURE = 0.3  # 推論の論理的整合性を保つため低めに設定

  def __init__(self, llm_client: LLM, embedding_client: Embeddings, feature_embedding_dict):
    self.llm_client = llm_client
    self.embedding_client = embedding_client
    self.feature_embedding_dict = feature_embedding_dict

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

  # ミスマッチ推論の具体例と、視聴履歴内の評価値４以上の映画の特徴を結び付ける
  def mismatch_reasoning_matched_watched_record_feature(
        self,
        watch_record,
        recent_movies_features_data,
        short_term_like,
        short_term_dislike,
        long_term_like,
        long_term_dislike,
        feature_hints_text,
    ):
        # 1. 従来のミスマッチ推論を実行
        mismatch_result = self.infer_mismatch_v2(
            recent_movies_features_data,
            short_term_like,
            short_term_dislike,
            long_term_like,
            long_term_dislike,
            feature_hints_text,
        )
        
        if mismatch_result.get("status") == "not_mismatch":
            return { "status": "not_mismatch" }

        mismatch_content = mismatch_result.get("content", [])
        print(mismatch_content)
        # ----------------------------------------------------
        # ★追加: すべての positive_example を集めて一括でベクトル化する
        # ----------------------------------------------------
        all_pos_elems = []
        for item in mismatch_content:
            for pos_elem in item.get("positive_example", []):
                if pos_elem and pos_elem not in all_pos_elems:
                    all_pos_elems.append(pos_elem)

        pos_embedding_dict = {}
        if all_pos_elems:
            # ご提示いただいた一括取得関数を使用
            vectors = self.embedding_client.get_embedding_byList(all_pos_elems)
            for elem, vec in zip(all_pos_elems, vectors):
                pos_embedding_dict[elem] = vec
        # ----------------------------------------------------

        # 2. watch_record から int型の movie_ids リストを作成
        watch_movie_ids = []
        for item in watch_record:
            try:
                m_id = int(item.get("movie_id"))
                watch_movie_ids.append(m_id)
            except (ValueError, TypeError):
                continue

        # 3. Supabase から watch_record の映画の features を一括取得し、辞書からベクトルを引く
        watch_feature_vectors_flat = []
        watch_feature_titles_flat = []

        if watch_movie_ids:
            res = extract_movie_features_by_ids(watch_movie_ids)
            for row in res.data or []:
                title = row.get("title")
                features = row.get("features") or []
                if isinstance(features, list):
                    for f in features:
                        f_text = str(f).strip()
                        if f_text and f_text in self.feature_embedding_dict:
                            vec = self.feature_embedding_dict[f_text]
                            watch_feature_vectors_flat.append(np.array(vec, dtype=np.float32))
                            watch_feature_titles_flat.append(title)

        # 視聴履歴のベクトル行列 (W, dim)
        if watch_feature_vectors_flat:
            W_matrix = np.vstack(watch_feature_vectors_flat).astype(np.float32)
            W_norm = W_matrix / (np.linalg.norm(W_matrix, axis=1, keepdims=True) + 1e-9)
        else:
            W_norm = None

        # 4. positive_example の各要素と視聴履歴特徴の類似度計算・マッピング
        for item in mismatch_content:
            positive_examples = item.get("positive_example", [])
            mapped_examples = []

            for pos_elem in positive_examples:
                best_title = None
                max_sim = 0.0

                # ★ 事前取得した pos_embedding_dict からベクトルを取得して判定
                if W_norm is not None and len(W_norm) > 0 and pos_elem in pos_embedding_dict:
                    pos_vec = pos_embedding_dict[pos_elem]
                    p_arr = np.array(pos_vec, dtype=np.float32)
                    p_norm = p_arr / (np.linalg.norm(p_arr) + 1e-9)

                    # コサイン類似度計算 (W_norm と p_norm のドット積)
                    sims = np.dot(W_norm, p_norm)
                    best_idx = np.argmax(sims)
                    max_sim = float(sims[best_idx])
                    
                    # 類似度が閾値以上の場合のみ映画タイトルを紐付ける
                    if max_sim >= self.THEREHOLD:
                        best_title = watch_feature_titles_flat[best_idx]

                # 指定されたフォーマットに格納
                mapped_examples.append({
                    "element": pos_elem,
                    "title": best_title  # 閾値未満、または該当なしの場合は None
                })

            # positive_example を置き換え
            item["positive_example"] = mapped_examples

        # 5. ステータスを含めた辞書形式で返却
        return {
            "status": "mismatch",
            "content": mismatch_content
        }   


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
    # 引数がリストの場合は文字列に整形
    st_like_str = self._format_preference(short_term_like)
    st_dislike_str = self._format_preference(short_term_dislike)
    lt_like_str = self._format_preference(long_term_like)
    lt_dislike_str = self._format_preference(long_term_dislike)
    
    prompt = self._build_prompt_v2(
      recent_movies_features_data,
      st_like_str,
      st_dislike_str,
      lt_like_str,
      lt_dislike_str,
      feature_hints_text
    )
        # LLMの呼び出し（ストリーミング結果を集約）
    # print(prompt)
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
    
    mismatch_definition = "ユーザーの短期的嗜好(好み)または長期的嗜好(好み)の大枠には一致しているが、その具体的内容や解釈がユーザーの期待とズレている要素。"
    preference_section = f"""\
  ## ユーザーの嗜好情報
  ・短期的嗜好（好み）: {short_term_like}
  ・短期的嗜好（苦手/嫌い）: {short_term_dislike}
  ・長期的嗜好（好み）: {long_term_like}
  ・長期的嗜好（苦手/嫌い）: {long_term_dislike}"""
        
    long_term_constraint_note = "・ユーザーの「長期的嗜好（苦手）」に記載されている文字列をそのまま `mismatch` の値として出力することは厳禁です。"
    # long_term が空かどうかを判定（整形済み文字列が空、または "なし" や "None" の場合を考慮）
    is_long_term_empty = not long_term_like or long_term_like.strip() in ["", "なし", "None", "[]", "特になし"]

    if is_long_term_empty:
        # ==========================================
        # パターンA: 短期的嗜好のみのバージョン
        # ==========================================
        mismatch_definition = "ユーザーの短期的嗜好（好み）の大枠には一致しているが、その具体的内容や解釈がユーザーの期待とズレている要素。"
        preference_section = f"""\
        ## ユーザーの嗜好情報
        ・短期的嗜好（好み）: {short_term_like}
        ・短期的嗜好（苦手/嫌い）: {short_term_dislike}
        """
        long_term_constraint_note = ""
    print(preference_section)
    # 共通のプロンプトテンプレートに切り替えた変数を埋め込む
    prompt = f"""\
  ## ミスマッチの定義
    ・{mismatch_definition}

  ## タスク概要
  ユーザーは直前に推薦された3作品に対して「微妙・何か違う」という否定的なフィードバックを示しています。
  提示された【直前推薦3作品の生の特徴フレーズ】と【ユーザーの嗜好情報】を分析し、なぜユーザーが微妙だと感じたのか、どの特徴のニュアンスや方向性がユーザーの期待とズレていたのか（ミスマッチ要素）を推論してください。
  一つの観点だけでなく、複数の観点から考えても構いません。

  ## 推論の前提条件
  1. ユーザーは「微妙」と言っているため、3作品の特徴の中には嗜好にマッチしているが、その具体的内容がユーザーの期待と異なるミスマッチ要素が存在している可能性があります。
  2. 「嗜好には合致しているが、その具体的内容がユーザーの期待と異なるのではないか？」という視点で特徴フレーズ（mismatch）を特定してください。

  ## タスク手順
  1. 【3作品の特徴分析】直前推薦3作品の特徴フレーズを観察し、作品間に共通する傾向や、特定の作品で目立つ特徴を整理します。
  2. 【ミスマッチ原因の特定】ステップ1で整理した特徴とユーザーの嗜好を照らし合わせ、不満の原因を特定します。（`mismatch`）。
     - ミスマッチ要素を特定する際は、必ず「ミスマッチの定義」に従ってください。
     - 【重要】「ドラマが混在している」「笑いの要素が多い」といった、他ジャンルや別要素の混入をミスマッチの原因にすることは禁止です。** あくまで「アクション」や「感動」といった嗜好の内部において、「どんな具体的な手法・スタイル・演出（例：銃撃戦、肉弾戦、カーチェイスなど）に偏っていたか」という**同一カテゴリ内の具体性のズレ**として特定してください
     - `mismatch` は**余計な説明文（「〜に寄り〜な方向性」など）をつけず、名詞句（20文字以内）で記述してください。
     - 短期的嗜好とのミスマッチを優先的に考えてください。短期的嗜好とのミスマッチがない場合に、長期的嗜好とのミスマッチを考えてください。
  3. 【方向性の異なる要素の選定】ミスマッチ要素は避けて、`target_preference`を維持したまま、そのジャンルや世界観特有のどのような別方向の切り口・アプローチにシフトすべきかを決定します（`positive`）。
     - `positive` は、「絆」や「人間関係」「仲間との信頼」といった、どんなジャンルにも当てはまる抽象的な言葉にするのではなく、ミスマッチを特定した嗜好における具体的な要素やギミックの方向性で表現してください。
     - `positive` は、「〜な演出」「〜する展開」といった長文にせず、どのような方向性にシフトするのかを示す端的な名詞句（20文字以内）で表現してください。
     - 【参考特徴リスト】はあくまで参考ですので、そのまま抽出する必要はありません。抽象的な特徴であれば必ず具体化してユーザーが答えやすいようにしてください。
  4. 【方向性の異なる要素の具体例の推論】ステップ3で定めた方向性を踏まえ、【ヒント】を参考にしながら、より作品の魅力やシチュエーションがイメージしやすい具体的な特徴タグを推論します（`positive_example`）。
     - 【ヒント】はあくまで参考ですので、そのまま抽出する必要はありません。抽象的な特徴であれば必ず具体化してユーザーが答えやすいようにしてください。
     - 「感動的なストーリー展開」や「ラストに感動」といった、ヒントにある**抽象的な言葉**をそのまま使うことは禁止です。そのヒントが示す内容は具体的にどのような場面・構造・シチュエーションなのか（例：「親子の別れ」「魔法による戦闘」「カンフーアクション」「自己犠牲のドラマ」など）を考えて表現してください。
     - `positive_example` は、その具体的な方向性を表す短い特徴タグ（20文字以内・体言止め）にしてください。
     - `positive_example`は最大4個考えてください。


  ## ミスマッチ推論の例1
  ・ユーザー嗜好: [短期的嗜好: SF・冒険]
  ・直前3作品の特徴: ["タイムトラベルの謎解き", "歴史改変のサスペンス", "複雑な因果律"]
  ・推論:
    - mismatch: 「時間改変や因果律を読み解く頭脳戦的なSF要素」
    - reason: ユーザーは「SF・冒険」を求めているが、「複雑な頭脳戦・謎解き」に偏っており、これがユーザーの期待するワクワクする冒険体験とズレて「微妙」と感じさせた可能性が高い。
    - positive: 「未知の領域の探索」
    - positive_example: ["巨大宇宙船での探索", "未知の惑星との遭遇", "サイバーパンクな都市描写"]

  ## ミスマッチ推論の例2（ヒントの具体的落とし込み例）
  ・ユーザー嗜好: [短期的嗜好: 感動]
  ・参考ヒント情報: [「ラストに感動」、「感動的なストーリー展開」、「衝撃の運命」、「泣ける展開」]
  ・直前3作品の特徴: ["コミカルな掛け合い", "日常のドタバタ劇", "ライトな笑いの要素"]
  ・推論:
    - mismatch: 「笑いや日常のドタバタに偏り、感情を揺さぶるドラマ性が弱い要素」
    - reason: ユーザーは「感動」を求めているが、笑いの要素が強すぎて物足りなさを感じさせた可能性が高い。
    - positive: 「運命の過酷さとそれを乗り越える」
    - positive_example: ["親子の離別と再会", "大切な人のための自己犠牲", "隠された過去の真実の判明"]

  ## 制約事項
  ・ユーザーの好む要素と重複したミスマッチ要素を推論しないでください。
  ・「アクションの中にドラマや笑いがある」「感動の中にコメディがある」といった、異なるジャンルや要素の混同をミスマッチの理由にしないでください。ユーザーの嗜好として指定されたジャンル（例：アクション）の“内部”における、具体的な演出手法やスタイルの偏り（例：大味な銃撃戦への偏り、派手なカーチェイスの欠如など）を推論してください
  ・`target_preference`は必ず、「ユーザーの嗜好情報」からそのまま抽出してください。
  ・`positive_example` は絶対に「〜する物語」「〜なファンタジー」といった説明文・あらすじ文章にしないでください。
  ・必ず「体言止め（名詞句）」の短くシャープな特徴タグ（15文字以内）として出力してください。
    - ✕ 悪い例: 「魔法の世界に包まれた心温まる物語」（要素が合体した説明文）
    - ✕ 悪い例: 「異世界の不思議さをじっくり味わえる」（述語になっている）
    - 〇 良い例: 「魔法世界の冒険」
    - 〇 良い例: 「未知の惑星での遭遇」
    - 〇 良い例: 「巨大建造物の探索」
    - 〇 良い例: 「特殊能力によるアクション」
  ・映画のタイトルは含めないでください。
  ・ミスマッチ要素とは異なる方向性を持つ要素を考える際、大枠のジャンルを変更しないでください（例：感動ドラマからアクションへ変更するのは禁止）。
  {long_term_constraint_note}

  ## 【直前推薦3作品の生の特徴フレーズ】
  {recent_movies_features_text}

  ## 【positiveとpositve_example導出用のヒント】
  {feature_hints_text}

{preference_section}

  ## 出力フォーマット
  必ず以下のJSON配列形式で出力してください。真に根拠がなくミスマッチが推論できない場合のみ "not_mismatch" と出力してください。

  [
    {{
      "target_preference": "ミスマッチ要素の予測元となった嗜好",
      "mismatch_target_features": ["直前3作品から特定した特徴フレーズ"],
      "reason": "なぜこの特徴がユーザーにとってミスマッチ（微妙の理由）だと考えられるかの考察",
      "mismatch": "ミスマッチ要素",
      "positive": "嗜好の大枠を維持したままミスマッチと方向性を変えた要素",
      "positive_example": [
        "具体例1",
        "具体例2",
        "具体例3",
        "具体例4",
      ]
    }}
  ]
  """
    return prompt



  def mismatch_reasoning_matched_watched_record_feature_no_hint(
        self,
        watch_record,
        recent_movies_features_data,
        short_term_like,
        short_term_dislike,
        long_term_like,
        long_term_dislike,
    ):
        # 1. 従来のミスマッチ推論を実行
        mismatch_result = self.infer_mismatch_v2_no_hint(
            recent_movies_features_data,
            short_term_like,
            short_term_dislike,
            long_term_like,
            long_term_dislike,
        )
        
        if mismatch_result.get("status") == "not_mismatch":
            return { "status": "not_mismatch" }

        mismatch_content = mismatch_result.get("content", [])

        # ----------------------------------------------------
        # ★追加: すべての positive_example を集めて一括でベクトル化する
        # ----------------------------------------------------
        all_pos_elems = []
        for item in mismatch_content:
            for pos_elem in item.get("positive_example", []):
                if pos_elem and pos_elem not in all_pos_elems:
                    all_pos_elems.append(pos_elem)

        pos_embedding_dict = {}
        if all_pos_elems:
            # ご提示いただいた一括取得関数を使用
            vectors = self.embedding_client.get_embedding_byList(all_pos_elems)
            for elem, vec in zip(all_pos_elems, vectors):
                pos_embedding_dict[elem] = vec
        # ----------------------------------------------------

        # 2. watch_record から int型の movie_ids リストを作成
        watch_movie_ids = []
        for item in watch_record:
            try:
                m_id = int(item.get("movie_id"))
                watch_movie_ids.append(m_id)
            except (ValueError, TypeError):
                continue

        # 3. Supabase から watch_record の映画の features を一括取得し、辞書からベクトルを引く
        watch_feature_vectors_flat = []
        watch_feature_titles_flat = []

        if watch_movie_ids:
            res = extract_movie_features_by_ids(watch_movie_ids)
            for row in res.data or []:
                title = row.get("title")
                features = row.get("features") or []
                if isinstance(features, list):
                    for f in features:
                        f_text = str(f).strip()
                        if f_text and f_text in self.feature_embedding_dict:
                            vec = self.feature_embedding_dict[f_text]
                            watch_feature_vectors_flat.append(np.array(vec, dtype=np.float32))
                            watch_feature_titles_flat.append(title)

        # 視聴履歴のベクトル行列 (W, dim)
        if watch_feature_vectors_flat:
            W_matrix = np.vstack(watch_feature_vectors_flat).astype(np.float32)
            W_norm = W_matrix / (np.linalg.norm(W_matrix, axis=1, keepdims=True) + 1e-9)
        else:
            W_norm = None

        # 4. positive_example の各要素と視聴履歴特徴の類似度計算・マッピング
        for item in mismatch_content:
            positive_examples = item.get("positive_example", [])
            mapped_examples = []

            for pos_elem in positive_examples:
                best_title = None
                max_sim = 0.0

                # ★ 事前取得した pos_embedding_dict からベクトルを取得して判定
                if W_norm is not None and len(W_norm) > 0 and pos_elem in pos_embedding_dict:
                    pos_vec = pos_embedding_dict[pos_elem]
                    p_arr = np.array(pos_vec, dtype=np.float32)
                    p_norm = p_arr / (np.linalg.norm(p_arr) + 1e-9)

                    # コサイン類似度計算 (W_norm と p_norm のドット積)
                    sims = np.dot(W_norm, p_norm)
                    best_idx = np.argmax(sims)
                    max_sim = float(sims[best_idx])
                    
                    # 類似度が閾値以上の場合のみ映画タイトルを紐付ける
                    if max_sim >= self.THEREHOLD:
                        best_title = watch_feature_titles_flat[best_idx]

                # 指定されたフォーマットに格納
                mapped_examples.append({
                    "element": pos_elem,
                    "title": best_title  # 閾値未満、または該当なしの場合は None
                })

            # positive_example を置き換え
            item["positive_example"] = mapped_examples

        # 5. ステータスを含めた辞書形式で返却
        return {
            "status": "mismatch",
            "content": mismatch_content
        }   



  
  def infer_mismatch_v2_no_hint(
        self,
        recent_movies_features_data,
        short_term_like,
        short_term_dislike,
        long_term_like,
        long_term_dislike,
    ):
        """ミスマッチのヒント（17作品の参考特徴リスト）を使用しない推論メソッド"""
        # 引数がリストの場合は文字列に整形
        st_like_str = self._format_preference(short_term_like)
        st_dislike_str = self._format_preference(short_term_dislike)
        lt_like_str = self._format_preference(long_term_like)
        lt_dislike_str = self._format_preference(long_term_dislike)
        
        prompt = self._build_prompt_v2_no_hint(
            recent_movies_features_data,
            st_like_str,
            st_dislike_str,
            lt_like_str,
            lt_dislike_str,
        )
        
        response = self.llm_client.generate_response(
            role=self.SYSTEM_ROLE,
            prompt=prompt,
            model=self.MODEL,
            temperature=self.TEMPERATURE,
            reasoning_level=self.REASONING_LEVEL
        )
        
        if response.strip() == "not_mismatch":
            return { "status" : "not_mismatch" }
        
        return { "status" : "mismatch", "content" : self._parse_json_response(response) }

  def _build_prompt_v2_no_hint(
        self,
        recent_movies_features_data,
        short_term_like,
        short_term_dislike,
        long_term_like,
        long_term_dislike,
    ):
        recent_movies_features_text = self._format_features_list(recent_movies_features_data)
        
        mismatch_definition = "ユーザーの短期的嗜好(好み)または長期的嗜好(好み)の大枠には一致しているが、その具体的内容や解釈がユーザーの期待とズレている要素。"
        preference_section = f"""\
    ## ユーザーの嗜好情報
    ・短期的嗜好（好み）: {short_term_like}
    ・短期的嗜好（苦手/嫌い）: {short_term_dislike}
    ・長期的嗜好（好み）: {long_term_like}
    ・長期的嗜好（苦手/嫌い）: {long_term_dislike}"""
            
        long_term_constraint_note = "・ユーザーの「長期的嗜好（苦手）」に記載されている文字列をそのまま `mismatch` の値として出力することは厳禁です。"
        
        is_long_term_empty = not long_term_like or long_term_like.strip() in ["", "なし", "None", "[]", "特になし"]

        if is_long_term_empty:
            mismatch_definition = "ユーザーの短期的嗜好（好み）の大枠には一致しているが、その具体的内容や解釈がユーザーの期待とズレている要素。"
            preference_section = f"""\
        ## ユーザーの嗜好情報
        ・短期的嗜好（好み）: {short_term_like}
        ・短期的嗜好（苦手/嫌い）: {short_term_dislike}
        """
            long_term_constraint_note = ""

        prompt = f"""\
    ## ミスマッチの定義
        ・{mismatch_definition}

    ## タスク概要
    ユーザーは直前に推薦された3作品に対して「微妙・何か違う」という否定的なフィードバックを示しています。
    提示された【直前推薦3作品の生の特徴フレーズ】と【ユーザーの嗜好情報】を分析し、**なぜユーザーが微妙だと感じたのか、どの特徴のニュアンスや方向性がユーザーの期待とズレていたのか（ミスマッチ要素）**を推論してください。
    一つの観点だけでなく、複数の観点から考えても構いません。

    ## 推論の前提条件（最重要）
    1. ユーザーは「微妙」と言っているため、3作品の特徴の中には嗜好にマッチしているが、その具体的内容がユーザーの期待と異なるミスマッチ要素が存在している可能性があります。
    2. 「嗜好には合致しているが、その具体的内容がユーザーの期待と異なるのではないか？」という視点で特徴フレーズ（mismatch）を特定してください。

    ## タスク手順
    1. 【3作品の特徴分析】直前推薦3作品の特徴フレーズを観察し、作品間に共通する傾向や、特定の作品で目立つ特徴を整理します。
    2. 【ミスマッチ原因の特定】ステップ1で整理した特徴とユーザーの嗜好を照らし合わせ、不満の原因を特定します。（`mismatch`）。
       - ミスマッチ要素を特定する際は、必ず「ミスマッチの定義」に従ってください。
       - `mismatch` は、直前3作品のどのような偏りや特徴が不満の原因になったのかを、**余計な説明文（「〜に寄り〜な方向性」など）をつけず、短くシャープな名詞句（20文字以内）**で記述してください。
    3. 【方向性の異なる要素の選定】ミスマッチ要素は避けて、`target_preference`を維持したまま、そのジャンルや世界観特有の別の具体的な要素・仕掛け・切り口を選定します（`positive`）。
       - `positive` は、「〜な演出」「〜する展開」といった長文にせず、**どのような方向性にシフトするのかを示す端的な名詞句（20文字以内）**で表現してください。
       - `positive_example` は、`positive`を具体的に表す具体例にしてください。

    ## ミスマッチ推論の例1
    ・ユーザー嗜好: [短期的嗜好: SF・冒険]
    ・直前3作品の特徴: ["タイムトラベルの謎解き", "歴史改変のサスペンス", "複雑な因果律"]
    ・推論:
      - mismatch: 「時間改変や因果律を読み解く頭脳戦的なSF要素」
      - reason: ユーザーは「SF・冒険」を求めているが、「複雑な頭脳戦・謎解き」に偏っており、これがユーザーの期待するワクワクする冒険体験とズレて「微妙」と感じさせた可能性が高い。
      - positive: 「未知の領域の探索やスケールの大きなダイナミックなSF体験」
      - positive_example: ["巨大宇宙船での探索", "未知の惑星との遭遇", "サイバーパンクな都市描写"]

    ## 制約事項
    ・ユーザーの好む要素と重複したミスマッチ要素を推論しないでください。
    ・`target_preference`は必ず、ユーザーの嗜好情報からそのまま抽出してください。
    ・`positive_example` は絶対に「〜する物語」「〜なファンタジー」といった説明文・あらすじ文章にしないでください。
    ・**必ず「体言止め（名詞句）」の短くシャープな特徴タグ（15文字以内）として出力してください。**
      - ✕ 悪い例: 「魔法の世界に包まれた心温まる物語」（要素が合体した説明文）
      - ✕ 悪い例: 「異世界の不思議さをじっくり味わえる」（述語になっている）
      - 〇 良い例: 「幻想的な世界観」
      - 〇 良い例: 「魔法世界の冒険」
      - 〇 良い例: 「未知の惑星での遭遇」
      - 〇 良い例: 「巨大建造物の探索」
      - 〇 良い例: 「特殊能力によるアクション」
    ・「長期的嗜好」にある単語を安易にジャンルへ合体させないでください。そのジャンルの世界観の中でどのように表現されるかの「視点・切り口」として記述してください。
    ・映画のタイトルは含めないでください。
    ・嗜好の大枠には合致するがミスマッチ要素とは異なる方向性を持つ要素を考える際、大枠のジャンルを変更しないでください（例：感動ドラマからアクションへ変更するのは禁止）。
    {long_term_constraint_note}

    ## 【直前推薦3作品の生の特徴フレーズ】
    {recent_movies_features_text}

    {preference_section}

    ## 出力フォーマット
    必ず以下のJSON配列形式で出力してください。真に根拠がなくミスマッチが推論できない場合のみ "not_mismatch" と出力してください。

    [
      {{
        "target_preference": "ミスマッチ要素の予測元となった嗜好",
        "mismatch_target_features": ["直前3作品から特定した特徴フレーズ"],
        "reason": "なぜこの特徴がユーザーにとってミスマッチ（微妙の理由）だと考えられるかの考察",
        "mismatch": "ミスマッチ要素",
        "positive": "嗜好の大枠を維持したままミスマッチと方向性を変えた要素",
        "positive_example": [
          "具体例1",
          "具体例2",
          "具体例3"
        ]
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
      return ", ".join(pref) if pref else ""
    return pref if pref else ""

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