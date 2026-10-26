import json
from utils.LLM import LLM


class MovieFeatureExtractor:
  """映画の推薦文から、作品ごとの特徴フレーズ（短文リスト）を抽出するクラス"""

  # クラス変数としての設定値
  SYSTEM_ROLE = "あなたは映画の推薦テキストから、作品ごとの重要な特徴や要素を的確に抽出・構造化する専門アシスタントです。"
  MODEL = "gpt-4o-mini-2024-07-18"
  TEMPERATURE = 0.2  # 事実に基づく抽出のため低めの設定
  REASONING_LEVEL = "none"
  

  def __init__(self, llm_client: LLM):
    self.llm_client = llm_client

  def extract(self, recommendation_text: str) -> dict[str, list[str]]:
    """推薦応答テキスト（string型）を受け取り、 {映画タイトル: [特徴フレーズリスト]} の辞書を返す"""
    prompt = self._build_prompt(recommendation_text)

    # LLMの呼び出し（ストリーミングの場合はテキストを結合）
    response = self.llm_client.generate_response(
        role=self.SYSTEM_ROLE,
        prompt=prompt,
        model=self.MODEL,
        temperature=self.TEMPERATURE,
        reasoning_level=self.REASONING_LEVEL
    )

    # JSONパース（余分なコードブロックが含まれる場合のクレンジング処理含む）
    return self._parse_json_response(response)

  def _build_prompt(self, recommendation_text: str) -> str:
    """プロンプトの組み立てを行うプライベートメソッド"""
    prompt = f"""\
## タスク
以下の映画推薦文から、各映画のタイトルと特徴フレーズのリストを抽出してください。

## タスク詳細
1. 推薦文に含まれるすべての「映画タイトル」を正確に特定してください。
2. 各映画について、文章内で説明されている要素（あらすじ、キャラクター、世界観、テーマ、映像美、演出、感情的効果など）を短いフレーズ（名詞句や簡潔な節）に分解・整理してください。
  - 抽出する特徴の粒度はタスク実行例のように短めで抽出してください
3. 出力は必ず以下の純粋なJSON形式のみとしてください。解説や装飾コードブロック（```json など）は含めないでください。

## タスク実行例
- 映画推薦文:
『ヒロイン失格』(2015年)は、高校を舞台にした青春恋愛コメディで、ファッションやビジュアルが重視され、三角関係のドキドキが満載です。
幼なじみの利太への強い想いと学校一のイケメン・弘光との恋の駆け引きが繰り広げられ、キュンキュンする恋愛描写やギャグ要素も豊富。
明るく親しみやすいキャラクターたちと、終盤の美しいラストシーンが印象的で、あなたの好みの青春恋愛コメディの魅力を存分に味わえます。

- 抽出する特徴
・高校が舞台
・青春恋愛コメディ
・ファッション重視
・三角関係
・恋の駆け引き
・キュンキュンする恋愛描写
・ギャグ要素
・美しいラストシーン

- 映画推薦文:
『今夜、世界からこの恋が消えても』(2022年)は、前向性健忘症という切ない病気を抱えたヒロインと、彼女との一日限りの恋を積み重ねる青年の物語です。
緻密な伏線回収があり、涙を誘う感動的なストーリー展開が特徴。透明感ある映像美や花火のシーンなどビジュアルの美しさも際立ち、純粋で切ない恋心の表現があなたの恋愛コメディへの興味を深めるでしょう。

- 抽出する特徴
・病気を抱えたヒロイン
・一日限りの恋
・青春の物語
・緻密な伏線回収
・感動的
・透明感のある映像美
・花火
・春水で切ない恋心

## 出力形式
{{
  "映画タイトル1": [
    "特徴フレーズ1",
    "特徴フレーズ2",
    "..."
  ],
  "映画タイトル2": [
    "特徴フレーズ1",
    "特徴フレーズ2"
  ]
}}

## 制約事項
 ・キャラクター名などの固有名詞や映画のタイトルなどは含めないでください。

## 対象の映画推薦文
{recommendation_text}
"""
    return prompt

  def _parse_json_response(self, response_text: str) -> dict[str, list[str]]:
    """LLMの返答テキストを安全に辞書型へパースするプライベートメソッド"""
    cleaned_text = response_text.strip()

    # Markdownのコードブロック記法がついている場合の除去
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
      # パースに失敗した場合はエラー情報付きでそのまま返すか例外を投げる
      return {"error": [cleaned_text]}