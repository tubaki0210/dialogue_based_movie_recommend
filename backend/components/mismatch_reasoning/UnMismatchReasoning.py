import json
from typing import Any, Dict, List, Union
from utils.LLM import LLM


class UnMismatchReasoning:
    """ユーザーの嗜好と離れているクラスターから、好まなかったと思われる要素と対照的な要素を推論するクラス"""

    # クラス変数
    SYSTEM_ROLE = "あなたは対話型映画推薦システムにおいて、ユーザーが好まなかったと思われる要素を精密に推論・分析する専門アナリストです。"
    # MODEL = "gpt-4.1-mini-2025-04-14"
    MODEL = "gpt-6-luna"

    TEMPERATURE = 0.3  # 推論の論理的整合性を保つため低めに設定

    def __init__(self, llm_client: LLM):
        self.llm_client = llm_client

    def _format_list(self, items: Union[List[str], None]) -> str:
        """リストデータをプロンプト埋め込み用に整形する"""
        if not items:
            return "特になし"
        return "、".join([str(item) for item in items if item])

    def _format_clusters(self, clusters: List[List[str]]) -> str:
        """2次元リストのクラスター情報をフォーマットする"""
        if not clusters:
            return "なし"

        formatted_lines = []
        for i, cluster in enumerate(clusters, 1):
            cluster_str = "、".join([f"'{item}'" for item in cluster])
            formatted_lines.append(f"  - クラスター{i}: [{cluster_str}]")
        return "\n".join(formatted_lines)

    def _build_prompt(
        self,
        un_sim_cluster: List[List[str]],
        short_term_like: List[str],
        short_term_dislike: List[str],
        long_term_like: List[str],
        long_term_dislike: List[str],
    ) -> str:
        """入力データを整形し、ミスマッチ推論用のプロンプトを作成する"""
        formatted_clusters_str = self._format_clusters(un_sim_cluster)
        short_like_str = self._format_list(short_term_like)
        short_dislike_str = self._format_list(short_term_dislike)
        long_like_str = self._format_list(long_term_like)
        long_dislike_str = self._format_list(long_term_dislike)

        return f"""\
## タスク
提示された「ユーザーの嗜好と離れているクラスター群」を分析し、ユーザーが今回の提案で【好まなかったと思われる要素（不適合要素）】を特定した上で、その対照的な要素（ユーザーが本当に求めている要素）および具体的な例を推論してください。

## タスクを行う際の手順
1. クラスターの共通点分析】 提示された各クラスターが、それぞれどのような特徴（ジャンル、トーン、展開、演出など）でまとめられているかを分析します。
2. 【好まなかった要素の推論】
   ・分析したクラスターの中から、ユーザーが好まなかったと思われる要素を選定します。
   ・【最重要】選定する際は「長期的嗜好（苦手/嫌い）」を強い手掛かりとしてください。長期的嗜好（苦手/嫌い）に合致する要素や、長期的嗜好（好み）の期待と直接衝突しているクラスター表現があれば、それを「好まなかったと思われる要素」として捉えます。
3. 【対照的要素とその具体例の推論】
   ・次に好まなかったと思われる要素と対照的な要素を考えてください。
   ・最後に対照的要素の具体例を最大3つ挙げてくだい。

## 制約事項
・映画のタイトル名は含めないでください。
・各要素は簡潔に記述してください。
・提示されたクラスター群を分析しても「ユーザーが好まなかった要素」として特定できる合理的な根拠がない（長期的嗜好等とも衝突しない）場合は、`none` とだけ出力してください。

## 【抽出された共通要素（嗜好と離れているクラスター情報）】
{formatted_clusters_str}

## ユーザーの嗜好情報
・短期的嗜好（好み）: {short_like_str}
・短期的嗜好（苦手/嫌い）: {short_dislike_str}
・長期的嗜好（好み）: {long_like_str}
・長期的嗜好（苦手/嫌い）: {long_dislike_str}

## 出力フォーマット
推論結果が存在する場合は、コードブロック記法（```jsonなど）を含まない純粋なJSON配列形式で出力してください。
該当する要素がない場合は、装飾なしで `not_unmismatch` という文字列のみを出力してください。

[
  {{
    "unliked_element": "好まなかったと思われる要素",
    "positive": "対照的な要素",
    "positive_example": [
      "対照的要素の具体例1",
      "対照的要素の具体例2",
      "対照的要素の具体例3"
    ]
  }}
]
"""

    def reasoning(
        self,
        un_sim_cluster: List[List[str]],
        short_term_like: List[str],
        short_term_dislike: List[str],
        long_term_like: List[str],
        long_term_dislike: List[str],
    ) -> Union[List[Dict[str, Any]], str]:
        """嗜好から離れているクラスターをもとに、好まなかったと思われる要素と対照的要素を推論する"""

        # プロンプトの構築
        prompt = self._build_prompt(
            un_sim_cluster=un_sim_cluster,
            short_term_like=short_term_like,
            short_term_dislike=short_term_dislike,
            long_term_like=long_term_like,
            long_term_dislike=long_term_dislike,
        )

        # LLMの呼び出し
        response_text = self.llm_client.generate_response(
            role=self.SYSTEM_ROLE,
            prompt=prompt,
            model=self.MODEL,
            temperature=self.TEMPERATURE,
        ).strip()

        # レスポンスが 'none' の場合の処理
        if response_text.lower() == "not_unmismatch":
            return {"status" : "not_unmismatch"}

        # JSONパース処理
        try:
            cleaned_text = response_text.replace("```json", "").replace("```", "").strip()
            parsed_json = json.loads(cleaned_text)
            return {"status" : "unmismatch", "content" : parsed_json }
        except json.JSONDecodeError:
            # パースエラー時のフォールバックとして 'none' を返す
            return {"status" : "not_unmismatch"}