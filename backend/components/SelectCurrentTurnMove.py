from utils.LLM import LLM

class SelectCurrentTurnMove:
      # クラス変数
    SYSTEM_ROLE = "あなたは対話型映画推薦システムにおいて、現在のターンでシステムが取るべき対話行為を適切に判断するアシスタントです"
    MODEL = "gpt-6-luna"
    REASONING_LEVEL = "none"
    TEMPERATURE = 0.3  # 推論の論理的整合性を保つため低めに設定
    
    def __init__(self, llm_client : LLM):
        self.llm_client = llm_client

    def select_move(self, dialogue, move_history):
        prompt = self._build_prompt(dialogue, move_history)
        response = self.llm_client.generate_response(self.SYSTEM_ROLE, prompt, self.MODEL, self.TEMPERATURE, self.REASONING_LEVEL)
        return response

    def _build_prompt(self, dialogue, move_history):
        format_dialogue = [f"[{utt['role']}]: {utt['content']}" for utt in dialogue]
        join_dialogue = "\n".join(format_dialogue)
        format_move_history = ""
        for turn, move in enumerate(move_history):
            format_move_history += f"{turn + 1}ターン目: {move}\n"
        prompt = f"""\
## タスク 
現在のターンで行うべき適切な対話履歴を参照し、対話行為を選択肢の中から１つ選んでしてください

## 対話行為の選択肢
・recommend : 推薦する
・mismatch_question : 推薦に対してユーザーが不満を持った場合に、質問する
・explanation : ユーザーが映画について尋ねた場合に、映画の説明を行う

## 対話行為の選択の基準
・推薦の後に、「他には？」などときたら再度recommendです

## 対話履歴
{join_dialogue}

## 対話行為の履歴
{format_move_history}

## 出力形式
選択肢の英語だけ出力してください。それ以外は出力しないでください
例）recommend
"""
        return prompt