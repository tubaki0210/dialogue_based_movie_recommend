from .FirstQuestionGenerator import FirstQuestionGenerator
from .MismatchQuestionGenerator import MismatchQuestionGenerator
from .UnMismatchQuestionGenerator import UnMismatchQuestionGenerator
from .FallbackQuestionGenerator import FallbackQuestionGenerator
from .QuestionContext import QuestionContext
from utils.LLM import LLM


class QuestionManager:
  """対話状況（ターン数等）に応じて適切な質問生成器を呼び出すマネージャー"""

  def __init__(self, llm_client: LLM):
    self.llm_client = llm_client

    # 各質問生成インスタンスを保持
    self.generators = {
        "first": FirstQuestionGenerator(llm_client),
        "mismatch" : MismatchQuestionGenerator(llm_client),
        "not_mismatch" : UnMismatchQuestionGenerator(llm_client),
        "fallback" : FallbackQuestionGenerator(llm_client)
        # 将来的に追加する例:
    }

  def generate_question(self, mode ,context: QuestionContext):
    """Contextの状態に応じて適切なGeneratorを選び、ストリーミングレスポンスを返す"""
    generator : FirstQuestionGenerator | MismatchQuestionGenerator | UnMismatchQuestionGenerator | FallbackQuestionGenerator = self.generators.get(mode)
    # ターン数やコンテキストの状態に応じてロジックを切り替える

    # 選ばれたGeneratorのgenerate()を呼んでストリーミング（Generator）をそのまま返す
    return generator.generate(context)