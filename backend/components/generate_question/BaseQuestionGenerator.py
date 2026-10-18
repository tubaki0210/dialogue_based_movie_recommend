from abc import ABC, abstractmethod
from .QuestionContext import QuestionContext

# ==============================================================================
# 2. 抽象基底クラス（インターフェース）
# ==============================================================================
class BaseQuestionGenerator(ABC):
    """すべての質問生成クラスが継承する抽象クラス"""

    @abstractmethod
    def generate(self, context: QuestionContext) -> str:
        """
        引数は context のみに統一！
        各サブクラスは context から必要なデータだけを取り出して使う。
        """
        pass