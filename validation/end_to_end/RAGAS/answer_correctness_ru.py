from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics._answer_similarity import AnswerSimilarity
from ragas.metrics._faithfulness_ru import (
    StatementGeneratorInput,
    StatementGeneratorOutput,
    StatementGeneratorPrompt,
)
from ragas.metrics.base import (
    MetricOutputType,
    MetricType,
    MetricWithEmbeddings,
    MetricWithLLM,
    SingleTurnMetric,
)
from ragas.metrics.utils import fbeta_score
from ragas.prompt import PydanticPrompt
from ragas.run_config import RunConfig

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks

logger = logging.getLogger(__name__)


class QuestionAnswerGroundTruth(BaseModel):
    question: str
    answer: list[str]
    ground_truth: list[str]


class StatementsWithReason(BaseModel):
    statement: str
    reason: str


class ClassificationWithReason(BaseModel):
    TP: list[StatementsWithReason]
    FP: list[StatementsWithReason]
    FN: list[StatementsWithReason]


class CorrectnessClassifier(
    PydanticPrompt[QuestionAnswerGroundTruth, ClassificationWithReason]
):
    instruction = """NDA""",
    input_model = QuestionAnswerGroundTruth
    output_model = ClassificationWithReason
    examples = [
        """NDA"""
    ]


@dataclass
class AnswerCorrectnessRU(MetricWithLLM, MetricWithEmbeddings, SingleTurnMetric):
    """
    Measures answer correctness compared to ground truth as a combination of
    factuality and semantic similarity.

    Attributes
    ----------
    name: string
        The name of the metrics
    weights:
        a list of two weights corresponding to factuality and semantic similarity
        Defaults [0.75, 0.25]
    answer_similarity:
        The AnswerSimilarity object
    """

    name: str = "answer_correctness_ru"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {"user_input", "response", "reference"}
        }
    )
    output_type = MetricOutputType.CONTINUOUS
    correctness_prompt: PydanticPrompt = field(default_factory=CorrectnessClassifier)
    statement_generator_prompt: PydanticPrompt = field(
        default_factory=StatementGeneratorPrompt
    )
    weights: list[float] = field(default_factory=lambda: [0.75, 0.25])
    beta: float = 1.0
    answer_similarity: t.Optional[AnswerSimilarity] = None
    max_retries: int = 1

    def __post_init__(self):
        if len(self.weights) != 2:
            raise ValueError(
                "Expects a list of two weights. First for factuality, second for semantic similarity"
            )
        if all([w == 0 for w in self.weights]):
            raise ValueError("At least one weight must be non-zero")
        if not all([w >= 0 for w in self.weights]):
            raise ValueError("Weights must be non-negative")

        if type(self.beta) is not float:
            raise ValueError(
                "Beta must be a float. A beta > 1 gives more weight to recall, while beta < 1 favors precision."
            )

    def init(self, run_config: RunConfig):
        super().init(run_config)
        if self.answer_similarity is None and self.weights[1] != 0:
            self.answer_similarity = AnswerSimilarity(embeddings=self.embeddings)

    def _compute_statement_presence(
        self, prediction: ClassificationWithReason
    ) -> float:
        tp = len(prediction.TP)
        fp = len(prediction.FP)
        fn = len(prediction.FN)
        score = fbeta_score(tp, fp, fn, self.beta)
        return score

    async def _create_simplified_statements(
        self, question: str, text: str, callbacks: Callbacks
    ) -> StatementGeneratorOutput:
        assert self.llm is not None, "llm is not set"

        prompt_input = StatementGeneratorInput(question=question, answer=text)
        statements = await self.statement_generator_prompt.generate(
            llm=self.llm,
            data=prompt_input,
            callbacks=callbacks,
        )

        return statements

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        row = sample.to_dict()
        score = await self._ascore(row, callbacks)
        return score

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        assert self.llm is not None, "LLM must be set"

        # extract the statements from the answer and the ground truth
        question = row["user_input"]
        statements: t.Dict[str, t.List[str]] = {}
        for item in ["response", "reference"]:
            statements_x = await self._create_simplified_statements(
                question, row[item], callbacks
            )
            statements_x = statements_x.statements
            statements[item] = statements_x

        if not all([val == [] for val in statements.values()]):
            ground_truth = [statement for statement in statements["reference"]]
            answer = [statement for statement in statements["response"]]
            answers = await self.correctness_prompt.generate(
                llm=self.llm,
                data=QuestionAnswerGroundTruth(
                    question=question,
                    answer=answer,
                    ground_truth=ground_truth,
                ),
                callbacks=callbacks,
            )
            if answers is None:
                return np.nan

            f1_score = self._compute_statement_presence(answers)
        else:
            f1_score = 1.0

        if self.weights[1] == 0:
            similarity_score = 0.0
        else:
            assert self.answer_similarity is not None, "AnswerSimilarity must be set"

            similarity_score = await self.answer_similarity.ascore(
                row, callbacks=callbacks
            )

        score = np.average(
            [f1_score, similarity_score],
            weights=self.weights,
        )

        return float(score)


answer_correctness_ru = AnswerCorrectnessRU()
