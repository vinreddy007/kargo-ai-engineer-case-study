from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from kargo_media_recommender.clarification import build_clarification_question
from kargo_media_recommender.formatting import format_recommendation_response
from kargo_media_recommender.llm_parser import OpenAIRequirementParser, RequirementParser
from kargo_media_recommender.recommender import RecommendationEngine
from kargo_media_recommender.schemas import ClientRequirements, RecommendationResult


class RecommendationState(TypedDict, total=False):
    latest_user_message: str
    requirements: ClientRequirements | None
    missing_fields: tuple[str, ...]
    clarification_question: str | None
    recommendation: RecommendationResult | None
    response_text: str | None


@dataclass(frozen=True)
class WorkflowOutput:
    requirements: ClientRequirements
    missing_fields: tuple[str, ...]
    clarification_question: str | None
    recommendation: RecommendationResult | None
    response_text: str

    @property
    def needs_clarification(self) -> bool:
        return bool(self.missing_fields)


class RecommendationWorkflow:
    def __init__(
        self,
        engine: RecommendationEngine,
        parser: RequirementParser | None = None,
    ) -> None:
        self.engine = engine
        self.parser = parser or OpenAIRequirementParser()
        self.graph = self._build_graph()

    @classmethod
    def from_data_dir(
        cls,
        data_dir: str | Path | None = None,
        parser: RequirementParser | None = None,
    ) -> "RecommendationWorkflow":
        return cls(RecommendationEngine.from_data_dir(data_dir), parser=parser)

    def run(
        self,
        message: str,
        current_requirements: ClientRequirements | None = None,
    ) -> WorkflowOutput:
        final_state = self.graph.invoke(
            {
                "latest_user_message": message,
                "requirements": current_requirements,
                "missing_fields": (),
                "clarification_question": None,
                "recommendation": None,
                "response_text": None,
            }
        )

        requirements = final_state["requirements"]
        response_text = final_state.get("response_text") or ""
        return WorkflowOutput(
            requirements=requirements,
            missing_fields=final_state.get("missing_fields", ()),
            clarification_question=final_state.get("clarification_question"),
            recommendation=final_state.get("recommendation"),
            response_text=response_text,
        )

    def _build_graph(self):
        graph = StateGraph(RecommendationState)
        graph.add_node("parse_requirements", self._parse_requirements)
        graph.add_node("check_required_fields", self._check_required_fields)
        graph.add_node("ask_clarification", self._ask_clarification)
        graph.add_node("recommend", self._recommend)
        graph.add_node("format_response", self._format_response)

        graph.add_edge(START, "parse_requirements")
        graph.add_edge("parse_requirements", "check_required_fields")
        graph.add_conditional_edges(
            "check_required_fields",
            self._route_after_check,
            {
                "ask_clarification": "ask_clarification",
                "recommend": "recommend",
            },
        )
        graph.add_edge("ask_clarification", END)
        graph.add_edge("recommend", "format_response")
        graph.add_edge("format_response", END)
        return graph.compile()

    def _parse_requirements(self, state: RecommendationState) -> RecommendationState:
        requirements = self.parser.parse(
            state["latest_user_message"],
            existing_requirements=state.get("requirements"),
        )
        return {"requirements": requirements}

    @staticmethod
    def _check_required_fields(state: RecommendationState) -> RecommendationState:
        requirements = state["requirements"]
        return {"missing_fields": requirements.missing_required_fields}

    @staticmethod
    def _route_after_check(state: RecommendationState) -> Literal["ask_clarification", "recommend"]:
        if state.get("missing_fields"):
            return "ask_clarification"
        return "recommend"

    @staticmethod
    def _ask_clarification(state: RecommendationState) -> RecommendationState:
        question = build_clarification_question(state.get("missing_fields", ()))
        return {
            "clarification_question": question,
            "response_text": question,
            "recommendation": None,
        }

    def _recommend(self, state: RecommendationState) -> RecommendationState:
        recommendation = self.engine.recommend(state["requirements"])
        return {"recommendation": recommendation}

    @staticmethod
    def _format_response(state: RecommendationState) -> RecommendationState:
        recommendation = state["recommendation"]
        return {"response_text": format_recommendation_response(recommendation)}
