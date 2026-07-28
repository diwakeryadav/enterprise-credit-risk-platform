import sys
import yaml
from typing import Any, Dict, List, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from src.constants import CONFIG_FILE_PATH
from src.exception import CreditRiskException
from src.explainability.prompts import (SYSTEM_COMPLIANCE_PROMPT,
                                        SYSTEM_DRAFT_PROMPT,
                                        USER_DRAFT_TEMPLATE)
from src.logger import logger


class ExplainabilityState(TypedDict):
    loan_data: Dict[str, Any]
    shap_values: Dict[str, float]
    top_features: List[Dict[str, Any]]
    draft_narrative: str
    compliance_feedback: str
    final_narrative: str
    is_compliant: bool
    revision_count: int
    max_revisions: int


class ExplainabilityAgent:
    def __init__(
        self,
        model_name: str = None,
        temperature: float = None,
        max_revisions: int = None,
    ):
        try:
            with open(CONFIG_FILE_PATH, "r") as f:
                config = yaml.safe_load(f)
            agent_config = config.get("explainability_agent", {})
        except Exception:
            agent_config = {}

        model_name = model_name or agent_config.get("model_name", "gpt-4o-mini")
        temperature = temperature if temperature is not None else agent_config.get("temperature", 0.0)
        self.max_revisions = max_revisions if max_revisions is not None else agent_config.get("max_revisions", 2)
        
        base_url = agent_config.get("base_url")
        api_key = agent_config.get("api_key")

        llm_kwargs = {
            "model": model_name,
            "temperature": temperature
        }
        if base_url:
            llm_kwargs["base_url"] = base_url
        if api_key:
            llm_kwargs["api_key"] = api_key

        logger.info(f"Initializing ExplainabilityAgent with model={model_name}, base_url={base_url}")
        self.llm = ChatOpenAI(**llm_kwargs)
        self.workflow = self._build_graph()

    def _build_graph(self) -> StateGraph:
        builder = StateGraph(ExplainabilityState)

        # Define nodes
        builder.add_node("analyze_shap", self.analyze_shap)
        builder.add_node("generate_explanation", self.generate_explanation)
        builder.add_node("evaluate_compliance", self.evaluate_compliance)

        # Set entry point
        builder.set_entry_point("analyze_shap")

        # Connect nodes
        builder.add_edge("analyze_shap", "generate_explanation")
        builder.add_edge("generate_explanation", "evaluate_compliance")

        # Add conditional edge for compliance review
        builder.add_conditional_edges(
            "evaluate_compliance",
            self.should_continue,
            {"continue": "generate_explanation", "end": END},
        )

        return builder.compile()

    def analyze_shap(self, state: ExplainabilityState) -> Dict[str, Any]:
        """Node: Extract top 3 features contributing to high default risk (negative impact)"""
        try:
            logger.info("Extracting top negative SHAP features")
            shap_dict = state["shap_values"]

            # Sort by SHAP value descending (higher SHAP = higher default risk score)
            sorted_features = sorted(
                shap_dict.items(), key=lambda item: item[1], reverse=True
            )
            top_3 = [
                {
                    "feature": k,
                    "shap_value": v,
                    "actual_value": state["loan_data"].get(k, "N/A"),
                }
                for k, v in sorted_features[:3]
            ]

            return {"top_features": top_3, "revision_count": 0}
        except Exception as e:
            raise CreditRiskException(e, sys)

    def generate_explanation(self, state: ExplainabilityState) -> Dict[str, Any]:
        """Node: Draft a narrative translation of SHAP features"""
        try:
            logger.info("Drafting business-friendly loan denial narratives")

            shap_contributors_str = "\n".join(
                [
                    f"- {item['feature']}: actual value = {item['actual_value']}, SHAP contribution = {item['shap_value']:.4f}"
                    for item in state["top_features"]
                ]
            )

            user_msg = USER_DRAFT_TEMPLATE.format(
                loan_data=str(state["loan_data"]),
                shap_contributors=shap_contributors_str,
            )

            messages = [
                SystemMessage(content=SYSTEM_DRAFT_PROMPT),
                HumanMessage(content=user_msg),
            ]

            # Append feedback if it is a revision cycle
            if state.get("compliance_feedback"):
                feedback_msg = (
                    "Previous draft was rejected. Compliance Feedback:\n"
                    f"{state['compliance_feedback']}\n\n"
                    "Please revise the draft to address these concerns."
                )
                messages.append(HumanMessage(content=feedback_msg))

            response = self.llm.invoke(messages)

            return {
                "draft_narrative": response.content,
                "revision_count": state.get("revision_count", 0) + 1,
            }
        except Exception as e:
            raise CreditRiskException(e, sys)

    def evaluate_compliance(self, state: ExplainabilityState) -> Dict[str, Any]:
        """Node: Audit the draft narrative for compliance rules (FCRA / ECOA)"""
        try:
            logger.info("Evaluating drafted narratives for compliance compliance")

            user_msg = (
                f"Draft narrative to evaluate:\n{state['draft_narrative']}\n\n"
                f"Original top features:\n{state['top_features']}"
            )

            messages = [
                SystemMessage(content=SYSTEM_COMPLIANCE_PROMPT),
                HumanMessage(content=user_msg),
            ]

            response = self.llm.invoke(messages)
            audit_result = response.content.strip()

            if audit_result.startswith("APPROVED"):
                logger.info("Narrative approved by compliance agent.")
                return {
                    "is_compliant": True,
                    "final_narrative": state["draft_narrative"],
                    "compliance_feedback": "",
                }
            else:
                logger.warning(
                    f"Narrative rejected by compliance agent. Feedback: {audit_result}"
                )
                return {"is_compliant": False, "compliance_feedback": audit_result}
        except Exception as e:
            raise CreditRiskException(e, sys)

    def should_continue(self, state: ExplainabilityState) -> str:
        """Decide whether to loop for revisions or end"""
        if (
            state.get("is_compliant")
            or state.get("revision_count", 0) >= self.max_revisions
        ):
            return "end"
        return "continue"

    def run(
        self, loan_data: Dict[str, Any], shap_values: Dict[str, float]
    ) -> Dict[str, Any]:
        inputs = {
            "loan_data": loan_data,
            "shap_values": shap_values,
            "max_revisions": self.max_revisions,
        }
        return self.workflow.invoke(inputs)
