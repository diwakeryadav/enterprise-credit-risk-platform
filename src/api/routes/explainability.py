from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.explainability.agent import ExplainabilityAgent
from src.logger import logger

router = APIRouter()


class ExplanationRequest(BaseModel):
    loan_application_data: Dict[str, Any] = Field(
        ..., description="Key-value features of the loan applicant"
    )
    shap_values: Dict[str, float] = Field(
        ..., description="SHAP feature importances from the ML model output"
    )


class ExplanationResponse(BaseModel):
    is_compliant: bool
    final_narrative: str
    revision_count: int


@router.post("/explain-denial", response_model=ExplanationResponse)
async def explain_denial(request: ExplanationRequest):
    """
    Exposes the LangGraph agent to generate compliant loan denial explanations
    from raw application data and model SHAP value outputs.
    """
    try:
        agent = ExplainabilityAgent()
        result = agent.run(
            loan_data=request.loan_application_data, shap_values=request.shap_values
        )
        return ExplanationResponse(
            is_compliant=result.get("is_compliant", False),
            final_narrative=result.get("final_narrative", ""),
            revision_count=result.get("revision_count", 0),
        )
    except Exception as e:
        logger.error(f"Error generating explanation narrative: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to generate explanation: {str(e)}"
        )
