import os
import uvicorn
import json
from fastapi import FastAPI
from langserve import add_routes
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from pydantic import BaseModel, Field

@tool
def check_policy_coverage(plan_type: str, procedure_name: str) -> str:
    """Check coverage status, copay, and deductible for a specific medical procedure under a plan tier."""
    plans = {
        "basic": {"coverage": "60%", "copay": "$50", "pre_auth_required": True},
        "silver": {"coverage": "80%", "copay": "$30", "pre_auth_required": False},
        "gold": {"coverage": "90%", "copay": "$15", "pre_auth_required": False}
    }
    plan_info = plans.get(plan_type.lower(), {"coverage": "70%", "copay": "$35", "pre_auth_required": True})
    return json.dumps({
        "plan_type": plan_type,
        "procedure": procedure_name,
        "details": plan_info
    })

@tool
def calculate_premium_estimate(age: int, plan_tier: str, family_members: int) -> str:
    """Calculate estimated monthly and annual health insurance premiums based on age, tier, and family size."""
    base_rate = 150 if age < 30 else (250 if age < 50 else 400)
    tier_multiplier = {"basic": 1.0, "silver": 1.3, "gold": 1.7}.get(plan_tier.lower(), 1.0)
    family_cost = (family_members - 1) * 100 if family_members > 1 else 0
    
    monthly_total = int((base_rate * tier_multiplier) + family_cost)
    annual_total = monthly_total * 12
    
    return json.dumps({
        "age": age,
        "plan_tier": plan_tier,
        "family_members": family_members,
        "monthly_estimate_usd": monthly_total,
        "annual_estimate_usd": annual_total
    })

@tool
def guide_claim_submission(claim_type: str) -> str:
    """Get step-by-step instructions and required documents for filing a cashless or reimbursement claim."""
    if "cashless" in claim_type.lower():
        return json.dumps({
            "claim_type": "Cashless",
            "steps": [
                "Show health card at network hospital desk.",
                "Fill out Pre-Authorization Form 48 hours prior to planned treatment.",
                "Hospital submits claim directly to insurer."
            ],
            "required_docs": ["Health Card ID", "Government Photo ID", "Doctor Recommendation Letter"]
        })
    else:
        return json.dumps({
            "claim_type": "Reimbursement",
            "steps": [
                "Pay hospital bills directly during discharge.",
                "Submit claim form within 15 days of discharge.",
                "Upload itemized bills and discharge summary."
            ],
            "required_docs": ["Original Hospital Bills", "Discharge Summary", "Payment Receipts", "Cancelled Check"]
        })

tools = [check_policy_coverage, calculate_premium_estimate, guide_claim_submission]
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

llm_flash = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    api_key=GEMINI_API_KEY
)

agent = create_agent(
    model=llm_flash,
    tools=tools,
    system_prompt=(
        "You are a specialized Health Insurance AI Assistant restricted strictly to policy coverage inquiries, "
        "premium estimations, and claims submission guidance. For any question outside health insurance, "
        "you must answer strictly: 'I am not authorized to answer questions outside of health insurance coverage, premiums, and claims.'"
    )
)

class InputSchema(BaseModel):
    input: str = Field(..., description="Enter your health insurance inquiry here")

app = FastAPI(
    title="Health Insurance Agent API",
    version="1.0",
    description="A deployment of a Health Insurance LangChain agent using Google Gemini"
)

add_routes(
    app,
    agent.with_types(input_type=InputSchema),
    path="/agent",
    playground_type="default"
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
