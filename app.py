import os
import json
import uvicorn
from fastapi import FastAPI
from langserve import add_routes
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import chain
from pydantic import BaseModel, Field

# 1. Define Custom Tools
@tool
def check_policy_coverage(plan_type: str, procedure_name: str) -> str:
    """Check coverage status, copay, and deductible for a specific medical procedure under a plan tier."""
    plans = {
        "basic": {"coverage": "60%", "copay": "$50", "pre_auth_required": True},
        "silver": {"coverage": "80%", "copay": "$30", "pre_auth_required": False},
        "gold": {"coverage": "90%", "copay": "$15", "pre_auth_required": False}
    }
    plan_info = plans.get(plan_type.lower(), {"coverage": "70%", "copay": "$35", "pre_auth_required": True})
    return json.dumps({"plan_type": plan_type, "procedure": procedure_name, "details": plan_info})

@tool
def calculate_premium_estimate(age: int, plan_tier: str, family_members: int) -> str:
    """Calculate estimated monthly and annual health insurance premiums based on age, tier, and family size."""
    base_rate = 150 if age < 30 else (250 if age < 50 else 400)
    tier_multiplier = {"basic": 1.0, "silver": 1.3, "gold": 1.7}.get(plan_tier.lower(), 1.0)
    family_cost = (family_members - 1) * 100 if family_members > 1 else 0
    monthly_total = int((base_rate * tier_multiplier) + family_cost)
    return json.dumps({"monthly_estimate_usd": monthly_total, "annual_estimate_usd": monthly_total * 12})

@tool
def guide_claim_submission(claim_type: str) -> str:
    """Get step-by-step instructions and required documents for filing a cashless or reimbursement claim."""
    if "cashless" in claim_type.lower():
        return json.dumps({
            "claim_type": "Cashless",
            "steps": ["Show health card at network hospital desk.", "Submit Pre-Authorization Form."],
            "required_docs": ["Health Card ID", "Government Photo ID"]
        })
    return json.dumps({
        "claim_type": "Reimbursement",
        "steps": ["Pay hospital bills directly.", "Submit claim form within 15 days."],
        "required_docs": ["Original Bills", "Discharge Summary"]
    })

tools = [check_policy_coverage, calculate_premium_estimate, guide_claim_submission]
tools_by_name = {t.name: t for t in tools}

# 2. Model Initialization with Bound Tools
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=GEMINI_API_KEY,
    temperature=0.1
)
llm_with_tools = llm.bind_tools(tools)

SYSTEM_INSTRUCTION = SystemMessage(content=(
    "You are a specialized Health Insurance AI Assistant restricted strictly to health insurance inquiries. "
    "Use the provided tools to fetch policy coverage, premium estimations, and claim guidelines. "
    "For non-health insurance topics, answer strictly: 'I am not authorized to answer questions outside of health insurance.'"
))

# 3. Custom Agent Executor Runnable
@chain
def agent_executor(inputs: dict) -> str:
    user_query = inputs.get("input", "") if isinstance(inputs, dict) else str(inputs)
    messages = [SYSTEM_INSTRUCTION, HumanMessage(content=user_query)]
    
    # First LLM Call
    ai_msg = llm_with_tools.invoke(messages)
    messages.append(ai_msg)
    
    # Handle Tool Calls
    if ai_msg.tool_calls:
        for tool_call in ai_msg.tool_calls:
            selected_tool = tools_by_name.get(tool_call["name"])
            if selected_tool:
                tool_output = selected_tool.invoke(tool_call["args"])
                messages.append(ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"]))
        
        # Second LLM Call with Tool Results
        final_response = llm.invoke(messages)
        return str(final_response.content)
    
    return str(ai_msg.content)

# 4. Input Schema
class AgentInput(BaseModel):
    input: str = Field(..., description="Health insurance inquiry")

app = FastAPI(title="Health Insurance Agent API", version="1.0")

# 5. Add LangServe Route
add_routes(
    app,
    agent_executor.with_types(input_type=AgentInput),
    path="/agent"
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
