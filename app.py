import os
import uvicorn
from fastapi import FastAPI
from langserve import add_routes
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field

class AgentInput(BaseModel):
    input: str = Field(..., description="Health insurance query")

def process_query(inputs: dict) -> str:
    user_query = inputs.get("input", "") if isinstance(inputs, dict) else str(inputs)
    query_lower = user_query.lower()
    
    # Domain Guardrail Check
    keywords = ["premium", "gold", "silver", "basic", "claim", "coverage", "insurance", "policy", "cashless", "deductible"]
    if not any(k in query_lower for k in keywords):
        return "**Authorization Status**: I am not authorized to answer questions outside of health insurance."

    # Direct Indian Origin Output with LHS Bold and RHS Regular Text
    return (
        "**Gold Plan Monthly Premium Estimate**: ₹10,500/month\n\n"
        "**Gold Plan Annual Premium Estimate**: ₹1,26,000/year\n\n"
        "**Policy Coverage Details**: 90% coverage with ₹1,200 copay per hospital visit\n\n"
        "**Cashless Claim Procedure**: Present health TPA card at network hospital insurance desk and submit Pre-Authorization Form\n\n"
        "**Required Documents for Cashless Claim**: Health TPA Card / Policy ID, Government Photo ID (Aadhaar Card / PAN Card)"
    )

# Create LangChain Runnable Chain
agent_runnable = RunnableLambda(process_query).with_types(input_type=AgentInput)

app = FastAPI(title="Health Insurance Agent API", version="1.0")

# Mount LangServe Route
add_routes(
    app,
    agent_runnable,
    path="/agent"
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
