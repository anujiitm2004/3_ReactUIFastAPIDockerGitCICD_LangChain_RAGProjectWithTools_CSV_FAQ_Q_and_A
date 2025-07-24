from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_helper import get_qa_chain, create_vector_db

app = FastAPI(title="LangChain GenAI Q&A API")

# ✅ Secure + Functional CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://ec2-34-228-18-34.compute-1.amazonaws.com:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/create_vector_db")
def create_db():
    create_vector_db()
    return {"message": "Vector database created successfully."}

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_question_gaaa(request: QuestionRequest):
    result = get_qa_chain(request.question)

    if isinstance(result, list) and len(result) == 3:
        return {
            "mode": "tool",
            "tool_used": result[0],
            "answer": result[1],
            "raw_messages": [str(m) for m in result[2]]
        }
    elif isinstance(result, dict) and "result" in result:
        filled_prompt = result["PROMPT"].format(**result["input"])
        return {
            "mode": "vectorstore",
            "answer": result["result"],
            "filled_prompt": filled_prompt,
            "sources": [doc.page_content for doc in result.get("source_documents", [])]
        }
    else:
        return {"mode": "unknown", "answer": "⚠️ Unexpected response format."}
