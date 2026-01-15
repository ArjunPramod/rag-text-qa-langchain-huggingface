import os
from fastapi import FastAPI
from pydantic import BaseModel

# Loaders & vector DB
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS

# Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

# LCEL core
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

# FastAPI app
app = FastAPI(title="RAG Document QA API")


# Load ALL text files from data/
data_dir = "data"
documents = []

for filename in os.listdir(data_dir):
    if filename.endswith(".txt"):
        loader = TextLoader(
            os.path.join(data_dir, filename),
            encoding="utf-8"
        )
        documents.extend(loader.load())

print(f"Loaded {len(documents)} documents")

# Split documents
splitter = CharacterTextSplitter(chunk_size=350, chunk_overlap=50)
docs = splitter.split_documents(documents)

# Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Vector store + retriever
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 2, "lambda_mult": 0.7}
)

# Local LLM
hf_pipeline = pipeline(
    task="text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=256,
)

llm = HuggingFacePipeline(pipeline=hf_pipeline)

# Prompt
prompt = PromptTemplate.from_template(
    """You are an assistant answering questions from documents.

Use ONLY the context below.
If the answer is NOT present in the context, say:
"I do not know based on the provided documents."

If the question asks for steps or a process, explain it step by step.
If the question asks about a specific tool or library, define its role clearly.

Do NOT repeat sentences or restate the same idea multiple times.
Summarize instead of copying.
Answer concisely in 2–3 complete sentences with sufficient explanation.

Context:
{context}

Question:
{question}

Answer:
"""
)

# LCEL RAG pipeline
rag_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
)

# Request / Response schemas
class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]

# Helper: get sources
def get_sources(question: str) -> list[str]:
    docs = retriever.invoke(question)
    sources = sorted({os.path.basename(doc.metadata["source"]) for doc in docs})
    return sources

# API endpoint
@app.post("/ask", response_model=AnswerResponse)
def ask_question(req: QuestionRequest):
    if not req.question.strip():
        return {
            "question": req.question,
            "answer": "I do not know based on the provided documents.",
            "sources": []
        }

    answer = rag_chain.invoke(req.question)
    clean_answer = " ".join(answer.replace("\n", " ").split())
    sources = get_sources(req.question)

    return {
        "question": req.question,
        "answer": clean_answer,
        "sources": sources,
    }
