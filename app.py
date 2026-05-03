import os
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime

from fastapi import FastAPI
from pydantic import BaseModel

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document


CHUNK_ID_NAMESPACE = "rag-chunk-v1"
INDEX_STATE_FILE = "index_state.json"


@dataclass
class ChunkMetadata:
    source: str
    chunk_id: str
    chunk_index: int
    total_chunks: int
    content_preview: str
    file_hash: str
    file_modified_time: float
    chunk_hash: str


@dataclass
class FileIndexState:
    file_hash: str
    file_modified_time: float
    chunk_ids: List[str]
    indexed_at: float


@dataclass
class IndexState:
    files: Dict[str, FileIndexState]
    created_at: float
    updated_at: float


@dataclass
class SourceChunk:
    source: str
    chunk_id: str
    chunk_index: int
    total_chunks: int
    content_preview: str
    file_hash: str


class QuestionRequest(BaseModel):
    question: str


class SourceChunkResponse(BaseModel):
    source: str
    chunk_id: str
    chunk_index: int
    total_chunks: int
    content_preview: str


class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceChunkResponse]


def generate_chunk_id(file_path: str, chunk_index: int, chunk_content: str) -> str:
    content_hash = hashlib.sha256(chunk_content.encode("utf-8")).hexdigest()[:16]
    file_basename = os.path.basename(file_path)
    return f"{CHUNK_ID_NAMESPACE}:{file_basename}:{chunk_index}:{content_hash}"


def compute_file_hash(file_path: str) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_content_preview(content: str, max_length: int = 150) -> str:
    preview = content[:max_length].replace("\n", " ").strip()
    if len(content) > max_length:
        preview += "..."
    return preview


class DocumentManager:
    def __init__(self, data_dir: str, index_dir: str = ".faiss_index"):
        self.data_dir = data_dir
        self.index_dir = index_dir
        self.state_file = os.path.join(index_dir, INDEX_STATE_FILE)
        self.splitter = CharacterTextSplitter(chunk_size=350, chunk_overlap=50)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vectorstore: Optional[FAISS] = None
        self.state: IndexState = self._load_state()

    def _load_state(self) -> IndexState:
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    files = {}
                    for filename, file_data in data.get("files", {}).items():
                        files[filename] = FileIndexState(
                            file_hash=file_data["file_hash"],
                            file_modified_time=file_data["file_modified_time"],
                            chunk_ids=file_data["chunk_ids"],
                            indexed_at=file_data["indexed_at"]
                        )
                    return IndexState(
                        files=files,
                        created_at=data.get("created_at", datetime.now().timestamp()),
                        updated_at=data.get("updated_at", datetime.now().timestamp())
                    )
            except Exception:
                pass
        return IndexState(
            files={},
            created_at=datetime.now().timestamp(),
            updated_at=datetime.now().timestamp()
        )

    def _save_state(self):
        os.makedirs(self.index_dir, exist_ok=True)
        data = {
            "files": {
                filename: {
                    "file_hash": state.file_hash,
                    "file_modified_time": state.file_modified_time,
                    "chunk_ids": state.chunk_ids,
                    "indexed_at": state.indexed_at
                }
                for filename, state in self.state.files.items()
            },
            "created_at": self.state.created_at,
            "updated_at": self.state.updated_at
        }
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _get_current_files(self) -> Dict[str, Dict[str, Any]]:
        files = {}
        if not os.path.exists(self.data_dir):
            return files
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".txt"):
                file_path = os.path.join(self.data_dir, filename)
                files[filename] = {
                    "path": file_path,
                    "modified_time": os.path.getmtime(file_path)
                }
        return files

    def _load_and_split_document(self, file_path: str) -> List[Document]:
        loader = TextLoader(file_path, encoding="utf-8")
        documents = loader.load()
        return self.splitter.split_documents(documents)

    def _create_enhanced_documents(
        self,
        file_path: str,
        chunks: List[Document],
        file_hash: str,
        file_modified_time: float
    ) -> List[Document]:
        enhanced_docs = []
        total_chunks = len(chunks)
        
        for chunk_index, chunk in enumerate(chunks):
            chunk_id = generate_chunk_id(file_path, chunk_index, chunk.page_content)
            chunk_hash = hashlib.sha256(chunk.page_content.encode("utf-8")).hexdigest()
            
            metadata = ChunkMetadata(
                source=os.path.basename(file_path),
                chunk_id=chunk_id,
                chunk_index=chunk_index,
                total_chunks=total_chunks,
                content_preview=get_content_preview(chunk.page_content),
                file_hash=file_hash,
                file_modified_time=file_modified_time,
                chunk_hash=chunk_hash
            )
            
            enhanced_doc = Document(
                page_content=chunk.page_content,
                metadata=asdict(metadata)
            )
            enhanced_docs.append(enhanced_doc)
        
        return enhanced_docs

    def _clear_disk_index(self):
        import shutil
        index_path = os.path.join(self.index_dir, "index")
        if os.path.exists(index_path):
            try:
                if os.path.isdir(index_path):
                    shutil.rmtree(index_path)
                else:
                    os.remove(index_path)
            except Exception:
                pass

    def _rebuild_full_index(self, files: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        all_docs: List[Document] = []
        new_state_files: Dict[str, FileIndexState] = {}
        
        for filename, file_info in files.items():
            file_path = file_info["path"]
            file_hash = compute_file_hash(file_path)
            file_mtime = file_info["modified_time"]
            
            chunks = self._load_and_split_document(file_path)
            enhanced_docs = self._create_enhanced_documents(
                file_path, chunks, file_hash, file_mtime
            )
            
            chunk_ids = [doc.metadata["chunk_id"] for doc in enhanced_docs]
            
            new_state_files[filename] = FileIndexState(
                file_hash=file_hash,
                file_modified_time=file_mtime,
                chunk_ids=chunk_ids,
                indexed_at=datetime.now().timestamp()
            )
            
            all_docs.extend(enhanced_docs)
        
        if all_docs:
            self.vectorstore = FAISS.from_documents(all_docs, self.embeddings)
            index_path = os.path.join(self.index_dir, "index")
            self.vectorstore.save_local(index_path)
        else:
            self.vectorstore = None
            self._clear_disk_index()
        
        self.state.files = new_state_files
        self.state.updated_at = datetime.now().timestamp()
        self._save_state()
        
        return {
            "rebuilt": True,
            "total_indexed_files": len(self.state.files),
            "total_indexed_chunks": sum(
                len(s.chunk_ids) for s in self.state.files.values()
            )
        }

    def _check_index_consistency(self, current_files: Dict[str, Dict[str, Any]]) -> bool:
        if set(self.state.files.keys()) != set(current_files.keys()):
            return False
        
        for filename, file_info in current_files.items():
            if filename not in self.state.files:
                return False
            
            state = self.state.files[filename]
            current_mtime = file_info["modified_time"]
            
            if current_mtime > state.file_modified_time:
                current_hash = compute_file_hash(file_info["path"])
                if current_hash != state.file_hash:
                    return False
        
        return True

    def sync_index(self, force_rebuild: bool = False) -> Dict[str, Any]:
        current_files = self._get_current_files()
        
        if force_rebuild or not self._check_index_consistency(current_files):
            return self._rebuild_full_index(current_files)
        
        return {
            "rebuilt": False,
            "message": "Index is already up to date",
            "total_indexed_files": len(self.state.files),
            "total_indexed_chunks": sum(
                len(s.chunk_ids) for s in self.state.files.values()
            )
        }

    def load_existing_index(self) -> bool:
        index_path = os.path.join(self.index_dir, "index")
        if os.path.exists(index_path):
            try:
                self.vectorstore = FAISS.load_local(
                    index_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                return True
            except Exception:
                pass
        return False

    def get_retriever(self):
        if self.vectorstore is None:
            raise RuntimeError("Vector store not initialized")
        return self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 2, "lambda_mult": 0.7}
        )


app = FastAPI(title="RAG Document QA API")

data_dir = "data"
doc_manager = DocumentManager(data_dir)

print("Initializing index...")
if doc_manager.load_existing_index():
    print("Loaded existing index from disk")

sync_result = doc_manager.sync_index()
print(f"Index sync complete: {sync_result}")

hf_pipeline = pipeline(
    task="text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=256,
)

llm = HuggingFacePipeline(pipeline=hf_pipeline)

prompt = PromptTemplate.from_template(
    """You are an assistant answering questions from documents.

Use ONLY the context below.
If the answer is NOT present in the context, say:
"I do not know based on the provided documents."

If the question asks for steps or a process, explain it step by step.
If the question asks about a specific tool or library, define its role clearly.

Do NOT repeat sentences or restate the same idea multiple times.
Summarize instead of copying.
Answer concisely in 2-3 complete sentences with sufficient explanation.

Context:
{context}

Question:
{question}

Answer:
"""
)


def docs_to_source_chunks(docs: List[Document]) -> List[SourceChunk]:
    chunks = []
    for doc in docs:
        metadata = doc.metadata
        chunk = SourceChunk(
            source=metadata.get("source", "unknown"),
            chunk_id=metadata.get("chunk_id", ""),
            chunk_index=metadata.get("chunk_index", 0),
            total_chunks=metadata.get("total_chunks", 1),
            content_preview=metadata.get("content_preview", ""),
            file_hash=metadata.get("file_hash", "")
        )
        chunks.append(chunk)
    return chunks


def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


@app.post("/ask", response_model=AnswerResponse)
def ask_question(req: QuestionRequest):
    if not req.question.strip():
        return {
            "question": req.question,
            "answer": "I do not know based on the provided documents.",
            "sources": []
        }

    if doc_manager.vectorstore is None:
        return {
            "question": req.question,
            "answer": "I do not know based on the provided documents.",
            "sources": []
        }

    retriever = doc_manager.get_retriever()
    retrieved_docs = retriever.invoke(req.question)

    context = format_docs(retrieved_docs)
    formatted_prompt = prompt.format(context=context, question=req.question)
    answer = llm.invoke(formatted_prompt)
    clean_answer = " ".join(answer.replace("\n", " ").split())

    source_chunks = docs_to_source_chunks(retrieved_docs)
    source_responses = [
        SourceChunkResponse(
            source=chunk.source,
            chunk_id=chunk.chunk_id,
            chunk_index=chunk.chunk_index,
            total_chunks=chunk.total_chunks,
            content_preview=chunk.content_preview
        )
        for chunk in source_chunks
    ]

    return {
        "question": req.question,
        "answer": clean_answer,
        "sources": source_responses,
    }


@app.post("/sync-index")
def sync_index_endpoint(force_rebuild: bool = False):
    result = doc_manager.sync_index(force_rebuild=force_rebuild)
    return result


@app.get("/index-status")
def get_index_status():
    return {
        "total_files": len(doc_manager.state.files),
        "total_chunks": sum(
            len(s.chunk_ids) for s in doc_manager.state.files.values()
        ),
        "files": [
            {
                "filename": filename,
                "chunk_count": len(state.chunk_ids),
                "file_hash": state.file_hash,
                "indexed_at": datetime.fromtimestamp(state.indexed_at).isoformat()
            }
            for filename, state in doc_manager.state.files.items()
        ]
    }
