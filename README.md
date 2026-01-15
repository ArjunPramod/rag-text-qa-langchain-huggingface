# Retrieval-Augmented Document Question Answering (RAG) System

---

## 1. Problem Statement

Large Language Models (LLMs) generate responses based on patterns learned during training. While effective for general knowledge, they fail in scenarios involving **private, domain-specific, or up-to-date documents**. When asked questions outside their training data, LLMs often produce **hallucinated or incorrect answers** without indicating uncertainty.

This project addresses the problem of **reliable question answering over custom documents**, where:
- The information is not publicly available to the model.
- Answers must be grounded in provided documents.
- The system should clearly indicate when an answer is not supported by the data.

---

## 2. Solution Overview

This system implements **Retrieval-Augmented Generation (RAG)** to combine document retrieval with language generation.

Instead of asking the LLM to answer directly, the system:
1. Retrieves relevant document excerpts based on the user query.
2. Provides only this retrieved context to the LLM.
3. Constrains the model to generate answers strictly from the provided context.

This approach reduces hallucinations and enables traceable, document-grounded answers.

---

## 3. System Architecture

**End-to-end flow:**

1. **User Query**
   - A question is sent to the API.

2. **Embedding Generation**
   - The query is converted into a dense vector using a sentence embedding model.

3. **Vector Retrieval**
   - FAISS performs similarity search over document embeddings.
   - Maximal Marginal Relevance (MMR) is used to select relevant and diverse chunks.

4. **Context Construction**
   - Retrieved document chunks are combined into a single context block.

5. **Prompt Construction**
   - A constrained prompt instructs the LLM to answer using only the provided context.

6. **LLM Generation**
   - The LLM generates a concise answer or explicitly states uncertainty.

7. **Response Assembly**
   - The final response includes the answer and the source document filenames.

---

## 4. Dataset & Preprocessing

### Documents
- Plain text (`.txt`) documents stored locally.
- Each file represents an independent knowledge source.

### Chunking Strategy
- **Chunk size:** 350 characters  
- **Chunk overlap:** 50 characters  

**Rationale:**
- Ensures each chunk fits within LLM context limits.
- Overlap prevents loss of information at chunk boundaries.
- Values were selected empirically to balance retrieval granularity and semantic coherence.

---

## 5. Models & Design Choices

### Embedding Model
- **Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Why:** Lightweight, fast, and well-suited for semantic similarity tasks.

### Language Model
- **Model:** `google/flan-t5-base`
- **Why:** Instruction-tuned and capable of following constrained prompts.
- **Trade-offs:** High memory usage for local inference; slower startup times.

### Vector Store
- **FAISS**
- **Why:** Efficient in-memory similarity search, suitable for small-to-medium datasets.

### Retrieval Strategy
- **MMR (Maximal Marginal Relevance)**
- **Why:** Reduces redundant chunks and improves coverage of distinct relevant information, which helps reduce hallucinations.

---

## 6. API Design

### Endpoint
`POST /ask`

### Request Example
```json
{
  "question": "What role does FAISS play in document retrieval?"
}
```

### Response Example

```json
{
  "question": "What role does FAISS play in document retrieval?",
  "answer": "FAISS is used to perform efficient similarity search over document embeddings to retrieve relevant context for answering questions.",
  "sources": ["vector_databases.txt"]
}
```

### Source Attribution
- Sources are extracted from the metadata of retrieved document chunks.
- Filenames are returned to provide transparency into where the answer originated.

## 7. Deployment

### Dockerization
- The application is containerized using Docker.
- All dependencies are installed at build time for reproducibility.

### AWS Deployment
- Deployed on AWS EC2 (Ubuntu 22.04 LTS).
- Container exposed via port 8000.
- Swagger UI available for interactive testing.

### Environment Configuration
- CPU-only inference.
- Instance memory requirements dictated local LLM usage.

## 8. Challenges & Learnings

### Memory Constraints
- Local inference with flan-t5-base caused Out-Of-Memory (OOM) crashes on low-memory instances.

### Debugging Process
- Inspected Docker logs to identify premature container exits.
- Used Linux kernel logs (dmesg) to confirm OOM kills.
- Adjusted instance memory configuration to stabilize the service.

### Key Learnings
- Local LLM inference has non-trivial infrastructure requirements.
- Production deployment requires careful resource planning.
- System-level debugging is critical for ML applications in production.

## 9. Limitations

- Scalability: FAISS is in-memory and not suitable for very large datasets.
- Accuracy: Retrieval quality directly impacts answer quality.
- Multi-document reasoning: The system does not perform advanced reasoning across multiple documents.
- Cold start latency: Model loading increases startup time.
- Evaluation: No automated metrics are implemented for answer quality.

## 10. Future Improvements
- Add a reranking stage to improve retrieval precision.
- Experiment with adaptive or semantic chunking strategies.
- Replace FAISS with a production-grade vector database.
- Introduce monitoring, logging, and evaluation pipelines.
- Support document updates without full re-indexing.
