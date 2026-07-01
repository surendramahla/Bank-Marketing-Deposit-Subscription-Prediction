"""
chains/rag_chain.py
-------------------
RAG (Retrieval-Augmented Generation) chain using ChromaDB + LangChain.

This chain allows the AI copilot to answer questions using banking-specific
documents (FAQs, policies, product information) instead of relying solely
on LLM parametric knowledge.

Architecture:
  User Question
       ↓
  ChromaDB (similarity search → top-K relevant chunks)
       ↓
  LLM prompt with retrieved context + question
       ↓
  Grounded answer (with citations)

Documents in rag/documents/ are:
  - banking_faq.txt          : Common banking FAQ
  - term_deposit_info.txt    : Term deposit product details
  - marketing_guidelines.txt : Internal marketing compliance rules
"""
import os
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from core.config import get_settings

settings = get_settings()

# ── Module-level vector store (lazily initialized) ────────────────────────────
_vector_store = None
_retriever = None


# ── Vector Store Initialization ───────────────────────────────────────────────
def _get_embeddings():
    """Returns the embedding model based on LLM provider."""
    if settings.LLM_PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY)
    else:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=settings.GOOGLE_API_KEY,
        )


def _get_llm():
    if settings.LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.OPENAI_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=0.2,
        )
    else:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0.2,
            convert_system_message_to_human=True,
        )


def initialize_vector_store(force_rebuild: bool = False):
    """
    Loads or builds the ChromaDB vector store from documents in rag/documents/.
    Persists the store to rag/vector_store/ for fast reloads.

    Call this once on app startup (see main.py lifespan).
    """
    global _vector_store, _retriever

    if _vector_store is not None and not force_rebuild:
        return

    try:
        from langchain_community.document_loaders import DirectoryLoader, TextFileLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_community.vectorstores import Chroma

        docs_dir = settings.RAG_DOCS_DIR
        store_dir = settings.VECTOR_STORE_DIR

        # ── Load documents ──────────────────────────────────────────
        loader = DirectoryLoader(
            docs_dir,
            glob="**/*.txt",
            loader_cls=TextFileLoader,
            show_progress=True,
        )
        raw_docs = loader.load()

        if not raw_docs:
            print(f"[RAG] Warning: No documents found in {docs_dir}. RAG will be unavailable.")
            return

        # ── Chunk documents ─────────────────────────────────────────
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.RAG_CHUNK_SIZE,
            chunk_overlap=settings.RAG_CHUNK_OVERLAP,
        )
        chunks = splitter.split_documents(raw_docs)
        print(f"[RAG] Loaded {len(raw_docs)} documents, split into {len(chunks)} chunks.")

        # ── Build / reload ChromaDB ─────────────────────────────────
        embeddings = _get_embeddings()
        os.makedirs(store_dir, exist_ok=True)

        _vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=store_dir,
            collection_name="bankai_docs",
        )

        _retriever = _vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": settings.RAG_TOP_K},
        )
        print(f"[RAG] Vector store ready with {_vector_store._collection.count()} vectors.")

    except Exception as e:
        print(f"[RAG] Initialization failed: {e}. RAG features will be disabled.")
        _vector_store = None
        _retriever = None


# ── RAG Prompt ────────────────────────────────────────────────────────────────
_RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert banking AI assistant for a retail bank.
Use the provided context documents to answer the user's question accurately.
If the context doesn't contain enough information, say so clearly and provide
general banking knowledge as a supplement.

Always be concise, professional, and helpful.
If relevant to the question, relate the answer back to bank marketing strategy."""),
    ("human", """
Context from bank documents:
{context}

User Question: {question}

Please provide a helpful, accurate answer based on the above context.
"""),
])


def _format_docs(docs) -> str:
    """Formats retrieved documents into a single context string."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def build_rag_chain():
    """
    Builds the full RAG chain using LCEL.

    Returns None if the vector store is not initialized.
    """
    if _retriever is None:
        return None

    llm = _get_llm()
    chain = (
        {"context": _retriever | _format_docs, "question": RunnablePassthrough()}
        | _RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain


# ── Convenience Function ──────────────────────────────────────────────────────
async def answer_question(question: str) -> str:
    """
    Answers a user's question using RAG over banking documents.

    Args:
        question: Natural language question from the employee

    Returns:
        Grounded answer string from the LLM
    """
    chain = build_rag_chain()

    if chain is None:
        # Fallback: LLM without RAG context
        try:
            llm = _get_llm()
            fallback_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an expert banking AI assistant. Answer the question based on general banking knowledge."),
                ("human", "{question}"),
            ])
            fallback_chain = fallback_prompt | llm | StrOutputParser()
            return await fallback_chain.ainvoke({"question": question})
        except Exception as e:
            return f"[RAG unavailable and fallback failed: {str(e)}]"

    try:
        return await chain.ainvoke(question)
    except Exception as e:
        return f"[RAG query failed: {str(e)}]"
