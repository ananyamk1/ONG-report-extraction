"""
RAG Q&A chain — lets engineers ask natural language questions
across all ingested drilling reports.

Examples:
  "What was the mud weight on July 15?"
  "Which report had the highest ROP and what was the formation?"
  "Summarise the pump pressure trend across all reports."
  "Were there any CO2 spikes above 1%?"
"""
from __future__ import annotations

from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

from src.config import LLM_PROVIDER, LLM_MODEL, OPENAI_API_KEY, ANTHROPIC_API_KEY
from src.vectorstore import DrillVectorStore


_RAG_TEMPLATE = """\
You are an expert petroleum drilling engineer assistant.
Use the following excerpts from daily drilling reports to answer the question.
If the answer is not in the provided context, say "I don't have enough data in \
the loaded reports to answer that." Do NOT make up numbers.

When citing data, mention the report date and/or report number it came from.

CONTEXT:
{context}

QUESTION: {question}

ANSWER (be concise and precise, include units):"""


def _get_llm():
    if LLM_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=LLM_MODEL or "claude-3-5-haiku-20241022",
            api_key=ANTHROPIC_API_KEY,
            temperature=0.1,
        )
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=LLM_MODEL or "gpt-4o-mini",
            api_key=OPENAI_API_KEY,
            temperature=0.1,
        )


class DrillingQA:
    """
    RAG retrieval + generation chain over the ChromaDB vector store.
    Answers questions by retrieving the most relevant report chunks first.
    """

    def __init__(self, vector_store: DrillVectorStore, k: int = 5):
        self._vs = vector_store
        self._k = k
        self._chain = self._build_chain()

    def ask(self, question: str, well_filter: str | None = None) -> dict:
        """
        Ask a natural language question.

        Args:
            question:    Engineer's question in plain English.
            well_filter: Optional well name to restrict retrieval scope.

        Returns:
            dict with 'answer' (str) and 'source_chunks' (list[Document]).
        """
        filter_dict = None
        if well_filter:
            filter_dict = {"well_name": {"$eq": well_filter}}

        # Retrieve relevant chunks
        chunks = self._vs.similarity_search(question, k=self._k, filter=filter_dict)

        if not chunks:
            return {
                "answer": "No relevant reports found in the vector store.",
                "source_chunks": [],
            }

        context = "\n\n---\n\n".join(
            f"[{c.metadata.get('source_file','?')} | "
            f"Date: {c.metadata.get('report_date','?')} | "
            f"Section: {c.metadata.get('section','?')}]\n{c.page_content}"
            for c in chunks
        )

        prompt = PromptTemplate.from_template(_RAG_TEMPLATE)
        llm = _get_llm()
        formatted = prompt.format(context=context, question=question)
        response = llm.invoke(formatted)

        return {
            "answer": response.content.strip(),
            "source_chunks": chunks,
        }

    def _build_chain(self):
        """Build a RetrievalQA chain (kept for compatibility)."""
        retriever = self._vs.as_retriever(k=self._k)
        prompt = PromptTemplate.from_template(_RAG_TEMPLATE)
        llm = _get_llm()
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
        )
