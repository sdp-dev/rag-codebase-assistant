"""LangChain conversation chain for code Q&A."""

import os
from typing import Optional

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from config import LLM_MODEL

# Ollama host (for Docker: http://ollama:11434, local: http://localhost:11434)
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
from src.retrieval import CodeRetriever
from .prompts import SYSTEM_PROMPT, format_prompt


class CodeAssistantChain:
    """RAG-powered code assistant using LangChain."""

    def __init__(
        self,
        retriever: CodeRetriever,
        model_name: str = LLM_MODEL,
        temperature: float = 0.1
    ):
        """Initialize the assistant chain.

        Args:
            retriever: CodeRetriever instance for semantic search
            model_name: Ollama model name (llama3, mistral, codellama, etc.)
            temperature: LLM temperature for response generation
        """
        self.retriever = retriever
        self.llm = ChatOllama(model=model_name, temperature=temperature, base_url=OLLAMA_HOST)
        self.history: list = []
        self.active_repos: Optional[list[str]] = None

    def set_active_repos(self, repos: Optional[list[str]]) -> None:
        """Set which repositories to search.

        Args:
            repos: List of repo names to search, or None for all
        """
        self.active_repos = repos

    def ask(self, question: str, k: int = 5) -> tuple[str, list]:
        """Ask a question about the codebase.

        Args:
            question: User's question
            k: Number of code chunks to retrieve

        Returns:
            Tuple of (answer string, list of retrieved chunks)
        """
        # Retrieve relevant code
        chunks = self.retriever.retrieve(question, k=k, repos=self.active_repos)
        context = self.retriever.format_context(chunks)

        # Build messages
        messages = [SystemMessage(content=SYSTEM_PROMPT)]

        # Add conversation history
        messages.extend(self.history)

        # Add current question with context
        prompt = format_prompt(context, question)
        messages.append(HumanMessage(content=prompt))

        # Get response
        response = self.llm.invoke(messages)
        answer = response.content

        # Save to history
        self.history.append(HumanMessage(content=question))
        self.history.append(AIMessage(content=answer))

        return answer, chunks

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.history = []

    def get_history(self) -> list[dict]:
        """Get conversation history as a list of messages.

        Returns:
            List of message dictionaries with 'role' and 'content'
        """
        messages = []
        for msg in self.history:
            if isinstance(msg, HumanMessage):
                messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                messages.append({"role": "assistant", "content": msg.content})
        return messages
