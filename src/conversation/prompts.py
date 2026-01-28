"""System prompts and templates for the code assistant."""

SYSTEM_PROMPT = """You are a helpful code assistant that answers questions about codebases.
You have access to code snippets retrieved from indexed repositories.

When answering questions:
1. Base your answers on the provided code context when available
2. Reference specific functions, classes, and file locations when relevant
3. If the code context doesn't contain enough information to answer, say so
4. Provide clear, concise explanations with code examples when helpful
5. If asked about code not in the context, explain what you can see and suggest what might be needed

Always cite the source file and line numbers when referencing specific code."""


CONTEXT_TEMPLATE = """## Retrieved Code Context

The following code snippets were found relevant to the user's question:

{context}

---

## User Question

{question}"""


def format_prompt(context: str, question: str) -> str:
    """Format the context and question into a prompt.

    Args:
        context: Formatted code context from retriever
        question: User's question

    Returns:
        Complete prompt for the LLM
    """
    return CONTEXT_TEMPLATE.format(context=context, question=question)
