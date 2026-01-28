"""Streamlit web UI for the RAG codebase assistant."""

import os
import shutil
from pathlib import Path

import streamlit as st

from config import REPOS_DIR, INDEXES_DIR, DEFAULT_TOP_K
from src.ingestion import clone_repo, RepoManager, get_repo_name
from src.parsing import CodeChunker
from src.retrieval import CodeRetriever
from src.conversation import CodeAssistantChain

# Ensure data directories exist
REPOS_DIR.mkdir(parents=True, exist_ok=True)
INDEXES_DIR.mkdir(parents=True, exist_ok=True)


@st.cache_resource
def get_retriever() -> CodeRetriever:
    """Get or create the code retriever (cached)."""
    retriever = CodeRetriever()
    retriever.load()
    return retriever


@st.cache_resource
def get_repo_manager() -> RepoManager:
    """Get or create the repo manager (cached)."""
    return RepoManager()


def index_repository(url: str, retriever: CodeRetriever, repo_manager: RepoManager) -> str:
    """Clone and index a repository.

    Returns status message.
    """
    try:
        # Clone repository
        repo_name = get_repo_name(url)
        st.info(f"Cloning {repo_name}...")
        repo_path = clone_repo(url, REPOS_DIR)

        # Parse and chunk code
        st.info("Parsing Python files...")
        chunker = CodeChunker()
        chunks = list(chunker.chunk_repository(repo_path, repo_name))
        file_count = chunker.count_python_files(repo_path)

        if not chunks:
            return f"No Python code found in {repo_name}"

        # Remove old chunks if re-indexing
        retriever.vector_store.remove_repo(repo_name)

        # Add to vector store
        st.info(f"Indexing {len(chunks)} code chunks...")
        retriever.add_chunks(chunks)
        retriever.save()

        # Update metadata
        repo_manager.add_repo(
            url=url,
            name=repo_name,
            local_path=repo_path,
            file_count=file_count,
            chunk_count=len(chunks)
        )

        return f"Successfully indexed {repo_name}: {file_count} files, {len(chunks)} chunks"

    except Exception as e:
        return f"Error indexing repository: {e}"


def delete_repository(repo_name: str, retriever: CodeRetriever, repo_manager: RepoManager) -> str:
    """Delete a repository and its index."""
    try:
        # Remove from vector store
        removed = retriever.vector_store.remove_repo(repo_name)
        retriever.save()

        # Remove local files
        repo_path = REPOS_DIR / repo_name
        if repo_path.exists():
            shutil.rmtree(repo_path)

        # Remove from metadata
        repo_manager.remove_repo(repo_name)

        return f"Deleted {repo_name} ({removed} chunks removed)"

    except Exception as e:
        return f"Error deleting repository: {e}"


def main():
    st.set_page_config(
        page_title="Codebase Assistant",
        page_icon="🔍",
        layout="wide"
    )

    st.title("🔍 RAG Codebase Assistant")
    st.caption("Ask questions about your indexed GitHub repositories")

    # Initialize components
    retriever = get_retriever()
    repo_manager = get_repo_manager()

    # Sidebar
    with st.sidebar:
        st.header("📦 Repositories")

        # Add repository section
        st.subheader("Add Repository")
        repo_url = st.text_input(
            "GitHub URL",
            placeholder="https://github.com/owner/repo"
        )

        if st.button("Index Repository", type="primary"):
            if repo_url:
                with st.spinner("Indexing..."):
                    result = index_repository(repo_url, retriever, repo_manager)
                    if "Successfully" in result:
                        st.success(result)
                        st.cache_resource.clear()
                    else:
                        st.error(result)
            else:
                st.warning("Please enter a repository URL")

        st.divider()

        # Listed indexed repos
        st.subheader("Indexed Repositories")
        repos = repo_manager.list_repos()

        if not repos:
            st.info("No repositories indexed yet")
        else:
            # Multi-select for active repos
            repo_names = [r.name for r in repos]
            active_repos = st.multiselect(
                "Select repos to search",
                repo_names,
                default=repo_names,
                help="Only selected repos will be searched"
            )

            # Store in session state
            st.session_state["active_repos"] = active_repos if active_repos else None

            st.divider()

            # Display each repo with delete option
            for repo in repos:
                with st.expander(f"📁 {repo.name}"):
                    st.caption(f"URL: {repo.url}")
                    st.caption(f"Files: {repo.file_count}")
                    st.caption(f"Chunks: {repo.chunk_count}")
                    st.caption(f"Indexed: {repo.last_indexed[:10]}")

                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("🔄 Re-index", key=f"reindex_{repo.name}"):
                            with st.spinner("Re-indexing..."):
                                result = index_repository(repo.url, retriever, repo_manager)
                                if "Successfully" in result:
                                    st.success(result)
                                    st.cache_resource.clear()
                                else:
                                    st.error(result)

                    with col2:
                        if st.button("🗑️ Delete", key=f"delete_{repo.name}"):
                            result = delete_repository(repo.name, retriever, repo_manager)
                            if "Deleted" in result:
                                st.success(result)
                                st.cache_resource.clear()
                                st.rerun()
                            else:
                                st.error(result)

        st.divider()

        # Settings
        st.subheader("⚙️ Settings")
        top_k = st.slider(
            "Results to retrieve",
            min_value=1,
            max_value=10,
            value=DEFAULT_TOP_K,
            help="Number of code snippets to retrieve per query"
        )
        st.session_state["top_k"] = top_k

        if st.button("Clear Chat History"):
            st.session_state["messages"] = []
            st.session_state["chain"] = None
            st.rerun()

    # Main chat area
    if retriever.vector_store.total_chunks == 0:
        st.info("👈 Add a GitHub repository to get started")
        return

    # Initialize chat state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if "chain" not in st.session_state or st.session_state["chain"] is None:
        try:
            chain = CodeAssistantChain(retriever)
            st.session_state["chain"] = chain
        except Exception as e:
            st.error(f"Could not initialize chat: {e}")
            st.info("Make sure OPENAI_API_KEY is set")
            return

    chain = st.session_state["chain"]
    chain.set_active_repos(st.session_state.get("active_repos"))

    # Display chat history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # Show retrieved chunks for assistant messages
            if msg["role"] == "assistant" and "chunks" in msg:
                with st.expander("📄 Retrieved Code Snippets"):
                    for chunk in msg["chunks"]:
                        st.caption(f"**{chunk.type}**: `{chunk.name}`")
                        st.caption(f"📍 {chunk.repo_name}/{chunk.file_path}:{chunk.line_start}")
                        st.code(chunk.code, language="python")
                        st.divider()

    # Chat input
    if question := st.chat_input("Ask a question about the code..."):
        # Add user message
        st.session_state["messages"].append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.markdown(question)

        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    top_k = st.session_state.get("top_k", DEFAULT_TOP_K)
                    answer, chunks = chain.ask(question, k=top_k)

                    st.markdown(answer)

                    # Show retrieved chunks
                    if chunks:
                        with st.expander("📄 Retrieved Code Snippets"):
                            for chunk in chunks:
                                st.caption(f"**{chunk.type}**: `{chunk.name}`")
                                st.caption(f"📍 {chunk.repo_name}/{chunk.file_path}:{chunk.line_start}")
                                st.code(chunk.code, language="python")
                                st.divider()

                    # Save assistant message
                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": answer,
                        "chunks": chunks
                    })

                except Exception as e:
                    st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
