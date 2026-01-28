"""Python code parsing using tree-sitter."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import tree_sitter_python as tspython
from tree_sitter import Language, Parser


@dataclass
class CodeChunk:
    """A parsed chunk of code."""
    type: str  # function, class, method
    name: str
    code: str
    docstring: Optional[str]
    file_path: str
    line_start: int
    line_end: int
    repo_name: str

    def to_text(self) -> str:
        """Convert chunk to text for embedding."""
        parts = [f"{self.type}: {self.name}"]
        if self.docstring:
            parts.append(f"Docstring: {self.docstring}")
        parts.append(f"File: {self.file_path}")
        parts.append(f"Code:\n{self.code}")
        return "\n".join(parts)


class PythonParser:
    """Parser for Python source files using tree-sitter."""

    def __init__(self):
        self.language = Language(tspython.language())
        self.parser = Parser(self.language)

    def parse_file(self, file_path: Path, repo_name: str) -> list[CodeChunk]:
        """Parse a Python file and extract code chunks.

        Args:
            file_path: Path to the Python file
            repo_name: Name of the parent repository

        Returns:
            List of extracted code chunks
        """
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            source_code = f.read()

        return self.parse_source(source_code, str(file_path), repo_name)

    def parse_source(self, source: str, file_path: str, repo_name: str) -> list[CodeChunk]:
        """Parse Python source code and extract code chunks.

        Args:
            source: Python source code
            file_path: Path to display in chunks
            repo_name: Name of the parent repository

        Returns:
            List of extracted code chunks
        """
        tree = self.parser.parse(bytes(source, "utf-8"))
        chunks = []

        # Walk the tree to find functions and classes
        self._extract_chunks(tree.root_node, source, file_path, repo_name, chunks)

        return chunks

    def _extract_chunks(
        self,
        node,
        source: str,
        file_path: str,
        repo_name: str,
        chunks: list[CodeChunk],
        parent_class: Optional[str] = None
    ) -> None:
        """Recursively extract code chunks from AST nodes."""

        if node.type == "function_definition":
            chunk = self._extract_function(node, source, file_path, repo_name, parent_class)
            if chunk:
                chunks.append(chunk)

        elif node.type == "class_definition":
            # Extract class itself
            chunk = self._extract_class(node, source, file_path, repo_name)
            if chunk:
                chunks.append(chunk)

            # Extract methods within the class
            class_name = self._get_node_name(node)
            for child in node.children:
                if child.type == "block":
                    for block_child in child.children:
                        self._extract_chunks(
                            block_child, source, file_path, repo_name, chunks, class_name
                        )
            return  # Don't recurse further for classes

        # Recurse into children
        for child in node.children:
            self._extract_chunks(child, source, file_path, repo_name, chunks, parent_class)

    def _extract_function(
        self,
        node,
        source: str,
        file_path: str,
        repo_name: str,
        parent_class: Optional[str]
    ) -> Optional[CodeChunk]:
        """Extract a function definition."""
        name = self._get_node_name(node)
        if not name:
            return None

        code = self._get_node_text(node, source)
        docstring = self._get_docstring(node, source)

        chunk_type = "method" if parent_class else "function"
        full_name = f"{parent_class}.{name}" if parent_class else name

        return CodeChunk(
            type=chunk_type,
            name=full_name,
            code=code,
            docstring=docstring,
            file_path=file_path,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            repo_name=repo_name
        )

    def _extract_class(
        self,
        node,
        source: str,
        file_path: str,
        repo_name: str
    ) -> Optional[CodeChunk]:
        """Extract a class definition (without method bodies for overview)."""
        name = self._get_node_name(node)
        if not name:
            return None

        code = self._get_node_text(node, source)
        docstring = self._get_docstring(node, source)

        return CodeChunk(
            type="class",
            name=name,
            code=code,
            docstring=docstring,
            file_path=file_path,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            repo_name=repo_name
        )

    def _get_node_name(self, node) -> Optional[str]:
        """Get the name identifier from a function or class node."""
        for child in node.children:
            if child.type == "identifier":
                return child.text.decode("utf-8")
        return None

    def _get_node_text(self, node, source: str) -> str:
        """Get the full text of a node."""
        return source[node.start_byte:node.end_byte]

    def _get_docstring(self, node, source: str) -> Optional[str]:
        """Extract docstring from a function or class node."""
        for child in node.children:
            if child.type == "block":
                for block_child in child.children:
                    if block_child.type == "expression_statement":
                        for expr_child in block_child.children:
                            if expr_child.type == "string":
                                docstring = expr_child.text.decode("utf-8")
                                # Remove quotes
                                if docstring.startswith('"""') or docstring.startswith("'''"):
                                    return docstring[3:-3].strip()
                                elif docstring.startswith('"') or docstring.startswith("'"):
                                    return docstring[1:-1].strip()
                        break
                break
        return None
