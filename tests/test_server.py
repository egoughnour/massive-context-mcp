"""Tests for the RLM MCP server."""

import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

# Set test data directory before importing server
TEST_DATA_DIR = tempfile.mkdtemp(prefix="rlm_test_")
os.environ["RLM_DATA_DIR"] = TEST_DATA_DIR

import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rlm_mcp_server import (
    _chunk_content,
    _hash_content,
    contexts,
)

# Import FastMCP tool wrappers and extract underlying functions
from rlm_mcp_server import (
    rlm_chunk_context as _rlm_chunk_context_tool,
    rlm_filter_context as _rlm_filter_context_tool,
    rlm_get_chunk as _rlm_get_chunk_tool,
    rlm_get_results as _rlm_get_results_tool,
    rlm_inspect_context as _rlm_inspect_context_tool,
    rlm_list_contexts as _rlm_list_contexts_tool,
    rlm_load_context as _rlm_load_context_tool,
    rlm_store_result as _rlm_store_result_tool,
    rlm_sub_query as _rlm_sub_query_tool,
    rlm_sub_query_batch as _rlm_sub_query_batch_tool,
)

# FastMCP wraps functions in FunctionTool objects - extract the underlying fn
rlm_chunk_context = _rlm_chunk_context_tool.fn
rlm_filter_context = _rlm_filter_context_tool.fn
rlm_get_chunk = _rlm_get_chunk_tool.fn
rlm_get_results = _rlm_get_results_tool.fn
rlm_inspect_context = _rlm_inspect_context_tool.fn
rlm_list_contexts = _rlm_list_contexts_tool.fn
rlm_load_context = _rlm_load_context_tool.fn
rlm_store_result = _rlm_store_result_tool.fn
rlm_sub_query = _rlm_sub_query_tool.fn
rlm_sub_query_batch = _rlm_sub_query_batch_tool.fn


@pytest.fixture(autouse=True)
def clear_contexts():
    """Clear contexts before each test."""
    contexts.clear()
    yield
    contexts.clear()


class TestPureFunctions:
    """Tests for pure helper functions."""

    def test_hash_content_deterministic(self):
        """Hash should be deterministic for same content."""
        content = "hello world"
        assert _hash_content(content) == _hash_content(content)

    def test_hash_content_different_for_different_content(self):
        """Different content should produce different hashes."""
        assert _hash_content("hello") != _hash_content("world")

    def test_hash_content_length(self):
        """Hash should be 12 characters (truncated SHA256)."""
        assert len(_hash_content("test")) == 12


class TestChunkContent:
    """Tests for the _chunk_content function."""

    def test_chunk_by_lines(self):
        """Chunking by lines should split on newlines."""
        content = "line1\nline2\nline3\nline4\nline5"
        chunks = _chunk_content(content, "lines", 2)
        assert len(chunks) == 3
        assert chunks[0] == "line1\nline2"
        assert chunks[1] == "line3\nline4"
        assert chunks[2] == "line5"

    def test_chunk_by_chars(self):
        """Chunking by chars should split on character count."""
        content = "abcdefghij"
        chunks = _chunk_content(content, "chars", 3)
        assert len(chunks) == 4
        assert chunks[0] == "abc"
        assert chunks[1] == "def"
        assert chunks[2] == "ghi"
        assert chunks[3] == "j"

    def test_chunk_by_paragraphs(self):
        """Chunking by paragraphs should split on blank lines."""
        content = "para1\n\npara2\n\npara3\n\npara4"
        chunks = _chunk_content(content, "paragraphs", 2)
        assert len(chunks) == 2
        assert chunks[0] == "para1\n\npara2"
        assert chunks[1] == "para3\n\npara4"

    def test_chunk_empty_content(self):
        """Empty content should return single empty chunk."""
        chunks = _chunk_content("", "lines", 10)
        assert len(chunks) == 1
        assert chunks[0] == ""

    def test_chunk_unknown_strategy(self):
        """Unknown strategy should return empty list."""
        chunks = _chunk_content("test", "unknown", 10)
        assert chunks == []


class TestLoadContext:
    """Tests for context loading."""

    @pytest.mark.asyncio
    async def test_load_context_success(self):
        """Loading a context should store it and return metadata."""
        result = await rlm_load_context(name="test_ctx", content="hello\nworld")

        assert result["status"] == "loaded"
        assert result["name"] == "test_ctx"
        assert result["length"] == 11
        assert result["lines"] == 2
        assert "hash" in result

    @pytest.mark.asyncio
    async def test_load_context_stored_in_memory(self):
        """Loaded context should be accessible in memory."""
        await rlm_load_context(name="mem_test", content="test content")

        assert "mem_test" in contexts
        assert contexts["mem_test"]["content"] == "test content"


class TestInspectContext:
    """Tests for context inspection."""

    @pytest.mark.asyncio
    async def test_inspect_context_success(self):
        """Inspecting a loaded context should return info."""
        await rlm_load_context(name="inspect_test", content="hello world " * 100)

        result = await rlm_inspect_context(name="inspect_test", preview_chars=20)

        assert result["name"] == "inspect_test"
        assert len(result["preview"]) == 20
        assert result["has_chunks"] is False

    @pytest.mark.asyncio
    async def test_inspect_nonexistent_context(self):
        """Inspecting nonexistent context should return error."""
        result = await rlm_inspect_context(name="nonexistent")
        assert "error" in result or "not found" in str(result).lower()


class TestChunkContext:
    """Tests for context chunking."""

    @pytest.mark.asyncio
    async def test_chunk_context_success(self):
        """Chunking a context should create chunks."""
        await rlm_load_context(
            name="chunk_test",
            content="\n".join([f"line{i}" for i in range(10)]),
        )

        result = await rlm_chunk_context(name="chunk_test", strategy="lines", size=3)

        assert result["status"] == "chunked"
        assert result["chunk_count"] == 4
        assert len(result["chunks"]) == 4

    @pytest.mark.asyncio
    async def test_chunk_nonexistent_context(self):
        """Chunking nonexistent context should return error."""
        result = await rlm_chunk_context(name="nonexistent")
        assert "error" in result or "not found" in str(result).lower()


class TestGetChunk:
    """Tests for chunk retrieval."""

    @pytest.mark.asyncio
    async def test_get_chunk_success(self):
        """Getting a chunk should return its content."""
        await rlm_load_context(
            name="get_chunk_test",
            content="chunk0\nchunk0\nchunk1\nchunk1",
        )
        await rlm_chunk_context(name="get_chunk_test", strategy="lines", size=2)

        result = await rlm_get_chunk(name="get_chunk_test", chunk_index=0)
        # Result is either the chunk content string or a dict with content
        assert "chunk0" in str(result)

    @pytest.mark.asyncio
    async def test_get_chunk_out_of_range(self):
        """Getting chunk out of range should return error."""
        await rlm_load_context(name="range_test", content="test")
        await rlm_chunk_context(name="range_test", strategy="lines", size=10)

        result = await rlm_get_chunk(name="range_test", chunk_index=99)
        assert "error" in result or "out of range" in str(result).lower()


class TestFilterContext:
    """Tests for context filtering."""

    @pytest.mark.asyncio
    async def test_filter_keep_mode(self):
        """Filter with keep mode should keep matching lines."""
        await rlm_load_context(
            name="filter_src",
            content="error: something\ninfo: data\nerror: else",
        )

        result = await rlm_filter_context(
            name="filter_src",
            output_name="errors_only",
            pattern="error:",
            mode="keep",
        )

        assert result["filtered_lines"] == 2
        assert "errors_only" in contexts

    @pytest.mark.asyncio
    async def test_filter_remove_mode(self):
        """Filter with remove mode should remove matching lines."""
        await rlm_load_context(
            name="filter_src2",
            content="error: something\ninfo: data\nerror: else",
        )

        result = await rlm_filter_context(
            name="filter_src2",
            output_name="no_errors",
            pattern="error:",
            mode="remove",
        )

        assert result["filtered_lines"] == 1


class TestStoreAndGetResults:
    """Tests for result storage and retrieval."""

    @pytest.mark.asyncio
    async def test_store_result(self):
        """Storing a result should succeed."""
        result = await rlm_store_result(
            name="test_results",
            result="found something",
            metadata={"chunk": 0},
        )
        assert "stored" in str(result).lower() or result.get("status") == "stored"

    @pytest.mark.asyncio
    async def test_get_results(self):
        """Getting stored results should return all results."""
        await rlm_store_result(name="multi_results", result="result1")
        await rlm_store_result(name="multi_results", result="result2")

        result = await rlm_get_results(name="multi_results")

        assert result["count"] == 2
        assert len(result["results"]) == 2

    @pytest.mark.asyncio
    async def test_get_nonexistent_results(self):
        """Getting nonexistent results should return error."""
        result = await rlm_get_results(name="nonexistent")
        assert "error" in result or "No results" in str(result)


class TestListContexts:
    """Tests for listing contexts."""

    @pytest.mark.asyncio
    async def test_list_contexts_empty(self):
        """Listing with no contexts should return empty list."""
        result = await rlm_list_contexts()
        # May include disk-only contexts, so just check structure
        assert "contexts" in result

    @pytest.mark.asyncio
    async def test_list_contexts_with_data(self):
        """Listing should include loaded contexts."""
        await rlm_load_context(name="list_test1", content="a")
        await rlm_load_context(name="list_test2", content="b")

        result = await rlm_list_contexts()

        names = [c["name"] for c in result["contexts"]]
        assert "list_test1" in names
        assert "list_test2" in names


class TestSubQuery:
    """Tests for sub-query handler."""

    @pytest.mark.asyncio
    async def test_sub_query_context_not_found(self):
        """Sub-query on nonexistent context should error."""
        result = await rlm_sub_query(query="test", context_name="nonexistent")
        assert result["error"] == "context_not_found"

    @pytest.mark.asyncio
    async def test_sub_query_chunk_not_available(self):
        """Sub-query on non-chunked context with chunk_index should error."""
        await rlm_load_context(name="no_chunks", content="test")

        result = await rlm_sub_query(
            query="test",
            context_name="no_chunks",
            chunk_index=0,
        )
        assert result["error"] == "chunk_not_available"

    @pytest.mark.asyncio
    async def test_sub_query_with_mock_provider(self):
        """Sub-query should call provider and return response."""
        await rlm_load_context(name="query_test", content="test content")

        with patch("rlm_mcp_server._make_provider_call", new_callable=AsyncMock) as mock:
            mock.return_value = ("mocked response", None)

            result = await rlm_sub_query(
                query="what is this?",
                context_name="query_test",
                provider="ollama",
            )

            assert result["response"] == "mocked response"
            mock.assert_called_once()


class TestSubQueryBatch:
    """Tests for batch sub-query handler."""

    @pytest.mark.asyncio
    async def test_batch_query_context_not_found(self):
        """Batch query on nonexistent context should error."""
        result = await rlm_sub_query_batch(
            query="test",
            context_name="nonexistent",
            chunk_indices=[0, 1],
        )
        assert result["error"] == "context_not_found"

    @pytest.mark.asyncio
    async def test_batch_query_not_chunked(self):
        """Batch query on non-chunked context should error."""
        await rlm_load_context(name="not_chunked", content="test")

        result = await rlm_sub_query_batch(
            query="test",
            context_name="not_chunked",
            chunk_indices=[0],
        )
        assert result["error"] == "context_not_chunked"

    @pytest.mark.asyncio
    async def test_batch_query_invalid_indices(self):
        """Batch query with invalid indices should error."""
        await rlm_load_context(name="batch_test", content="line1\nline2")
        await rlm_chunk_context(name="batch_test", strategy="lines", size=1)

        result = await rlm_sub_query_batch(
            query="test",
            context_name="batch_test",
            chunk_indices=[0, 99],
        )
        assert result["error"] == "invalid_chunk_indices"

    @pytest.mark.asyncio
    async def test_batch_query_with_mock_provider(self):
        """Batch query should process all chunks."""
        await rlm_load_context(name="batch_success", content="chunk1\nchunk2\nchunk3")
        await rlm_chunk_context(name="batch_success", strategy="lines", size=1)

        with patch("rlm_mcp_server._make_provider_call", new_callable=AsyncMock) as mock:
            mock.return_value = ("response", None)

            result = await rlm_sub_query_batch(
                query="analyze",
                context_name="batch_success",
                chunk_indices=[0, 1, 2],
                concurrency=2,
            )

            assert result["status"] == "completed"
            assert result["total_chunks"] == 3
            assert result["successful"] == 3
            assert result["failed"] == 0
            assert mock.call_count == 3
