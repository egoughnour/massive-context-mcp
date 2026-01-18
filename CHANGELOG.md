# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2025-01-18

Initial release. Based on the [Recursive Language Model pattern](https://arxiv.org/html/2512.24601v1), inspired by [richardwhiteii/rlm](https://github.com/richardwhiteii/rlm).

### Added

- **Core Context Tools**
  - `rlm_load_context` - Load large contexts as external variables
  - `rlm_inspect_context` - Get structure info without loading into prompt
  - `rlm_chunk_context` - Chunk by lines/chars/paragraphs
  - `rlm_get_chunk` - Retrieve specific chunks
  - `rlm_filter_context` - Filter with regex (keep/remove matching lines)
  - `rlm_sub_query` - Make sub-LLM calls on chunks
  - `rlm_sub_query_batch` - Process multiple chunks in parallel
  - `rlm_store_result` - Store sub-call results for aggregation
  - `rlm_get_results` - Retrieve stored results
  - `rlm_list_contexts` - List all loaded contexts
  - `rlm_auto_analyze` - One-step analysis with auto-detection
  - `rlm_exec` - Execute Python code against loaded context (sandboxed)

- **Ollama Integration**
  - `rlm_ollama_status` - Check Ollama availability and models
  - `rlm_system_check` - Check system requirements for Ollama
  - `rlm_setup_ollama` - Install Ollama via Homebrew (macOS)
  - `rlm_setup_ollama_direct` - Install Ollama via direct download (no sudo required)
  - Auto-provider detection: prefers Ollama when available, falls back to Claude SDK
  - Status caching with 60-second TTL

- **Provider Support**
  - Ollama provider with gemma3:12b default model
  - Claude SDK provider with claude-haiku-4-5 default model
  - `auto` provider that selects best available option

- **MCPB Bundle Support**
  - MCPB manifest v0.4 with UV runtime
  - User configuration for data directory and Ollama URL
  - Environment variable mapping

- **Testing**
  - 69 unit tests covering all tools
  - pytest-asyncio for async test support
  - Coverage reporting

- **CI/CD**
  - GitHub Actions workflow for testing (Ubuntu, macOS, Python 3.10-3.12)
  - GitHub Actions workflow for release (MCPB packing, PyPI publishing)
  - Linting with Ruff
  - Type checking with mypy

### Technical Details

- Based on Recursive Language Model paper: https://arxiv.org/html/2512.24601v1
- Handles 10M+ token contexts by treating them as external variables
- Contexts persist to disk for reuse across sessions
- Sandboxed Python execution for programmatic analysis

[Unreleased]: https://github.com/egoughnour/massive-context-mcp/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/egoughnour/massive-context-mcp/releases/tag/v0.1.0
