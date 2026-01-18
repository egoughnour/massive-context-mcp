#!/usr/bin/env python3
"""
RLM MCP Server - Recursive Language Model patterns for massive context handling.

Implements the core insight from https://arxiv.org/html/2512.24601v1:
Treat context as external variable, chunk programmatically, sub-call recursively.

Refactored to use FastMCP for cleaner tool definitions and PyPI distribution.
"""

import asyncio
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

from fastmcp import FastMCP

try:
    import httpx

    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

try:
    from claude_agent_sdk import ClaudeAgentOptions
    from claude_agent_sdk import query as claude_query

    HAS_CLAUDE_SDK = True
except ImportError:
    HAS_CLAUDE_SDK = False

# Storage directories
DATA_DIR = Path(os.environ.get("RLM_DATA_DIR", "/tmp/rlm"))
CONTEXTS_DIR = DATA_DIR / "contexts"
CHUNKS_DIR = DATA_DIR / "chunks"
RESULTS_DIR = DATA_DIR / "results"

for directory in [CONTEXTS_DIR, CHUNKS_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# In-memory context storage (also persisted to disk)
contexts: dict[str, dict] = {}

# Initialize FastMCP server
mcp = FastMCP("massive-context-mcp")

# Default models per provider
DEFAULT_MODELS = {
    "ollama": "gemma3:12b",
    "claude-sdk": "claude-haiku-4-5-20251101",
}

# Ollama availability cache
_ollama_status_cache: dict[str, Any] = {
    "checked_at": None,
    "running": False,
    "models": [],
    "default_model_available": False,
    "ttl_seconds": 60,  # Re-check every 60 seconds
}

# Minimum RAM required for gemma3:12b (model needs ~8GB, system needs headroom)
MIN_RAM_GB = 16
GEMMA3_12B_RAM_GB = 8


# =============================================================================
# Helper Functions
# =============================================================================


def _check_system_requirements() -> dict:
    """Check if the system meets requirements for running Ollama with gemma3:12b."""
    import platform

    result = {
        "platform": platform.system(),
        "machine": platform.machine(),
        "is_macos": False,
        "is_apple_silicon": False,
        "ram_gb": 0,
        "ram_sufficient": False,
        "homebrew_installed": False,
        "ollama_installed": False,
        "meets_requirements": False,
        "issues": [],
        "recommendations": [],
    }

    # Check macOS
    if platform.system() == "Darwin":
        result["is_macos"] = True
    else:
        result["issues"].append(f"Not macOS (detected: {platform.system()})")
        result["recommendations"].append("Ollama auto-setup is only supported on macOS")

    # Check Apple Silicon (M1, M2, M3, M4)
    machine = platform.machine()
    if machine == "arm64":
        result["is_apple_silicon"] = True
        # Try to get specific chip info
        try:
            chip_info = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if chip_info.returncode == 0:
                result["chip"] = chip_info.stdout.strip()
        except Exception:
            result["chip"] = "Apple Silicon (arm64)"
    else:
        result["issues"].append(f"Not Apple Silicon (detected: {machine})")
        result["recommendations"].append("Apple Silicon (M1/M2/M3/M4) recommended for optimal Ollama performance")

    # Check RAM
    try:
        if platform.system() == "Darwin":
            mem_info = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if mem_info.returncode == 0:
                ram_bytes = int(mem_info.stdout.strip())
                ram_gb = ram_bytes / (1024**3)
                result["ram_gb"] = round(ram_gb, 1)
                result["ram_sufficient"] = ram_gb >= MIN_RAM_GB

                if not result["ram_sufficient"]:
                    result["issues"].append(
                        f"Insufficient RAM: {result['ram_gb']}GB (need {MIN_RAM_GB}GB+ for gemma3:12b)"
                    )
                    result["recommendations"].append(
                        f"gemma3:12b requires ~{GEMMA3_12B_RAM_GB}GB RAM. "
                        f"With {result['ram_gb']}GB total, consider using a smaller model."
                    )
    except Exception as e:
        result["issues"].append(f"Could not determine RAM: {e}")

    # Check Homebrew
    try:
        brew_check = subprocess.run(
            ["which", "brew"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        result["homebrew_installed"] = brew_check.returncode == 0
        if result["homebrew_installed"]:
            result["homebrew_path"] = brew_check.stdout.strip()
        else:
            result["issues"].append("Homebrew not installed")
            result["recommendations"].append(
                'Install Homebrew first: /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
            )
    except Exception:
        result["issues"].append("Could not check for Homebrew")

    # Check if Ollama is already installed
    try:
        ollama_check = subprocess.run(
            ["which", "ollama"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        result["ollama_installed"] = ollama_check.returncode == 0
        if result["ollama_installed"]:
            result["ollama_path"] = ollama_check.stdout.strip()
            # Get version
            try:
                version_check = subprocess.run(
                    ["ollama", "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if version_check.returncode == 0:
                    result["ollama_version"] = version_check.stdout.strip()
            except Exception:
                pass
    except Exception:
        pass

    # Determine if all requirements are met
    result["meets_requirements"] = (
        result["is_macos"] and result["is_apple_silicon"] and result["ram_sufficient"] and result["homebrew_installed"]
    )

    return result


async def _setup_ollama_direct(
    install: bool = False,
    start_service: bool = False,
    pull_model: bool = False,
    model: str = "gemma3:12b",
) -> dict:
    """Setup Ollama via direct download - no Homebrew, no sudo, fully headless."""
    import shutil

    result = {
        "method": "direct_download",
        "actions_taken": [],
        "actions_skipped": [],
        "errors": [],
        "warnings": [],
        "success": True,
    }

    # Check basic system requirements (macOS, Apple Silicon, RAM)
    sys_check = _check_system_requirements()
    result["system_check"] = {
        "is_macos": sys_check["is_macos"],
        "is_apple_silicon": sys_check["is_apple_silicon"],
        "ram_gb": sys_check["ram_gb"],
        "ram_sufficient": sys_check["ram_sufficient"],
    }

    if not sys_check["is_macos"]:
        result["errors"].append("Direct download setup only supported on macOS")
        result["success"] = False
        return result

    # Define paths
    home = Path.home()
    install_dir = home / "Applications"
    app_path = install_dir / "Ollama.app"
    cli_path = app_path / "Contents" / "Resources" / "ollama"

    # Install Ollama via direct download
    if install:
        if app_path.exists():
            result["actions_skipped"].append(f"Ollama already installed at {app_path}")
        else:
            try:
                # Create ~/Applications if needed
                install_dir.mkdir(parents=True, exist_ok=True)

                # Download URL
                download_url = "https://ollama.com/download/Ollama-darwin.zip"
                zip_path = Path("/tmp/Ollama-darwin.zip")
                extract_dir = Path("/tmp/ollama-extract")

                result["actions_taken"].append(f"Downloading from {download_url}...")

                # Download using curl (available on all macOS)
                download_proc = subprocess.run(
                    ["curl", "-L", "-o", str(zip_path), download_url],
                    capture_output=True,
                    text=True,
                    timeout=600,  # 10 minute timeout for download
                )

                if download_proc.returncode != 0:
                    result["errors"].append(f"Download failed: {download_proc.stderr}")
                    result["success"] = False
                    return result

                result["actions_taken"].append("Download complete")

                # Clean up any previous extraction
                if extract_dir.exists():
                    shutil.rmtree(extract_dir)
                extract_dir.mkdir(parents=True, exist_ok=True)

                # Extract
                result["actions_taken"].append("Extracting...")
                extract_proc = subprocess.run(
                    ["unzip", "-q", str(zip_path), "-d", str(extract_dir)],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )

                if extract_proc.returncode != 0:
                    result["errors"].append(f"Extraction failed: {extract_proc.stderr}")
                    result["success"] = False
                    return result

                # Move to ~/Applications
                extracted_app = extract_dir / "Ollama.app"
                if not extracted_app.exists():
                    # Try to find it
                    for item in extract_dir.iterdir():
                        if item.name == "Ollama.app" or item.suffix == ".app":
                            extracted_app = item
                            break

                if extracted_app.exists():
                    shutil.move(str(extracted_app), str(app_path))
                    result["actions_taken"].append(f"Installed to {app_path}")
                else:
                    result["errors"].append("Could not find Ollama.app in extracted contents")
                    result["success"] = False
                    return result

                # Clean up
                zip_path.unlink(missing_ok=True)
                shutil.rmtree(extract_dir, ignore_errors=True)

                # Note about PATH
                result["path_setup"] = {
                    "cli_path": str(cli_path),
                    "add_to_path": f'export PATH="{cli_path.parent}:$PATH"',
                    "shell_config": "Add the above line to ~/.zshrc or ~/.bashrc",
                }

            except subprocess.TimeoutExpired:
                result["errors"].append("Download timed out (10 min limit)")
                result["success"] = False
            except Exception as e:
                result["errors"].append(f"Installation error: {e}")
                result["success"] = False

    # Start Ollama service
    if start_service and result["success"]:
        # Check if CLI exists
        effective_cli = None
        if cli_path.exists():
            effective_cli = cli_path
        else:
            # Check if ollama is in PATH
            which_proc = subprocess.run(["which", "ollama"], capture_output=True, text=True)
            if which_proc.returncode == 0:
                effective_cli = Path(which_proc.stdout.strip())

        if not effective_cli:
            result["errors"].append(
                f"Ollama CLI not found. Expected at {cli_path} or in PATH. You may need to add it to your PATH first."
            )
            result["success"] = False
        else:
            # Check if already running
            status = await _check_ollama_status(force_refresh=True)
            if status.get("running"):
                result["actions_skipped"].append("Ollama service already running")
            else:
                try:
                    # Start ollama serve in background
                    # Using nohup to detach from terminal
                    subprocess.Popen(
                        ["nohup", str(effective_cli), "serve"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        start_new_session=True,
                    )
                    result["actions_taken"].append("Started Ollama service (ollama serve)")

                    # Wait for service to be ready
                    await asyncio.sleep(3)

                    # Verify it started
                    status = await _check_ollama_status(force_refresh=True)
                    if status.get("running"):
                        result["actions_taken"].append("Service is running")
                    else:
                        result["warnings"].append(
                            "Service may still be starting. Check with rlm_ollama_status in a few seconds."
                        )
                except Exception as e:
                    result["errors"].append(f"Failed to start service: {e}")

    # Pull model
    if pull_model and result["success"]:
        # Check RAM before pulling large model
        if model == "gemma3:12b" and not sys_check["ram_sufficient"]:
            result["errors"].append(
                f"Insufficient RAM ({sys_check['ram_gb']}GB) for {model}. "
                f"Need {MIN_RAM_GB}GB+. Consider: gemma3:4b or gemma3:1b"
            )
            result["success"] = False
        else:
            # Find CLI
            effective_cli = None
            if cli_path.exists():
                effective_cli = cli_path
            else:
                which_proc = subprocess.run(["which", "ollama"], capture_output=True, text=True)
                if which_proc.returncode == 0:
                    effective_cli = Path(which_proc.stdout.strip())

            if not effective_cli:
                result["errors"].append("Ollama CLI not found. Cannot pull model.")
                result["success"] = False
            else:
                # Check if model already exists
                status = await _check_ollama_status(force_refresh=True)
                model_base = model.split(":")[0]
                already_pulled = any(m.startswith(model_base) for m in status.get("models", []))

                if already_pulled:
                    result["actions_skipped"].append(f"Model {model} already available")
                else:
                    try:
                        result["actions_taken"].append(f"Pulling model {model} (this may take several minutes)...")
                        pull_proc = subprocess.run(
                            [str(effective_cli), "pull", model],
                            capture_output=True,
                            text=True,
                            timeout=1800,  # 30 minute timeout
                        )
                        if pull_proc.returncode == 0:
                            result["actions_taken"].append(f"Successfully pulled {model}")
                        else:
                            result["errors"].append(f"Failed to pull {model}: {pull_proc.stderr}")
                            result["success"] = False
                    except subprocess.TimeoutExpired:
                        result["errors"].append("Model pull timed out (30 min limit)")
                        result["success"] = False
                    except Exception as e:
                        result["errors"].append(f"Pull error: {e}")
                        result["success"] = False

    # Final status check
    if result["success"]:
        final_status = await _check_ollama_status(force_refresh=True)
        result["ollama_status"] = final_status

    return result


async def _setup_ollama(
    install: bool = False,
    start_service: bool = False,
    pull_model: bool = False,
    model: str = "gemma3:12b",
) -> dict:
    """Setup Ollama: install via Homebrew, start service, and pull model."""
    result = {
        "actions_taken": [],
        "actions_skipped": [],
        "errors": [],
        "success": True,
    }

    # First check system requirements
    sys_check = _check_system_requirements()
    result["system_check"] = sys_check

    if not sys_check["is_macos"]:
        result["errors"].append("Ollama auto-setup only supported on macOS")
        result["success"] = False
        return result

    if not sys_check["homebrew_installed"] and install:
        result["errors"].append(
            "Homebrew required for installation. Install with: "
            '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
        )
        result["success"] = False
        return result

    # Install Ollama via Homebrew
    if install:
        if sys_check["ollama_installed"]:
            result["actions_skipped"].append("Ollama already installed")
        else:
            try:
                install_proc = subprocess.run(
                    ["brew", "install", "ollama"],
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 minute timeout for install
                )
                if install_proc.returncode == 0:
                    result["actions_taken"].append("Installed Ollama via Homebrew")
                    sys_check["ollama_installed"] = True
                else:
                    result["errors"].append(f"Failed to install Ollama: {install_proc.stderr}")
                    result["success"] = False
            except subprocess.TimeoutExpired:
                result["errors"].append("Ollama installation timed out (5 min limit)")
                result["success"] = False
            except Exception as e:
                result["errors"].append(f"Installation error: {e}")
                result["success"] = False

    # Start Ollama service
    if start_service and result["success"]:
        if not sys_check["ollama_installed"]:
            result["errors"].append("Cannot start service: Ollama not installed")
            result["success"] = False
        else:
            try:
                # Check if already running
                status = await _check_ollama_status(force_refresh=True)
                if status.get("running"):
                    result["actions_skipped"].append("Ollama service already running")
                else:
                    # Start via brew services
                    start_proc = subprocess.run(
                        ["brew", "services", "start", "ollama"],
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    if start_proc.returncode == 0:
                        result["actions_taken"].append("Started Ollama service via Homebrew")
                        # Wait a moment for service to start
                        await asyncio.sleep(2)
                    else:
                        # Fallback: try running ollama serve in background
                        result["actions_skipped"].append("brew services failed, try: ollama serve &")
            except Exception as e:
                result["errors"].append(f"Failed to start service: {e}")

    # Pull model
    if pull_model and result["success"]:
        # Check RAM before pulling large model
        if model == "gemma3:12b" and not sys_check["ram_sufficient"]:
            result["errors"].append(
                f"Insufficient RAM ({sys_check['ram_gb']}GB) for {model}. "
                f"Need {MIN_RAM_GB}GB+. Consider: gemma3:4b or gemma3:1b"
            )
            result["success"] = False
        elif not sys_check["ollama_installed"]:
            result["errors"].append("Cannot pull model: Ollama not installed")
            result["success"] = False
        else:
            # Check if model already exists
            status = await _check_ollama_status(force_refresh=True)
            model_base = model.split(":")[0]
            already_pulled = any(m.startswith(model_base) for m in status.get("models", []))

            if already_pulled:
                result["actions_skipped"].append(f"Model {model} already available")
            else:
                try:
                    result["actions_taken"].append(f"Pulling model {model} (this may take several minutes)...")
                    pull_proc = subprocess.run(
                        ["ollama", "pull", model],
                        capture_output=True,
                        text=True,
                        timeout=1800,  # 30 minute timeout for model download
                    )
                    if pull_proc.returncode == 0:
                        result["actions_taken"].append(f"Successfully pulled {model}")
                    else:
                        result["errors"].append(f"Failed to pull {model}: {pull_proc.stderr}")
                        result["success"] = False
                except subprocess.TimeoutExpired:
                    result["errors"].append("Model pull timed out (30 min limit)")
                    result["success"] = False
                except Exception as e:
                    result["errors"].append(f"Pull error: {e}")
                    result["success"] = False

    # Final status check
    if result["success"]:
        final_status = await _check_ollama_status(force_refresh=True)
        result["ollama_status"] = final_status

    return result


async def _check_ollama_status(force_refresh: bool = False) -> dict:
    """Check Ollama server status and available models. Cached with TTL."""
    import time

    cache = _ollama_status_cache
    now = time.time()

    # Return cached result if still valid
    if not force_refresh and cache["checked_at"] is not None:
        if now - cache["checked_at"] < cache["ttl_seconds"]:
            return {
                "running": cache["running"],
                "models": cache["models"],
                "default_model_available": cache["default_model_available"],
                "cached": True,
                "checked_at": cache["checked_at"],
            }

    # Check Ollama status
    if not HAS_HTTPX:
        cache.update(
            {
                "checked_at": now,
                "running": False,
                "models": [],
                "default_model_available": False,
            }
        )
        return {
            "running": False,
            "error": "httpx not installed",
            "models": [],
            "default_model_available": False,
            "cached": False,
        }

    ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Check if Ollama is running
            response = await client.get(f"{ollama_url}/api/tags")
            response.raise_for_status()

            data = response.json()
            models = [m.get("name", "") for m in data.get("models", [])]

            # Check if default model is available
            default_model = DEFAULT_MODELS["ollama"]
            # Handle model name variations (gemma3:12b vs gemma3:12b-instruct-q4_0)
            default_available = any(m.startswith(default_model.split(":")[0]) for m in models)

            cache.update(
                {
                    "checked_at": now,
                    "running": True,
                    "models": models,
                    "default_model_available": default_available,
                }
            )

            return {
                "running": True,
                "url": ollama_url,
                "models": models,
                "model_count": len(models),
                "default_model": default_model,
                "default_model_available": default_available,
                "cached": False,
                "checked_at": now,
            }

    except httpx.ConnectError:
        cache.update(
            {
                "checked_at": now,
                "running": False,
                "models": [],
                "default_model_available": False,
            }
        )
        return {
            "running": False,
            "url": ollama_url,
            "error": "connection_refused",
            "message": "Ollama server not running. Start with: ollama serve",
            "models": [],
            "default_model_available": False,
            "cached": False,
        }
    except Exception as e:
        cache.update(
            {
                "checked_at": now,
                "running": False,
                "models": [],
                "default_model_available": False,
            }
        )
        return {
            "running": False,
            "url": ollama_url,
            "error": "check_failed",
            "message": str(e),
            "models": [],
            "default_model_available": False,
            "cached": False,
        }


def _get_best_provider() -> str:
    """Get the best available provider. Prefers Ollama if available."""
    cache = _ollama_status_cache
    if cache["running"] and cache["default_model_available"]:
        return "ollama"
    return "claude-sdk"


def _get_best_model_for_provider(provider: str) -> str:
    """Get the best available model for a provider."""
    if provider == "ollama":
        cache = _ollama_status_cache
        default = DEFAULT_MODELS["ollama"]
        # If default model available, use it
        if cache["default_model_available"]:
            return default
        # Otherwise pick first available model
        if cache["models"]:
            return cache["models"][0]
        return default
    return DEFAULT_MODELS.get(provider, "claude-haiku-4-5-20251101")


def _hash_content(content: str) -> str:
    """Create short hash for content identification."""
    return hashlib.sha256(content.encode()).hexdigest()[:12]


def _load_context_from_disk(name: str) -> Optional[dict]:
    """Load context from disk if it exists."""
    meta_path = CONTEXTS_DIR / f"{name}.meta.json"
    content_path = CONTEXTS_DIR / f"{name}.txt"

    if not (meta_path.exists() and content_path.exists()):
        return None

    meta = json.loads(meta_path.read_text())
    meta["content"] = content_path.read_text()
    return meta


def _save_context_to_disk(name: str, content: str, meta: dict) -> None:
    """Persist context to disk."""
    (CONTEXTS_DIR / f"{name}.txt").write_text(content)
    meta_without_content = {k: v for k, v in meta.items() if k != "content"}
    (CONTEXTS_DIR / f"{name}.meta.json").write_text(json.dumps(meta_without_content, indent=2))


def _ensure_context_loaded(name: str) -> Optional[str]:
    """Ensure context is loaded into memory. Returns error message if not found."""
    if name in contexts:
        return None

    disk_context = _load_context_from_disk(name)
    if disk_context:
        content = disk_context.pop("content")
        contexts[name] = {"meta": disk_context, "content": content}
        return None

    return f"Context '{name}' not found"


def _context_summary(name: str, content: str, **extra: Any) -> dict:
    """Build a common context summary dict."""
    summary = {
        "name": name,
        "length": len(content),
        "lines": content.count("\n") + 1,
    }
    summary.update(extra)
    return summary


async def _call_ollama(query: str, context_content: str, model: str) -> tuple[Optional[str], Optional[str]]:
    """Make a sub-call to Ollama. Returns (result, error)."""
    if not HAS_HTTPX:
        return None, "httpx required for Ollama calls"

    ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            response = await client.post(
                f"{ollama_url}/api/generate",
                json={
                    "model": model,
                    "prompt": f"{query}\n\nContext:\n{context_content}",
                    "stream": False,
                },
            )
            response.raise_for_status()
            return response.json().get("response", ""), None
    except Exception as e:
        return None, str(e)


async def _call_claude_sdk(query: str, context_content: str, model: str) -> tuple[Optional[str], Optional[str]]:
    """Make a sub-call to Claude SDK. Returns (result, error)."""
    if not HAS_CLAUDE_SDK:
        return None, "claude-agent-sdk required for claude-sdk provider"

    try:
        prompt = f"{query}\n\nContext:\n{context_content}"
        options = ClaudeAgentOptions(max_turns=1)

        texts = []
        async for message in claude_query(prompt=prompt, options=options):
            if hasattr(message, "content"):
                content = message.content
                # Extract text from TextBlock objects
                if isinstance(content, list):
                    for block in content:
                        if hasattr(block, "text"):
                            texts.append(block.text)
                elif hasattr(content, "text"):
                    texts.append(content.text)
                else:
                    texts.append(str(content))

        result = "\n".join(texts) if texts else ""
        return result, None
    except Exception as e:
        return None, str(e)


async def _resolve_provider_and_model(
    provider: str,
    model: Optional[str],
) -> tuple[str, str]:
    """Resolve 'auto' provider and get appropriate model."""
    # Handle auto provider selection
    if provider == "auto":
        # Check Ollama status (uses cache)
        await _check_ollama_status()
        provider = _get_best_provider()

    # Get model if not specified
    if not model:
        model = _get_best_model_for_provider(provider)

    return provider, model


async def _make_provider_call(
    provider: str,
    model: str,
    query: str,
    context_content: str,
) -> tuple[Optional[str], Optional[str]]:
    """Route a sub-call to the appropriate provider. Returns (result, error)."""
    # Resolve auto provider
    resolved_provider, resolved_model = await _resolve_provider_and_model(provider, model)

    if resolved_provider == "ollama":
        return await _call_ollama(query, context_content, resolved_model)
    elif resolved_provider == "claude-sdk":
        return await _call_claude_sdk(query, context_content, resolved_model)
    else:
        return None, f"Unknown provider: {resolved_provider}"


def _chunk_content(content: str, strategy: str, size: int) -> list[str]:
    """Chunk content using the specified strategy."""
    if strategy == "lines":
        lines = content.split("\n")
        return ["\n".join(lines[i : i + size]) for i in range(0, len(lines), size)]
    elif strategy == "chars":
        return [content[i : i + size] for i in range(0, len(content), size)]
    elif strategy == "paragraphs":
        paragraphs = re.split(r"\n\s*\n", content)
        return ["\n\n".join(paragraphs[i : i + size]) for i in range(0, len(paragraphs), size)]
    return []


def _detect_content_type(content: str) -> dict:
    """Detect content type from first 1000 chars. Returns type and confidence."""
    sample = content[:1000]

    # Python detection
    python_patterns = ["import ", "def ", "class ", "if __name__"]
    python_score = sum(1 for p in python_patterns if p in sample)

    # JSON detection
    json_score = 0
    stripped = sample.strip()
    if stripped.startswith(("{", "[")):
        try:
            json.loads(content[:10000])  # Try parsing first 10K
            json_score = 10
        except json.JSONDecodeError:
            json_score = 3 if stripped.startswith(("{", "[")) else 0

    # Markdown detection
    md_patterns = ["# ", "## ", "**", "```"]
    md_score = sum(1 for p in md_patterns if p in sample)

    # Log detection
    log_patterns = ["ERROR", "INFO", "DEBUG", "WARN"]
    log_score = sum(1 for p in log_patterns if p in sample)
    if re.search(r"\d{4}-\d{2}-\d{2}", sample):  # Date pattern
        log_score += 2

    # Generic code detection
    code_indicators = ["{", "}", ";", "=>", "->"]
    code_score = sum(sample.count(c) for c in code_indicators) / 10

    # Prose detection
    sentence_count = len(re.findall(r"[.!?]\s+[A-Z]", sample))
    prose_score = sentence_count

    scores = {
        "python": python_score,
        "json": json_score,
        "markdown": md_score,
        "logs": log_score,
        "code": code_score,
        "prose": prose_score,
    }

    detected_type = max(scores, key=scores.get)
    max_score = scores[detected_type]
    confidence = min(1.0, max_score / 10.0) if max_score > 0 else 0.5

    return {"type": detected_type, "confidence": round(confidence, 2)}


def _select_chunking_strategy(content_type: str) -> dict:
    """Select chunking strategy based on content type."""
    strategies = {
        "python": {"strategy": "lines", "size": 150},
        "code": {"strategy": "lines", "size": 150},
        "json": {"strategy": "chars", "size": 10000},
        "markdown": {"strategy": "paragraphs", "size": 20},
        "logs": {"strategy": "lines", "size": 500},
        "prose": {"strategy": "paragraphs", "size": 30},
    }
    return strategies.get(content_type, {"strategy": "lines", "size": 100})


def _adapt_query_for_goal(goal: str, content_type: str) -> str:
    """Generate appropriate sub-query based on goal and content type."""
    if goal.startswith("answer:"):
        return goal[7:].strip()

    goal_templates = {
        "find_bugs": {
            "python": "Identify bugs, issues, or potential errors in this Python code. Look for: syntax errors, logic errors, unhandled exceptions, type mismatches, missing imports.",
            "code": "Identify bugs, issues, or potential errors in this code. Look for: syntax errors, logic errors, unhandled exceptions.",
            "default": "Identify any errors, issues, or problems in this content.",
        },
        "summarize": {
            "python": "Summarize what this Python code does. List main functions/classes and their purpose.",
            "code": "Summarize what this code does. List main functions and their purpose.",
            "markdown": "Summarize the main points of this documentation in 2-3 sentences.",
            "prose": "Summarize the main points of this text in 2-3 sentences.",
            "logs": "Summarize the key events and errors in these logs.",
            "json": "Summarize the structure and key data in this JSON.",
            "default": "Summarize the main points of this content in 2-3 sentences.",
        },
        "extract_structure": {
            "python": "Extract the code structure: list all classes, functions, and their signatures.",
            "code": "Extract the code structure: list all functions/classes and their signatures.",
            "json": "Extract the JSON schema: list top-level keys and their types.",
            "markdown": "Extract the document structure: list all headings and hierarchy.",
            "default": "Extract the main structural elements of this content.",
        },
        "security_audit": {
            "python": "Find security vulnerabilities: SQL injection, command injection, eval(), exec(), unsafe deserialization, hardcoded secrets, path traversal.",
            "code": "Find security vulnerabilities: injection flaws, unsafe functions, hardcoded credentials.",
            "default": "Identify potential security issues or sensitive information.",
        },
    }

    templates = goal_templates.get(goal, {})
    return templates.get(content_type, templates.get("default", f"Analyze this content for: {goal}"))


# =============================================================================
# FastMCP Tool Definitions
# =============================================================================


@mcp.tool()
async def rlm_system_check() -> dict:
    """Check if system meets requirements for Ollama with gemma3:12b.

    Verifies: macOS, Apple Silicon (M1/M2/M3/M4), 16GB+ RAM, Homebrew installed.
    Use before attempting Ollama setup.
    """
    result = _check_system_requirements()

    # Add summary
    if result["meets_requirements"]:
        result["summary"] = (
            f"System ready for Ollama! {result.get('chip', 'Apple Silicon')} with "
            f"{result['ram_gb']}GB RAM. Use rlm_setup_ollama to install."
        )
    else:
        result["summary"] = (
            f"System check: {len(result['issues'])} issue(s) found. See 'issues' and 'recommendations' for details."
        )

    return result


@mcp.tool()
async def rlm_setup_ollama(
    install: bool = False,
    start_service: bool = False,
    pull_model: bool = False,
    model: str = "gemma3:12b",
) -> dict:
    """Install Ollama via Homebrew (macOS).

    Requires Homebrew pre-installed. Uses 'brew install' and 'brew services'.
    PROS: Auto-updates, pre-built binaries, managed service.
    CONS: Requires Homebrew, may prompt for sudo on first Homebrew install.

    Args:
        install: Install Ollama via Homebrew (requires Homebrew)
        start_service: Start Ollama as a background service via brew services
        pull_model: Pull the default model (gemma3:12b)
        model: Model to pull (default: gemma3:12b). Use gemma3:4b or gemma3:1b for lower RAM systems.
    """
    # If no actions specified, just do a system check
    if not any([install, start_service, pull_model]):
        sys_check = _check_system_requirements()
        return {
            "message": "No actions specified. Use install=true, start_service=true, or pull_model=true.",
            "system_check": sys_check,
            "example": "rlm_setup_ollama(install=true, start_service=true, pull_model=true)",
        }

    result = await _setup_ollama(
        install=install,
        start_service=start_service,
        pull_model=pull_model,
        model=model,
    )

    # Add summary
    if result["success"]:
        result["summary"] = (
            f"Setup complete! Actions: {', '.join(result['actions_taken']) or 'none'}. "
            f"Skipped: {', '.join(result['actions_skipped']) or 'none'}."
        )
    else:
        result["summary"] = f"Setup failed: {'; '.join(result['errors'])}"

    return result


@mcp.tool()
async def rlm_setup_ollama_direct(
    install: bool = False,
    start_service: bool = False,
    pull_model: bool = False,
    model: str = "gemma3:12b",
) -> dict:
    """Install Ollama via direct download (macOS).

    Downloads from ollama.com to ~/Applications.
    PROS: No Homebrew needed, no sudo required, fully headless, works on locked-down machines.
    CONS: Manual PATH setup, no auto-updates, service runs as foreground process.

    Args:
        install: Download and install Ollama to ~/Applications (no sudo needed)
        start_service: Start Ollama server (ollama serve) in background
        pull_model: Pull the default model (gemma3:12b)
        model: Model to pull (default: gemma3:12b). Use gemma3:4b or gemma3:1b for lower RAM systems.
    """
    # If no actions specified, show comparison
    if not any([install, start_service, pull_model]):
        return {
            "message": "No actions specified. Use install=true, start_service=true, or pull_model=true.",
            "method": "direct_download",
            "advantages": [
                "No Homebrew required",
                "No sudo/admin permissions needed",
                "Fully headless automation",
                "Works on locked-down/managed machines",
            ],
            "disadvantages": [
                "Manual PATH setup needed (CLI at ~/Applications/Ollama.app/Contents/Resources/ollama)",
                "No automatic updates",
                "Service runs via 'ollama serve' (not a managed launchd service)",
            ],
            "example": "rlm_setup_ollama_direct(install=true, start_service=true, pull_model=true)",
            "alternative": "Use rlm_setup_ollama for Homebrew-based installation if you have Homebrew",
        }

    result = await _setup_ollama_direct(
        install=install,
        start_service=start_service,
        pull_model=pull_model,
        model=model,
    )

    # Add summary
    if result["success"]:
        result["summary"] = (
            f"Setup complete (direct download)! Actions: {', '.join(result['actions_taken']) or 'none'}. "
            f"Skipped: {', '.join(result['actions_skipped']) or 'none'}."
        )
        if result.get("path_setup"):
            result["summary"] += f" NOTE: Add to PATH: {result['path_setup']['add_to_path']}"
    else:
        result["summary"] = f"Setup failed: {'; '.join(result['errors'])}"

    return result


@mcp.tool()
async def rlm_ollama_status(force_refresh: bool = False) -> dict:
    """Check Ollama server status and available models.

    Returns whether Ollama is running, list of available models, and if the
    default model (gemma3:12b) is available. Use this to determine if free
    local inference is available.

    Args:
        force_refresh: Force refresh the cached status (default: false)
    """
    status = await _check_ollama_status(force_refresh=force_refresh)

    # Add recommendation based on status
    if status["running"] and status["default_model_available"]:
        status["recommendation"] = "Ollama is ready! Sub-queries will use free local inference by default."
    elif status["running"] and not status["default_model_available"]:
        default_model = DEFAULT_MODELS["ollama"]
        status["recommendation"] = f"Ollama is running but default model not found. Run: ollama pull {default_model}"
    else:
        status["recommendation"] = (
            "Ollama not available. Sub-queries will use Claude API. To enable free local inference, install Ollama and run: ollama serve"
        )

    # Add current best provider
    status["best_provider"] = _get_best_provider()

    return status


async def _load_context_impl(name: str, content: str) -> dict:
    """Implementation of context loading."""
    content_hash = _hash_content(content)
    meta = _context_summary(name, content, hash=content_hash, chunks=None)
    contexts[name] = {"meta": meta, "content": content}
    _save_context_to_disk(name, content, meta)

    return {
        "status": "loaded",
        "name": name,
        "length": meta["length"],
        "lines": meta["lines"],
        "hash": content_hash,
    }


@mcp.tool()
async def rlm_load_context(name: str, content: str) -> dict:
    """Load a large context as an external variable.

    Returns metadata without the content itself.

    Args:
        name: Identifier for this context
        content: The full context content
    """
    return await _load_context_impl(name, content)


@mcp.tool()
async def rlm_inspect_context(name: str, preview_chars: int = 500) -> dict:
    """Inspect a loaded context - get structure info without loading full content into prompt.

    Args:
        name: Context identifier
        preview_chars: Number of chars to preview (default 500)
    """
    error = _ensure_context_loaded(name)
    if error:
        return {"error": "context_not_found", "message": error}

    ctx = contexts[name]
    content = ctx["content"]
    chunk_meta = ctx["meta"].get("chunks")

    return _context_summary(
        name,
        content,
        preview=content[:preview_chars],
        has_chunks=chunk_meta is not None,
        chunk_count=len(chunk_meta) if chunk_meta else 0,
    )


async def _chunk_context_impl(
    name: str,
    strategy: str = "lines",
    size: int = 100,
) -> dict:
    """Implementation of context chunking."""
    error = _ensure_context_loaded(name)
    if error:
        return {"error": "context_not_found", "message": error}

    content = contexts[name]["content"]
    chunks = _chunk_content(content, strategy, size)

    chunk_meta = [{"index": i, "length": len(chunk), "preview": chunk[:100]} for i, chunk in enumerate(chunks)]

    contexts[name]["meta"]["chunks"] = chunk_meta
    contexts[name]["chunks"] = chunks

    chunk_dir = CHUNKS_DIR / name
    chunk_dir.mkdir(exist_ok=True)
    for i, chunk in enumerate(chunks):
        (chunk_dir / f"{i}.txt").write_text(chunk)

    return {
        "status": "chunked",
        "name": name,
        "strategy": strategy,
        "chunk_count": len(chunks),
        "chunks": chunk_meta,
    }


@mcp.tool()
async def rlm_chunk_context(
    name: str,
    strategy: str = "lines",
    size: int = 100,
) -> dict:
    """Chunk a loaded context by strategy. Returns chunk metadata, not full content.

    Args:
        name: Context identifier
        strategy: Chunking strategy - 'lines', 'chars', or 'paragraphs'
        size: Chunk size (lines/chars depending on strategy)
    """
    return await _chunk_context_impl(name, strategy, size)


@mcp.tool()
async def rlm_get_chunk(name: str, chunk_index: int) -> str | dict:
    """Get a specific chunk by index. Use after chunking to retrieve individual pieces.

    Args:
        name: Context identifier
        chunk_index: Index of chunk to retrieve
    """
    error = _ensure_context_loaded(name)
    if error:
        return {"error": "context_not_found", "message": error}

    chunks = contexts[name].get("chunks")
    if not chunks:
        chunk_path = CHUNKS_DIR / name / f"{chunk_index}.txt"
        if chunk_path.exists():
            return chunk_path.read_text()
        return {"error": "context_not_chunked", "message": f"Context '{name}' has not been chunked yet"}

    if chunk_index >= len(chunks):
        return {
            "error": "chunk_out_of_range",
            "message": f"Chunk index {chunk_index} out of range (max {len(chunks) - 1})",
        }

    return chunks[chunk_index]


@mcp.tool()
async def rlm_filter_context(
    name: str,
    output_name: str,
    pattern: str,
    mode: str = "keep",
) -> dict:
    """Filter context using regex/string operations. Creates a new filtered context.

    Args:
        name: Source context identifier
        output_name: Name for filtered context
        pattern: Regex pattern to match
        mode: 'keep' or 'remove' matching lines
    """
    error = _ensure_context_loaded(name)
    if error:
        return {"error": "context_not_found", "message": error}

    content = contexts[name]["content"]
    lines = content.split("\n")
    regex = re.compile(pattern)

    if mode == "keep":
        filtered = [line for line in lines if regex.search(line)]
    else:
        filtered = [line for line in lines if not regex.search(line)]

    new_content = "\n".join(filtered)
    meta = _context_summary(
        output_name,
        new_content,
        hash=_hash_content(new_content),
        source=name,
        filter_pattern=pattern,
        filter_mode=mode,
        chunks=None,
    )
    contexts[output_name] = {"meta": meta, "content": new_content}
    _save_context_to_disk(output_name, new_content, meta)

    return {
        "status": "filtered",
        "name": output_name,
        "original_lines": len(lines),
        "filtered_lines": len(filtered),
        "length": len(new_content),
    }


@mcp.tool()
async def rlm_sub_query(
    query: str,
    context_name: str,
    chunk_index: Optional[int] = None,
    provider: str = "auto",
    model: Optional[str] = None,
) -> dict:
    """Make a sub-LLM call on a chunk or filtered context. Core of recursive pattern.

    Args:
        query: Question/instruction for the sub-call
        context_name: Context identifier to query against
        chunk_index: Optional: specific chunk index
        provider: LLM provider - 'auto', 'ollama', or 'claude-sdk'. 'auto' prefers Ollama if available (free local inference)
        model: Model to use (provider-specific defaults apply)
    """
    # Resolve auto provider and model
    resolved_provider, resolved_model = await _resolve_provider_and_model(provider, model)

    error = _ensure_context_loaded(context_name)
    if error:
        return {"error": "context_not_found", "message": error}

    if chunk_index is not None:
        chunks = contexts[context_name].get("chunks")
        if not chunks or chunk_index >= len(chunks):
            return {"error": "chunk_not_available", "message": f"Chunk {chunk_index} not available"}
        context_content = chunks[chunk_index]
    else:
        context_content = contexts[context_name]["content"]

    result, call_error = await _make_provider_call(resolved_provider, resolved_model, query, context_content)

    if call_error:
        return {
            "error": "provider_error",
            "provider": resolved_provider,
            "model": resolved_model,
            "requested_provider": provider,
            "message": call_error,
        }

    return {
        "provider": resolved_provider,
        "model": resolved_model,
        "requested_provider": provider if provider == "auto" else None,
        "response": result,
    }


@mcp.tool()
async def rlm_store_result(
    name: str,
    result: str,
    metadata: Optional[dict] = None,
) -> str:
    """Store a sub-call result for later aggregation.

    Args:
        name: Result set identifier
        result: Result content to store
        metadata: Optional metadata about this result
    """
    results_file = RESULTS_DIR / f"{name}.jsonl"
    with open(results_file, "a") as f:
        f.write(json.dumps({"result": result, "metadata": metadata or {}}) + "\n")

    return f"Result stored to '{name}'"


@mcp.tool()
async def rlm_get_results(name: str) -> dict | str:
    """Retrieve stored results for aggregation.

    Args:
        name: Result set identifier
    """
    results_file = RESULTS_DIR / f"{name}.jsonl"

    if not results_file.exists():
        return f"No results found for '{name}'"

    results = [json.loads(line) for line in results_file.read_text().splitlines()]

    return {
        "name": name,
        "count": len(results),
        "results": results,
    }


@mcp.tool()
async def rlm_list_contexts() -> dict:
    """List all loaded contexts and their metadata."""
    ctx_list = [
        {
            "name": name,
            "length": ctx["meta"]["length"],
            "lines": ctx["meta"]["lines"],
            "chunked": ctx["meta"].get("chunks") is not None,
        }
        for name, ctx in contexts.items()
    ]

    for meta_file in CONTEXTS_DIR.glob("*.meta.json"):
        disk_name = meta_file.stem.replace(".meta", "")
        if disk_name not in contexts:
            meta = json.loads(meta_file.read_text())
            ctx_list.append(
                {
                    "name": disk_name,
                    "length": meta["length"],
                    "lines": meta["lines"],
                    "chunked": meta.get("chunks") is not None,
                    "disk_only": True,
                }
            )

    return {"contexts": ctx_list}


async def _sub_query_batch_impl(
    query: str,
    context_name: str,
    chunk_indices: list[int],
    provider: str = "auto",
    model: Optional[str] = None,
    concurrency: int = 4,
) -> dict:
    """Implementation of batch sub-query processing."""
    concurrency = min(concurrency, 8)

    # Resolve auto provider and model once for the entire batch
    resolved_provider, resolved_model = await _resolve_provider_and_model(provider, model)

    error = _ensure_context_loaded(context_name)
    if error:
        return {"error": "context_not_found", "message": error}

    chunks = contexts[context_name].get("chunks")
    if not chunks:
        return {"error": "context_not_chunked", "message": f"Context '{context_name}' has not been chunked yet"}

    invalid_indices = [idx for idx in chunk_indices if idx >= len(chunks)]
    if invalid_indices:
        return {
            "error": "invalid_chunk_indices",
            "message": f"Invalid chunk indices: {invalid_indices} (max: {len(chunks) - 1})",
        }

    semaphore = asyncio.Semaphore(concurrency)

    async def process_chunk(chunk_idx: int) -> dict:
        async with semaphore:
            chunk_content = chunks[chunk_idx]
            result, call_error = await _make_provider_call(resolved_provider, resolved_model, query, chunk_content)

            if call_error:
                return {
                    "chunk_index": chunk_idx,
                    "error": "provider_error",
                    "message": call_error,
                }

            return {
                "chunk_index": chunk_idx,
                "response": result,
                "provider": resolved_provider,
                "model": resolved_model,
            }

    results = await asyncio.gather(*[process_chunk(idx) for idx in chunk_indices])

    successful = sum(1 for r in results if "response" in r)
    failed = len(results) - successful

    return {
        "status": "completed",
        "total_chunks": len(chunk_indices),
        "successful": successful,
        "failed": failed,
        "concurrency": concurrency,
        "provider": resolved_provider,
        "model": resolved_model,
        "requested_provider": provider if provider == "auto" else None,
        "results": results,
    }


@mcp.tool()
async def rlm_sub_query_batch(
    query: str,
    context_name: str,
    chunk_indices: list[int],
    provider: str = "auto",
    model: Optional[str] = None,
    concurrency: int = 4,
) -> dict:
    """Process multiple chunks in parallel. Respects concurrency limit to manage system resources.

    Args:
        query: Question/instruction for each sub-call
        context_name: Context identifier
        chunk_indices: List of chunk indices to process
        provider: LLM provider - 'auto', 'ollama', or 'claude-sdk'
        model: Model to use (provider-specific defaults apply)
        concurrency: Max parallel requests (default 4, max 8)
    """
    return await _sub_query_batch_impl(query, context_name, chunk_indices, provider, model, concurrency)


@mcp.tool()
async def rlm_auto_analyze(
    name: str,
    content: str,
    goal: str,
    provider: str = "auto",
    concurrency: int = 4,
) -> dict:
    """Automatically detect content type and analyze with optimal chunking strategy.

    One-step analysis for common tasks.

    Args:
        name: Context identifier
        content: The content to analyze
        goal: Analysis goal: 'summarize', 'find_bugs', 'extract_structure', 'security_audit', or 'answer:<your question>'
        provider: LLM provider - 'auto' prefers Ollama if available
        concurrency: Max parallel requests (default 4, max 8)
    """
    concurrency = min(concurrency, 8)

    # Debug: verify _load_context_impl is callable
    import sys
    print(f"DEBUG: _load_context_impl type = {type(_load_context_impl)}", file=sys.stderr)
    print(f"DEBUG: _chunk_context_impl type = {type(_chunk_context_impl)}", file=sys.stderr)
    print(f"DEBUG: _sub_query_batch_impl type = {type(_sub_query_batch_impl)}", file=sys.stderr)

    # Load the content (call implementation directly, not the tool)
    await _load_context_impl(name=name, content=content)

    # Detect content type
    detection = _detect_content_type(content)
    detected_type = detection["type"]
    confidence = detection["confidence"]

    # Select chunking strategy
    strategy_config = _select_chunking_strategy(detected_type)

    # Chunk the content (call implementation directly, not the tool)
    chunk_result = await _chunk_context_impl(
        name=name,
        strategy=strategy_config["strategy"],
        size=strategy_config["size"],
    )
    chunk_count = chunk_result["chunk_count"]

    # Sample if too many chunks (max 20)
    chunk_indices = list(range(chunk_count))
    sampled = False
    if chunk_count > 20:
        step = max(1, chunk_count // 20)
        chunk_indices = list(range(0, chunk_count, step))[:20]
        sampled = True

    # Adapt query for goal and content type
    adapted_query = _adapt_query_for_goal(goal, detected_type)

    # Run batch query (call implementation directly, not the tool)
    batch_result = await _sub_query_batch_impl(
        query=adapted_query,
        context_name=name,
        chunk_indices=chunk_indices,
        provider=provider,
        concurrency=concurrency,
    )

    return {
        "status": "completed",
        "detected_type": detected_type,
        "confidence": confidence,
        "strategy": strategy_config,
        "chunk_count": chunk_count,
        "chunks_analyzed": len(chunk_indices),
        "sampled": sampled,
        "goal": goal,
        "adapted_query": adapted_query,
        "provider": provider,
        "successful": batch_result["successful"],
        "failed": batch_result["failed"],
        "results": batch_result["results"],
    }


@mcp.tool()
async def rlm_exec(
    code: str,
    context_name: str,
    timeout: int = 30,
) -> dict:
    """Execute Python code against a loaded context in a sandboxed subprocess.

    Set result variable for output.

    Args:
        code: Python code to execute. User sets result variable for output.
        context_name: Name of previously loaded context
        timeout: Max execution time in seconds (default 30)
    """
    # Ensure context is loaded
    error = _ensure_context_loaded(context_name)
    if error:
        return {"error": "context_not_found", "message": error}

    content = contexts[context_name]["content"]

    # Create a temporary Python file with the execution environment
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        temp_file = f.name
        # Write the execution wrapper
        f.write("""
import sys
import json
import re
import collections

# Inject context as read-only variable
context = sys.stdin.read()

# User code execution
result = None
try:
""")
        # Indent user code
        for line in code.split("\n"):
            f.write(f"    {line}\n")

        # Capture result
        f.write("""
    # Output result
    if result is not None:
        print("__RESULT_START__")
        print(json.dumps(result, indent=2) if isinstance(result, (dict, list)) else str(result))
        print("__RESULT_END__")
except Exception as e:
    print(f"__ERROR__: {type(e).__name__}: {e}", file=sys.stderr)
    sys.exit(1)
""")

    try:
        # Run the subprocess with minimal environment (no shell=True for security)
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        }

        process = subprocess.run(
            [sys.executable, temp_file],
            input=content,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )

        # Parse output
        stdout = process.stdout
        stderr = process.stderr
        return_code = process.returncode

        # Extract result
        result = None
        if "__RESULT_START__" in stdout and "__RESULT_END__" in stdout:
            result_start = stdout.index("__RESULT_START__") + len("__RESULT_START__\n")
            result_end = stdout.index("__RESULT_END__")
            result_str = stdout[result_start:result_end].strip()
            try:
                result = json.loads(result_str)
            except json.JSONDecodeError:
                result = result_str

            # Clean stdout
            stdout = stdout[: stdout.index("__RESULT_START__")].strip()

        return {
            "result": result,
            "stdout": stdout,
            "stderr": stderr,
            "return_code": return_code,
            "timed_out": False,
        }

    except subprocess.TimeoutExpired:
        return {
            "result": None,
            "stdout": "",
            "stderr": f"Execution timed out after {timeout} seconds",
            "return_code": -1,
            "timed_out": True,
        }
    except Exception as e:
        return {"error": "execution_error", "message": str(e)}
    finally:
        # Clean up temp file
        try:
            os.unlink(temp_file)
        except Exception:
            pass


# =============================================================================
# Entry Point
# =============================================================================


def main():
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
