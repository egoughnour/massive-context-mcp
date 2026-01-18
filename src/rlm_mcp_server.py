#!/usr/bin/env python3
"""
RLM MCP Server - Recursive Language Model patterns for massive context handling.

Implements the core insight from https://arxiv.org/html/2512.24601v1:
Treat context as external variable, chunk programmatically, sub-call recursively.
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

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

try:
    from claude_agent_sdk import ClaudeAgentOptions, query as claude_query
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

server = Server("rlm")

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
        result["recommendations"].append(
            "Apple Silicon (M1/M2/M3/M4) recommended for optimal Ollama performance"
        )

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
                "Install Homebrew first: /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
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
        result["is_macos"]
        and result["is_apple_silicon"]
        and result["ram_sufficient"]
        and result["homebrew_installed"]
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
                f"Ollama CLI not found. Expected at {cli_path} or in PATH. "
                "You may need to add it to your PATH first."
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
                        result["actions_skipped"].append(
                            "brew services failed, try: ollama serve &"
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
                    result["errors"].append(f"Model pull timed out (30 min limit)")
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
        cache.update({
            "checked_at": now,
            "running": False,
            "models": [],
            "default_model_available": False,
        })
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
            default_available = any(
                m.startswith(default_model.split(":")[0]) for m in models
            )

            cache.update({
                "checked_at": now,
                "running": True,
                "models": models,
                "default_model_available": default_available,
            })

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
        cache.update({
            "checked_at": now,
            "running": False,
            "models": [],
            "default_model_available": False,
        })
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
        cache.update({
            "checked_at": now,
            "running": False,
            "models": [],
            "default_model_available": False,
        })
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
    (CONTEXTS_DIR / f"{name}.meta.json").write_text(
        json.dumps(meta_without_content, indent=2)
    )


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


def _text_response(data: Any) -> list[TextContent]:
    """Create a JSON text response."""
    if isinstance(data, str):
        return [TextContent(type="text", text=data)]
    return [TextContent(type="text", text=json.dumps(data, indent=2))]


def _error_response(code: str, message: str) -> list[TextContent]:
    """Create a structured error response."""
    return _text_response({"error": code, "message": message})


def _context_summary(name: str, content: str, **extra: Any) -> dict:
    """Build a common context summary dict."""
    summary = {
        "name": name,
        "length": len(content),
        "lines": content.count("\n") + 1,
    }
    summary.update(extra)
    return summary


# Shared schema fragments for tool definitions
PROVIDER_SCHEMA = {
    "type": "string",
    "enum": ["auto", "ollama", "claude-sdk"],
    "description": "LLM provider for sub-call. 'auto' prefers Ollama if available (free local inference)",
    "default": "auto",
}

PROVIDER_SCHEMA_CLAUDE_DEFAULT = {
    **PROVIDER_SCHEMA,
    "description": "LLM provider for sub-calls. 'auto' prefers Ollama if available",
    "default": "auto",
}


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
        return [
            "\n\n".join(paragraphs[i : i + size])
            for i in range(0, len(paragraphs), size)
        ]
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


# Tool definitions
TOOL_DEFINITIONS = [
    Tool(
        name="rlm_system_check",
        description="Check if system meets requirements for Ollama with gemma3:12b. Verifies: macOS, Apple Silicon (M1/M2/M3/M4), 16GB+ RAM, Homebrew installed. Use before attempting Ollama setup.",
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="rlm_setup_ollama",
        description="Install Ollama via Homebrew (macOS). Requires Homebrew pre-installed. Uses 'brew install' and 'brew services'. PROS: Auto-updates, pre-built binaries, managed service. CONS: Requires Homebrew, may prompt for sudo on first Homebrew install.",
        inputSchema={
            "type": "object",
            "properties": {
                "install": {
                    "type": "boolean",
                    "description": "Install Ollama via Homebrew (requires Homebrew)",
                    "default": False,
                },
                "start_service": {
                    "type": "boolean",
                    "description": "Start Ollama as a background service via brew services",
                    "default": False,
                },
                "pull_model": {
                    "type": "boolean",
                    "description": "Pull the default model (gemma3:12b)",
                    "default": False,
                },
                "model": {
                    "type": "string",
                    "description": "Model to pull (default: gemma3:12b). Use gemma3:4b or gemma3:1b for lower RAM systems.",
                    "default": "gemma3:12b",
                },
            },
        },
    ),
    Tool(
        name="rlm_setup_ollama_direct",
        description="Install Ollama via direct download (macOS). Downloads from ollama.com to ~/Applications. PROS: No Homebrew needed, no sudo required, fully headless, works on locked-down machines. CONS: Manual PATH setup, no auto-updates, service runs as foreground process.",
        inputSchema={
            "type": "object",
            "properties": {
                "install": {
                    "type": "boolean",
                    "description": "Download and install Ollama to ~/Applications (no sudo needed)",
                    "default": False,
                },
                "start_service": {
                    "type": "boolean",
                    "description": "Start Ollama server (ollama serve) in background",
                    "default": False,
                },
                "pull_model": {
                    "type": "boolean",
                    "description": "Pull the default model (gemma3:12b)",
                    "default": False,
                },
                "model": {
                    "type": "string",
                    "description": "Model to pull (default: gemma3:12b). Use gemma3:4b or gemma3:1b for lower RAM systems.",
                    "default": "gemma3:12b",
                },
            },
        },
    ),
    Tool(
        name="rlm_ollama_status",
        description="Check Ollama server status and available models. Returns whether Ollama is running, list of available models, and if the default model (gemma3:12b) is available. Use this to determine if free local inference is available.",
        inputSchema={
            "type": "object",
            "properties": {
                "force_refresh": {
                    "type": "boolean",
                    "description": "Force refresh the cached status (default: false)",
                    "default": False,
                },
            },
        },
    ),
    Tool(
        name="rlm_load_context",
        description="Load a large context as an external variable. Returns metadata without the content itself.",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Identifier for this context"},
                "content": {"type": "string", "description": "The full context content"},
            },
            "required": ["name", "content"],
        },
    ),
    Tool(
        name="rlm_inspect_context",
        description="Inspect a loaded context - get structure info without loading full content into prompt.",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Context identifier"},
                "preview_chars": {
                    "type": "integer",
                    "description": "Number of chars to preview (default 500)",
                    "default": 500,
                },
            },
            "required": ["name"],
        },
    ),
    Tool(
        name="rlm_chunk_context",
        description="Chunk a loaded context by strategy. Returns chunk metadata, not full content.",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Context identifier"},
                "strategy": {
                    "type": "string",
                    "enum": ["lines", "chars", "paragraphs"],
                    "description": "Chunking strategy",
                    "default": "lines",
                },
                "size": {
                    "type": "integer",
                    "description": "Chunk size (lines/chars depending on strategy)",
                    "default": 100,
                },
            },
            "required": ["name"],
        },
    ),
    Tool(
        name="rlm_get_chunk",
        description="Get a specific chunk by index. Use after chunking to retrieve individual pieces.",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Context identifier"},
                "chunk_index": {"type": "integer", "description": "Index of chunk to retrieve"},
            },
            "required": ["name", "chunk_index"],
        },
    ),
    Tool(
        name="rlm_filter_context",
        description="Filter context using regex/string operations. Creates a new filtered context.",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Source context identifier"},
                "output_name": {"type": "string", "description": "Name for filtered context"},
                "pattern": {"type": "string", "description": "Regex pattern to match"},
                "mode": {
                    "type": "string",
                    "enum": ["keep", "remove"],
                    "description": "Keep or remove matching lines",
                    "default": "keep",
                },
            },
            "required": ["name", "output_name", "pattern"],
        },
    ),
    Tool(
        name="rlm_sub_query",
        description="Make a sub-LLM call on a chunk or filtered context. Core of recursive pattern.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Question/instruction for the sub-call"},
                "context_name": {"type": "string", "description": "Context identifier to query against"},
                "chunk_index": {"type": "integer", "description": "Optional: specific chunk index"},
                "provider": PROVIDER_SCHEMA,
                "model": {
                    "type": "string",
                    "description": "Model to use (provider-specific defaults apply)",
                },
            },
            "required": ["query", "context_name"],
        },
    ),
    Tool(
        name="rlm_store_result",
        description="Store a sub-call result for later aggregation.",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Result set identifier"},
                "result": {"type": "string", "description": "Result content to store"},
                "metadata": {"type": "object", "description": "Optional metadata about this result"},
            },
            "required": ["name", "result"],
        },
    ),
    Tool(
        name="rlm_get_results",
        description="Retrieve stored results for aggregation.",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Result set identifier"},
            },
            "required": ["name"],
        },
    ),
    Tool(
        name="rlm_list_contexts",
        description="List all loaded contexts and their metadata.",
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="rlm_sub_query_batch",
        description="Process multiple chunks in parallel. Respects concurrency limit to manage system resources.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Question/instruction for each sub-call"},
                "context_name": {"type": "string", "description": "Context identifier"},
                "chunk_indices": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "List of chunk indices to process",
                },
                "provider": PROVIDER_SCHEMA,
                "model": {
                    "type": "string",
                    "description": "Model to use (provider-specific defaults apply)",
                },
                "concurrency": {
                    "type": "integer",
                    "description": "Max parallel requests (default 4, max 8)",
                    "default": 4,
                },
            },
            "required": ["query", "context_name", "chunk_indices"],
        },
    ),
    Tool(
        name="rlm_auto_analyze",
        description="Automatically detect content type and analyze with optimal chunking strategy. One-step analysis for common tasks.",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Context identifier"},
                "content": {"type": "string", "description": "The content to analyze"},
                "goal": {
                    "type": "string",
                    "description": "Analysis goal: 'summarize', 'find_bugs', 'extract_structure', 'security_audit', or 'answer:<your question>'",
                },
                "provider": PROVIDER_SCHEMA_CLAUDE_DEFAULT,
                "concurrency": {
                    "type": "integer",
                    "description": "Max parallel requests (default 4, max 8)",
                    "default": 4,
                },
            },
            "required": ["name", "content", "goal"],
        },
    ),
    Tool(
        name="rlm_exec",
        description="Execute Python code against a loaded context in a sandboxed subprocess. Set result variable for output.",
        inputSchema={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute. User sets result variable for output.",
                },
                "context_name": {
                    "type": "string",
                    "description": "Name of previously loaded context",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Max execution time in seconds (default 30)",
                    "default": 30,
                },
            },
            "required": ["code", "context_name"],
        },
    ),
]


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available RLM tools."""
    return TOOL_DEFINITIONS


# Tool handlers
async def _handle_system_check(_arguments: dict) -> list[TextContent]:
    """Check if system meets requirements for Ollama."""
    result = _check_system_requirements()

    # Add summary
    if result["meets_requirements"]:
        result["summary"] = (
            f"System ready for Ollama! {result.get('chip', 'Apple Silicon')} with "
            f"{result['ram_gb']}GB RAM. Use rlm_setup_ollama to install."
        )
    else:
        result["summary"] = (
            f"System check: {len(result['issues'])} issue(s) found. "
            "See 'issues' and 'recommendations' for details."
        )

    return _text_response(result)


async def _handle_setup_ollama(arguments: dict) -> list[TextContent]:
    """Install and configure Ollama."""
    install = arguments.get("install", False)
    start_service = arguments.get("start_service", False)
    pull_model = arguments.get("pull_model", False)
    model = arguments.get("model", "gemma3:12b")

    # If no actions specified, just do a system check
    if not any([install, start_service, pull_model]):
        sys_check = _check_system_requirements()
        return _text_response({
            "message": "No actions specified. Use install=true, start_service=true, or pull_model=true.",
            "system_check": sys_check,
            "example": "rlm_setup_ollama(install=true, start_service=true, pull_model=true)",
        })

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

    return _text_response(result)


async def _handle_setup_ollama_direct(arguments: dict) -> list[TextContent]:
    """Install Ollama via direct download - fully headless, no sudo."""
    install = arguments.get("install", False)
    start_service = arguments.get("start_service", False)
    pull_model = arguments.get("pull_model", False)
    model = arguments.get("model", "gemma3:12b")

    # If no actions specified, show comparison
    if not any([install, start_service, pull_model]):
        return _text_response({
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
        })

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

    return _text_response(result)


async def _handle_ollama_status(arguments: dict) -> list[TextContent]:
    """Check Ollama server status and available models."""
    force_refresh = arguments.get("force_refresh", False)
    status = await _check_ollama_status(force_refresh=force_refresh)

    # Add recommendation based on status
    if status["running"] and status["default_model_available"]:
        status["recommendation"] = "Ollama is ready! Sub-queries will use free local inference by default."
    elif status["running"] and not status["default_model_available"]:
        default_model = DEFAULT_MODELS["ollama"]
        status["recommendation"] = f"Ollama is running but default model not found. Run: ollama pull {default_model}"
    else:
        status["recommendation"] = "Ollama not available. Sub-queries will use Claude API. To enable free local inference, install Ollama and run: ollama serve"

    # Add current best provider
    status["best_provider"] = _get_best_provider()

    return _text_response(status)


async def _handle_load_context(arguments: dict) -> list[TextContent]:
    """Load a large context as an external variable."""
    ctx_name = arguments["name"]
    content = arguments["content"]

    content_hash = _hash_content(content)
    meta = _context_summary(ctx_name, content, hash=content_hash, chunks=None)
    contexts[ctx_name] = {"meta": meta, "content": content}
    _save_context_to_disk(ctx_name, content, meta)

    return _text_response({
        "status": "loaded",
        "name": ctx_name,
        "length": meta["length"],
        "lines": meta["lines"],
        "hash": content_hash,
    })


async def _handle_inspect_context(arguments: dict) -> list[TextContent]:
    """Inspect a loaded context."""
    ctx_name = arguments["name"]
    preview_chars = arguments.get("preview_chars", 500)

    error = _ensure_context_loaded(ctx_name)
    if error:
        return _error_response("context_not_found", error)

    ctx = contexts[ctx_name]
    content = ctx["content"]
    chunk_meta = ctx["meta"].get("chunks")

    summary = _context_summary(
        ctx_name,
        content,
        preview=content[:preview_chars],
        has_chunks=chunk_meta is not None,
        chunk_count=len(chunk_meta) if chunk_meta else 0,
    )
    return _text_response(summary)


async def _handle_chunk_context(arguments: dict) -> list[TextContent]:
    """Chunk a loaded context by strategy."""
    ctx_name = arguments["name"]
    strategy = arguments.get("strategy", "lines")
    size = arguments.get("size", 100)

    error = _ensure_context_loaded(ctx_name)
    if error:
        return _error_response("context_not_found", error)

    content = contexts[ctx_name]["content"]
    chunks = _chunk_content(content, strategy, size)

    chunk_meta = [
        {"index": i, "length": len(chunk), "preview": chunk[:100]}
        for i, chunk in enumerate(chunks)
    ]

    contexts[ctx_name]["meta"]["chunks"] = chunk_meta
    contexts[ctx_name]["chunks"] = chunks

    chunk_dir = CHUNKS_DIR / ctx_name
    chunk_dir.mkdir(exist_ok=True)
    for i, chunk in enumerate(chunks):
        (chunk_dir / f"{i}.txt").write_text(chunk)

    return _text_response({
        "status": "chunked",
        "name": ctx_name,
        "strategy": strategy,
        "chunk_count": len(chunks),
        "chunks": chunk_meta,
    })


async def _handle_get_chunk(arguments: dict) -> list[TextContent]:
    """Get a specific chunk by index."""
    ctx_name = arguments["name"]
    chunk_index = arguments["chunk_index"]

    error = _ensure_context_loaded(ctx_name)
    if error:
        return _error_response("context_not_found", error)

    chunks = contexts[ctx_name].get("chunks")
    if not chunks:
        chunk_path = CHUNKS_DIR / ctx_name / f"{chunk_index}.txt"
        if chunk_path.exists():
            return _text_response(chunk_path.read_text())
        return _error_response(
            "context_not_chunked",
            f"Context '{ctx_name}' has not been chunked yet",
        )

    if chunk_index >= len(chunks):
        return _error_response(
            "chunk_out_of_range",
            f"Chunk index {chunk_index} out of range (max {len(chunks) - 1})",
        )

    return _text_response(chunks[chunk_index])


async def _handle_filter_context(arguments: dict) -> list[TextContent]:
    """Filter context using regex."""
    src_name = arguments["name"]
    out_name = arguments["output_name"]
    pattern = arguments["pattern"]
    mode = arguments.get("mode", "keep")

    error = _ensure_context_loaded(src_name)
    if error:
        return _error_response("context_not_found", error)

    content = contexts[src_name]["content"]
    lines = content.split("\n")
    regex = re.compile(pattern)

    if mode == "keep":
        filtered = [line for line in lines if regex.search(line)]
    else:
        filtered = [line for line in lines if not regex.search(line)]

    new_content = "\n".join(filtered)
    meta = _context_summary(
        out_name,
        new_content,
        hash=_hash_content(new_content),
        source=src_name,
        filter_pattern=pattern,
        filter_mode=mode,
        chunks=None,
    )
    contexts[out_name] = {"meta": meta, "content": new_content}
    _save_context_to_disk(out_name, new_content, meta)

    return _text_response({
        "status": "filtered",
        "name": out_name,
        "original_lines": len(lines),
        "filtered_lines": len(filtered),
        "length": len(new_content),
    })


async def _handle_sub_query(arguments: dict) -> list[TextContent]:
    """Make a sub-LLM call on a chunk or context."""
    query = arguments["query"]
    ctx_name = arguments["context_name"]
    chunk_index = arguments.get("chunk_index")
    provider = arguments.get("provider", "auto")
    model = arguments.get("model")

    # Resolve auto provider and model
    resolved_provider, resolved_model = await _resolve_provider_and_model(provider, model)

    error = _ensure_context_loaded(ctx_name)
    if error:
        return _error_response("context_not_found", error)

    if chunk_index is not None:
        chunks = contexts[ctx_name].get("chunks")
        if not chunks or chunk_index >= len(chunks):
            return _error_response(
                "chunk_not_available", f"Chunk {chunk_index} not available"
            )
        context_content = chunks[chunk_index]
    else:
        context_content = contexts[ctx_name]["content"]

    result, call_error = await _make_provider_call(resolved_provider, resolved_model, query, context_content)

    if call_error:
        return _text_response({
            "error": "provider_error",
            "provider": resolved_provider,
            "model": resolved_model,
            "requested_provider": provider,
            "message": call_error,
        })

    return _text_response({
        "provider": resolved_provider,
        "model": resolved_model,
        "requested_provider": provider if provider == "auto" else None,
        "response": result,
    })


async def _handle_store_result(arguments: dict) -> list[TextContent]:
    """Store a sub-call result for later aggregation."""
    result_name = arguments["name"]
    result = arguments["result"]
    metadata = arguments.get("metadata", {})

    results_file = RESULTS_DIR / f"{result_name}.jsonl"
    with open(results_file, "a") as f:
        f.write(json.dumps({"result": result, "metadata": metadata}) + "\n")

    return _text_response(f"Result stored to '{result_name}'")


async def _handle_get_results(arguments: dict) -> list[TextContent]:
    """Retrieve stored results for aggregation."""
    result_name = arguments["name"]
    results_file = RESULTS_DIR / f"{result_name}.jsonl"

    if not results_file.exists():
        return _text_response(f"No results found for '{result_name}'")

    results = [json.loads(line) for line in results_file.read_text().splitlines()]

    return _text_response({
        "name": result_name,
        "count": len(results),
        "results": results,
    })


async def _handle_list_contexts(_arguments: dict) -> list[TextContent]:
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
            ctx_list.append({
                "name": disk_name,
                "length": meta["length"],
                "lines": meta["lines"],
                "chunked": meta.get("chunks") is not None,
                "disk_only": True,
            })

    return _text_response({"contexts": ctx_list})


async def _handle_sub_query_batch(arguments: dict) -> list[TextContent]:
    """Process multiple chunks in parallel."""
    query = arguments["query"]
    ctx_name = arguments["context_name"]
    chunk_indices = arguments["chunk_indices"]
    provider = arguments.get("provider", "auto")
    model = arguments.get("model")
    concurrency = min(arguments.get("concurrency", 4), 8)

    # Resolve auto provider and model once for the entire batch
    resolved_provider, resolved_model = await _resolve_provider_and_model(provider, model)

    error = _ensure_context_loaded(ctx_name)
    if error:
        return _error_response("context_not_found", error)

    chunks = contexts[ctx_name].get("chunks")
    if not chunks:
        return _error_response(
            "context_not_chunked",
            f"Context '{ctx_name}' has not been chunked yet",
        )

    invalid_indices = [idx for idx in chunk_indices if idx >= len(chunks)]
    if invalid_indices:
        return _error_response(
            "invalid_chunk_indices",
            f"Invalid chunk indices: {invalid_indices} (max: {len(chunks) - 1})",
        )

    semaphore = asyncio.Semaphore(concurrency)

    async def process_chunk(chunk_idx: int) -> dict:
        async with semaphore:
            chunk_content = chunks[chunk_idx]
            result, call_error = await _make_provider_call(
                resolved_provider, resolved_model, query, chunk_content
            )

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

    return _text_response({
        "status": "completed",
        "total_chunks": len(chunk_indices),
        "successful": successful,
        "failed": failed,
        "concurrency": concurrency,
        "provider": resolved_provider,
        "model": resolved_model,
        "requested_provider": provider if provider == "auto" else None,
        "results": results,
    })


async def _handle_auto_analyze(arguments: dict) -> list[TextContent]:
    """Automatically detect content type and analyze with optimal strategy."""
    ctx_name = arguments["name"]
    content = arguments["content"]
    goal = arguments["goal"]
    provider = arguments.get("provider", "auto")
    concurrency = min(arguments.get("concurrency", 4), 8)

    # Load the content
    await _handle_load_context({"name": ctx_name, "content": content})

    # Detect content type
    detection = _detect_content_type(content)
    detected_type = detection["type"]
    confidence = detection["confidence"]

    # Select chunking strategy
    strategy_config = _select_chunking_strategy(detected_type)

    # Chunk the content
    chunk_result = await _handle_chunk_context({
        "name": ctx_name,
        "strategy": strategy_config["strategy"],
        "size": strategy_config["size"],
    })
    chunk_data = json.loads(chunk_result[0].text)
    chunk_count = chunk_data["chunk_count"]

    # Sample if too many chunks (max 20)
    chunk_indices = list(range(chunk_count))
    sampled = False
    if chunk_count > 20:
        step = max(1, chunk_count // 20)
        chunk_indices = list(range(0, chunk_count, step))[:20]
        sampled = True

    # Adapt query for goal and content type
    adapted_query = _adapt_query_for_goal(goal, detected_type)

    # Run batch query
    batch_result = await _handle_sub_query_batch({
        "query": adapted_query,
        "context_name": ctx_name,
        "chunk_indices": chunk_indices,
        "provider": provider,
        "concurrency": concurrency,
    })
    batch_data = json.loads(batch_result[0].text)

    return _text_response({
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
        "successful": batch_data["successful"],
        "failed": batch_data["failed"],
        "results": batch_data["results"],
    })


async def _handle_exec(arguments: dict) -> list[TextContent]:
    """Execute Python code against a loaded context in a sandboxed subprocess."""
    code = arguments["code"]
    ctx_name = arguments["context_name"]
    timeout = arguments.get("timeout", 30)

    # Ensure context is loaded
    error = _ensure_context_loaded(ctx_name)
    if error:
        return _error_response("context_not_found", error)

    content = contexts[ctx_name]["content"]

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
            stdout = stdout[:stdout.index("__RESULT_START__")].strip()

        return _text_response({
            "result": result,
            "stdout": stdout,
            "stderr": stderr,
            "return_code": return_code,
            "timed_out": False,
        })

    except subprocess.TimeoutExpired:
        return _text_response({
            "result": None,
            "stdout": "",
            "stderr": f"Execution timed out after {timeout} seconds",
            "return_code": -1,
            "timed_out": True,
        })
    except Exception as e:
        return _error_response("execution_error", str(e))
    finally:
        # Clean up temp file
        try:
            os.unlink(temp_file)
        except Exception:
            pass


# Tool dispatch table
TOOL_HANDLERS = {
    "rlm_system_check": _handle_system_check,
    "rlm_setup_ollama": _handle_setup_ollama,
    "rlm_setup_ollama_direct": _handle_setup_ollama_direct,
    "rlm_ollama_status": _handle_ollama_status,
    "rlm_load_context": _handle_load_context,
    "rlm_inspect_context": _handle_inspect_context,
    "rlm_chunk_context": _handle_chunk_context,
    "rlm_get_chunk": _handle_get_chunk,
    "rlm_filter_context": _handle_filter_context,
    "rlm_sub_query": _handle_sub_query,
    "rlm_store_result": _handle_store_result,
    "rlm_get_results": _handle_get_results,
    "rlm_list_contexts": _handle_list_contexts,
    "rlm_sub_query_batch": _handle_sub_query_batch,
    "rlm_auto_analyze": _handle_auto_analyze,
    "rlm_exec": _handle_exec,
}


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Route tool calls to their handlers."""
    handler = TOOL_HANDLERS.get(name)
    if handler:
        return await handler(arguments)
    return _text_response(f"Unknown tool: {name}")


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
