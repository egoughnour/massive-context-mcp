"""Tests for Ollama setup and status tools.

These tests verify the setup tools work correctly without actually
installing anything - they mock subprocess calls and network requests.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Set test data directory before importing server
TEST_DATA_DIR = tempfile.mkdtemp(prefix="rlm_test_")
os.environ["RLM_DATA_DIR"] = TEST_DATA_DIR

import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rlm_mcp_server import (
    _check_ollama_status,
    _check_system_requirements,
    _get_best_provider,
    _ollama_status_cache,
    _setup_ollama,
    _setup_ollama_direct,
)

# Import FastMCP tool wrappers and extract underlying functions
from rlm_mcp_server import (
    rlm_ollama_status as _rlm_ollama_status_tool,
)
from rlm_mcp_server import (
    rlm_setup_ollama as _rlm_setup_ollama_tool,
)
from rlm_mcp_server import (
    rlm_setup_ollama_direct as _rlm_setup_ollama_direct_tool,
)
from rlm_mcp_server import (
    rlm_system_check as _rlm_system_check_tool,
)

# FastMCP wraps functions in FunctionTool objects - extract the underlying fn
rlm_ollama_status = _rlm_ollama_status_tool.fn
rlm_setup_ollama = _rlm_setup_ollama_tool.fn
rlm_setup_ollama_direct = _rlm_setup_ollama_direct_tool.fn
rlm_system_check = _rlm_system_check_tool.fn


@pytest.fixture(autouse=True)
def reset_ollama_cache():
    """Reset Ollama status cache before each test."""
    _ollama_status_cache["checked_at"] = None
    _ollama_status_cache["running"] = False
    _ollama_status_cache["models"] = []
    _ollama_status_cache["default_model_available"] = False
    yield


class TestCheckSystemRequirements:
    """Tests for _check_system_requirements function."""

    def test_returns_dict_with_required_keys(self):
        """Should return dict with all expected keys."""
        result = _check_system_requirements()

        expected_keys = [
            "platform",
            "machine",
            "is_macos",
            "is_apple_silicon",
            "ram_gb",
            "ram_sufficient",
            "homebrew_installed",
            "ollama_installed",
            "meets_requirements",
            "issues",
            "recommendations",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    @patch("platform.system", return_value="Darwin")
    @patch("platform.machine", return_value="arm64")
    @patch("subprocess.run")
    def test_macos_apple_silicon_detected(self, mock_run, mock_machine, mock_system):
        """Should detect macOS with Apple Silicon."""
        # Mock sysctl for RAM (32GB)
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="34359738368\n",  # 32GB in bytes
        )

        result = _check_system_requirements()

        assert result["is_macos"] is True
        assert result["is_apple_silicon"] is True

    @patch("platform.system", return_value="Linux")
    @patch("platform.machine", return_value="x86_64")
    def test_linux_not_macos(self, mock_machine, mock_system):
        """Should detect non-macOS system."""
        result = _check_system_requirements()

        assert result["is_macos"] is False
        assert result["is_apple_silicon"] is False
        assert result["meets_requirements"] is False
        assert any("Not macOS" in issue for issue in result["issues"])

    @patch("platform.system", return_value="Darwin")
    @patch("platform.machine", return_value="x86_64")
    def test_intel_mac_detected(self, mock_machine, mock_system):
        """Should detect Intel Mac (not Apple Silicon)."""
        result = _check_system_requirements()

        assert result["is_macos"] is True
        assert result["is_apple_silicon"] is False
        assert any("Not Apple Silicon" in issue for issue in result["issues"])

    @patch("platform.system", return_value="Darwin")
    @patch("platform.machine", return_value="arm64")
    @patch("subprocess.run")
    def test_ram_check_sufficient(self, mock_run, mock_machine, mock_system):
        """Should detect sufficient RAM (>= 16GB)."""

        def run_side_effect(cmd, **kwargs):
            if "hw.memsize" in cmd:
                return MagicMock(returncode=0, stdout="34359738368\n")  # 32GB
            elif "which" in cmd and "brew" in cmd:
                return MagicMock(returncode=0, stdout="/opt/homebrew/bin/brew\n")
            elif "which" in cmd and "ollama" in cmd:
                return MagicMock(returncode=1, stdout="")
            return MagicMock(returncode=0, stdout="")

        mock_run.side_effect = run_side_effect
        result = _check_system_requirements()

        assert result["ram_gb"] == 32.0
        assert result["ram_sufficient"] is True

    @patch("platform.system", return_value="Darwin")
    @patch("platform.machine", return_value="arm64")
    @patch("subprocess.run")
    def test_ram_check_insufficient(self, mock_run, mock_machine, mock_system):
        """Should detect insufficient RAM (< 16GB)."""

        def run_side_effect(cmd, **kwargs):
            if "hw.memsize" in cmd:
                return MagicMock(returncode=0, stdout="8589934592\n")  # 8GB
            return MagicMock(returncode=1, stdout="")

        mock_run.side_effect = run_side_effect
        result = _check_system_requirements()

        assert result["ram_gb"] == 8.0
        assert result["ram_sufficient"] is False
        assert any("Insufficient RAM" in issue for issue in result["issues"])


class TestRlmSystemCheck:
    """Tests for the rlm_system_check tool."""

    @pytest.mark.asyncio
    async def test_returns_dict_response(self):
        """Should return dict with summary."""
        result = await rlm_system_check()

        assert isinstance(result, dict)
        assert "summary" in result

    @pytest.mark.asyncio
    @patch("rlm_mcp_server._check_system_requirements")
    async def test_summary_when_meets_requirements(self, mock_check):
        """Should show positive summary when requirements met."""
        mock_check.return_value = {
            "is_macos": True,
            "is_apple_silicon": True,
            "ram_gb": 32.0,
            "ram_sufficient": True,
            "homebrew_installed": True,
            "ollama_installed": False,
            "meets_requirements": True,
            "chip": "Apple M2 Pro",
            "issues": [],
            "recommendations": [],
        }

        result = await rlm_system_check()

        assert "ready" in result["summary"].lower()
        assert "32.0GB" in result["summary"]

    @pytest.mark.asyncio
    @patch("rlm_mcp_server._check_system_requirements")
    async def test_summary_when_issues_found(self, mock_check):
        """Should show issue count when requirements not met."""
        mock_check.return_value = {
            "is_macos": False,
            "is_apple_silicon": False,
            "ram_gb": 8.0,
            "ram_sufficient": False,
            "homebrew_installed": False,
            "ollama_installed": False,
            "meets_requirements": False,
            "issues": ["Not macOS", "Insufficient RAM"],
            "recommendations": ["Use macOS", "Get more RAM"],
        }

        result = await rlm_system_check()

        assert "2 issue" in result["summary"]


class TestCheckOllamaStatus:
    """Tests for _check_ollama_status function."""

    @pytest.mark.asyncio
    async def test_returns_not_running_without_httpx(self):
        """Should return not running if httpx not available."""
        with patch("rlm_mcp_server.HAS_HTTPX", False):
            result = await _check_ollama_status(force_refresh=True)

            assert result["running"] is False
            assert result["error"] == "httpx not installed"

    @pytest.mark.asyncio
    async def test_caches_result(self):
        """Should cache result and return cached on subsequent calls."""
        with patch("rlm_mcp_server.HAS_HTTPX", True):
            with patch("httpx.AsyncClient") as mock_client:
                mock_response = MagicMock()
                mock_response.json.return_value = {"models": [{"name": "gemma3:12b"}]}
                mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)

                # First call - should hit network
                result1 = await _check_ollama_status(force_refresh=True)
                assert result1["cached"] is False

                # Second call - should use cache
                result2 = await _check_ollama_status(force_refresh=False)
                assert result2["cached"] is True

    @pytest.mark.asyncio
    async def test_force_refresh_bypasses_cache(self):
        """Should bypass cache when force_refresh=True."""
        # Set up cache with old data
        _ollama_status_cache["checked_at"] = 999999999999
        _ollama_status_cache["running"] = True
        _ollama_status_cache["models"] = ["old_model"]

        with patch("rlm_mcp_server.HAS_HTTPX", True):
            with patch("httpx.AsyncClient") as mock_client:
                mock_response = MagicMock()
                mock_response.json.return_value = {"models": [{"name": "new_model"}]}
                mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)

                result = await _check_ollama_status(force_refresh=True)

                assert result["cached"] is False
                assert "new_model" in result["models"]

    @pytest.mark.asyncio
    async def test_connection_refused_handling(self):
        """Should handle connection refused gracefully."""
        import httpx

        with patch("rlm_mcp_server.HAS_HTTPX", True):
            with patch("httpx.AsyncClient") as mock_client:
                mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                    side_effect=httpx.ConnectError("Connection refused")
                )

                result = await _check_ollama_status(force_refresh=True)

                assert result["running"] is False
                assert result["error"] == "connection_refused"
                assert "ollama serve" in result["message"]


class TestRlmOllamaStatus:
    """Tests for the rlm_ollama_status tool."""

    @pytest.mark.asyncio
    async def test_returns_recommendation_when_ready(self):
        """Should include recommendation when Ollama is ready."""
        # Set up cache to indicate Ollama is ready
        _ollama_status_cache["running"] = True
        _ollama_status_cache["models"] = ["gemma3:12b"]
        _ollama_status_cache["default_model_available"] = True

        with patch("rlm_mcp_server._check_ollama_status", new_callable=AsyncMock) as mock:
            mock.return_value = {
                "running": True,
                "models": ["gemma3:12b"],
                "default_model_available": True,
            }

            result = await rlm_ollama_status(force_refresh=False)

            assert "ready" in result["recommendation"].lower()
            assert result["best_provider"] == "ollama"

    @pytest.mark.asyncio
    async def test_returns_recommendation_when_not_running(self):
        """Should include help when Ollama not running."""
        with patch("rlm_mcp_server._check_ollama_status", new_callable=AsyncMock) as mock:
            mock.return_value = {
                "running": False,
                "models": [],
                "default_model_available": False,
            }

            result = await rlm_ollama_status(force_refresh=True)

            assert "not available" in result["recommendation"].lower()
            assert result["best_provider"] == "claude-sdk"


class TestGetBestProvider:
    """Tests for _get_best_provider function."""

    def test_returns_ollama_when_available(self):
        """Should return ollama when running with model."""
        _ollama_status_cache["running"] = True
        _ollama_status_cache["default_model_available"] = True

        assert _get_best_provider() == "ollama"

    def test_returns_claude_sdk_when_ollama_not_running(self):
        """Should return claude-sdk when Ollama not running."""
        _ollama_status_cache["running"] = False
        _ollama_status_cache["default_model_available"] = False

        assert _get_best_provider() == "claude-sdk"

    def test_returns_claude_sdk_when_model_not_available(self):
        """Should return claude-sdk when model not available."""
        _ollama_status_cache["running"] = True
        _ollama_status_cache["default_model_available"] = False

        assert _get_best_provider() == "claude-sdk"


class TestSetupOllama:
    """Tests for _setup_ollama (Homebrew method)."""

    @pytest.mark.asyncio
    @patch("platform.system", return_value="Linux")
    async def test_fails_on_non_macos(self, mock_system):
        """Should fail on non-macOS systems."""
        result = await _setup_ollama(install=True)

        assert result["success"] is False
        assert any("macOS" in err for err in result["errors"])

    @pytest.mark.asyncio
    @patch("rlm_mcp_server._check_system_requirements")
    async def test_fails_without_homebrew(self, mock_check):
        """Should fail when Homebrew not installed."""
        mock_check.return_value = {
            "is_macos": True,
            "is_apple_silicon": True,
            "ram_gb": 32.0,
            "ram_sufficient": True,
            "homebrew_installed": False,
            "ollama_installed": False,
        }

        result = await _setup_ollama(install=True)

        assert result["success"] is False
        assert any("Homebrew" in err for err in result["errors"])

    @pytest.mark.asyncio
    @patch("rlm_mcp_server._check_system_requirements")
    @patch("subprocess.run")
    async def test_skips_if_already_installed(self, mock_run, mock_check):
        """Should skip installation if Ollama already installed."""
        mock_check.return_value = {
            "is_macos": True,
            "is_apple_silicon": True,
            "ram_gb": 32.0,
            "ram_sufficient": True,
            "homebrew_installed": True,
            "ollama_installed": True,
        }

        result = await _setup_ollama(install=True)

        assert result["success"] is True
        assert any("already installed" in skip for skip in result["actions_skipped"])

    @pytest.mark.asyncio
    @patch("rlm_mcp_server._check_system_requirements")
    @patch("subprocess.run")
    async def test_brew_install_called(self, mock_run, mock_check):
        """Should call brew install when installing."""
        mock_check.return_value = {
            "is_macos": True,
            "is_apple_silicon": True,
            "ram_gb": 32.0,
            "ram_sufficient": True,
            "homebrew_installed": True,
            "ollama_installed": False,
        }
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        result = await _setup_ollama(install=True)

        assert result["success"] is True
        # Check that brew install was called
        brew_calls = [c for c in mock_run.call_args_list if "brew" in str(c)]
        assert len(brew_calls) > 0

    @pytest.mark.asyncio
    @patch("rlm_mcp_server._check_system_requirements")
    async def test_ram_check_for_large_model(self, mock_check):
        """Should fail when RAM insufficient for gemma3:12b."""
        mock_check.return_value = {
            "is_macos": True,
            "is_apple_silicon": True,
            "ram_gb": 8.0,
            "ram_sufficient": False,
            "homebrew_installed": True,
            "ollama_installed": True,
        }

        result = await _setup_ollama(pull_model=True, model="gemma3:12b")

        assert result["success"] is False
        assert any("Insufficient RAM" in err for err in result["errors"])


class TestSetupOllamaDirect:
    """Tests for _setup_ollama_direct (direct download method)."""

    @pytest.mark.asyncio
    @patch("platform.system", return_value="Linux")
    async def test_fails_on_non_macos(self, mock_system):
        """Should fail on non-macOS systems."""
        with patch("rlm_mcp_server._check_system_requirements") as mock_check:
            mock_check.return_value = {
                "is_macos": False,
                "is_apple_silicon": False,
                "ram_gb": 32.0,
                "ram_sufficient": True,
            }

            result = await _setup_ollama_direct(install=True)

            assert result["success"] is False
            assert any("macOS" in err for err in result["errors"])

    @pytest.mark.asyncio
    @patch("rlm_mcp_server._check_system_requirements")
    async def test_skips_if_app_exists(self, mock_check):
        """Should skip if ~/Applications/Ollama.app already exists."""
        mock_check.return_value = {
            "is_macos": True,
            "is_apple_silicon": True,
            "ram_gb": 32.0,
            "ram_sufficient": True,
        }

        with patch.object(Path, "exists", return_value=True):
            result = await _setup_ollama_direct(install=True)

            assert result["success"] is True
            assert any("already installed" in skip for skip in result["actions_skipped"])

    @pytest.mark.asyncio
    @patch("rlm_mcp_server._check_system_requirements")
    @patch("subprocess.run")
    @patch("shutil.move")
    @patch("shutil.rmtree")
    async def test_download_and_extract(self, mock_rmtree, mock_move, mock_run, mock_check):
        """Should download, extract, and install to ~/Applications."""
        mock_check.return_value = {
            "is_macos": True,
            "is_apple_silicon": True,
            "ram_gb": 32.0,
            "ram_sufficient": True,
        }
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        # Mock Path.exists() to return False for app_path but True for extracted_app
        original_exists = Path.exists

        def mock_exists(self):
            if "Ollama.app" in str(self) and "Applications" in str(self):
                return False  # App not installed yet
            if "ollama-extract" in str(self) and "Ollama.app" in str(self):
                return True  # Extracted app exists
            return original_exists(self)

        with patch.object(Path, "exists", mock_exists):
            with patch.object(Path, "mkdir"):
                with patch.object(Path, "unlink"):
                    await _setup_ollama_direct(install=True)

        # Check that curl was called with the download URL
        curl_calls = [c for c in mock_run.call_args_list if "curl" in str(c)]
        assert len(curl_calls) > 0
        assert "ollama.com" in str(curl_calls[0])

    @pytest.mark.asyncio
    @patch("rlm_mcp_server._check_system_requirements")
    @patch("subprocess.run")
    async def test_download_failure_handling(self, mock_run, mock_check):
        """Should handle download failure gracefully."""
        mock_check.return_value = {
            "is_macos": True,
            "is_apple_silicon": True,
            "ram_gb": 32.0,
            "ram_sufficient": True,
        }
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="curl: (7) Failed to connect")

        with patch.object(Path, "exists", return_value=False):
            with patch.object(Path, "mkdir"):
                result = await _setup_ollama_direct(install=True)

        assert result["success"] is False
        assert any("Download failed" in err for err in result["errors"])

    @pytest.mark.asyncio
    @patch("rlm_mcp_server._check_system_requirements")
    async def test_ram_check_for_large_model(self, mock_check):
        """Should fail when RAM insufficient for gemma3:12b."""
        mock_check.return_value = {
            "is_macos": True,
            "is_apple_silicon": True,
            "ram_gb": 8.0,
            "ram_sufficient": False,
        }

        with patch.object(Path, "exists", return_value=True):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=1, stdout="")
                result = await _setup_ollama_direct(pull_model=True, model="gemma3:12b")

        assert result["success"] is False
        assert any("Insufficient RAM" in err for err in result["errors"])


class TestRlmSetupOllama:
    """Tests for the rlm_setup_ollama tool."""

    @pytest.mark.asyncio
    async def test_no_actions_returns_help(self):
        """Should return help message when no actions specified."""
        result = await rlm_setup_ollama()

        assert "No actions specified" in result["message"]
        assert "example" in result

    @pytest.mark.asyncio
    @patch("rlm_mcp_server._setup_ollama", new_callable=AsyncMock)
    async def test_passes_arguments_correctly(self, mock_setup):
        """Should pass all arguments to _setup_ollama."""
        mock_setup.return_value = {
            "success": True,
            "actions_taken": ["test"],
            "actions_skipped": [],
            "errors": [],
        }

        await rlm_setup_ollama(
            install=True,
            start_service=True,
            pull_model=True,
            model="gemma3:4b",
        )

        mock_setup.assert_called_once_with(
            install=True,
            start_service=True,
            pull_model=True,
            model="gemma3:4b",
        )


class TestRlmSetupOllamaDirect:
    """Tests for the rlm_setup_ollama_direct tool."""

    @pytest.mark.asyncio
    async def test_no_actions_returns_comparison(self):
        """Should return comparison info when no actions specified."""
        result = await rlm_setup_ollama_direct()

        assert "No actions specified" in result["message"]
        assert "advantages" in result
        assert "disadvantages" in result
        assert "No Homebrew required" in result["advantages"]

    @pytest.mark.asyncio
    @patch("rlm_mcp_server._setup_ollama_direct", new_callable=AsyncMock)
    async def test_includes_path_setup_in_summary(self, mock_setup):
        """Should include PATH setup note in summary."""
        mock_setup.return_value = {
            "success": True,
            "actions_taken": ["Installed"],
            "actions_skipped": [],
            "errors": [],
            "path_setup": {
                "add_to_path": 'export PATH="$HOME/Applications/Ollama.app/Contents/Resources:$PATH"',
            },
        }

        result = await rlm_setup_ollama_direct(install=True)

        assert "PATH" in result["summary"]


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Should handle subprocess timeout gracefully."""
        import subprocess

        with patch("rlm_mcp_server._check_system_requirements") as mock_check:
            mock_check.return_value = {
                "is_macos": True,
                "is_apple_silicon": True,
                "ram_gb": 32.0,
                "ram_sufficient": True,
                "homebrew_installed": True,
                "ollama_installed": False,
            }

            with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 300)):
                result = await _setup_ollama(install=True)

                assert result["success"] is False
                assert any("timed out" in err for err in result["errors"])

    @pytest.mark.asyncio
    async def test_generic_exception_handling(self):
        """Should handle unexpected exceptions gracefully."""
        with patch("rlm_mcp_server._check_system_requirements") as mock_check:
            mock_check.return_value = {
                "is_macos": True,
                "is_apple_silicon": True,
                "ram_gb": 32.0,
                "ram_sufficient": True,
                "homebrew_installed": True,
                "ollama_installed": False,
            }

            with patch("subprocess.run", side_effect=Exception("Unexpected error")):
                result = await _setup_ollama(install=True)

                assert result["success"] is False
                assert any("error" in err.lower() for err in result["errors"])

    @pytest.mark.asyncio
    @patch("rlm_mcp_server._check_system_requirements")
    @patch("rlm_mcp_server._check_ollama_status", new_callable=AsyncMock)
    @patch("subprocess.run")
    async def test_service_already_running(self, mock_run, mock_status, mock_check):
        """Should skip starting service if already running."""
        mock_check.return_value = {
            "is_macos": True,
            "is_apple_silicon": True,
            "ram_gb": 32.0,
            "ram_sufficient": True,
            "homebrew_installed": True,
            "ollama_installed": True,
        }
        mock_status.return_value = {"running": True, "models": ["gemma3:12b"]}

        result = await _setup_ollama(start_service=True)

        assert result["success"] is True
        assert any("already running" in skip for skip in result["actions_skipped"])

    @pytest.mark.asyncio
    @patch("rlm_mcp_server._check_system_requirements")
    @patch("rlm_mcp_server._check_ollama_status", new_callable=AsyncMock)
    async def test_model_already_pulled(self, mock_status, mock_check):
        """Should skip pulling model if already available."""
        mock_check.return_value = {
            "is_macos": True,
            "is_apple_silicon": True,
            "ram_gb": 32.0,
            "ram_sufficient": True,
            "homebrew_installed": True,
            "ollama_installed": True,
        }
        mock_status.return_value = {"running": True, "models": ["gemma3:12b"]}

        result = await _setup_ollama(pull_model=True, model="gemma3:12b")

        assert result["success"] is True
        assert any("already available" in skip for skip in result["actions_skipped"])
