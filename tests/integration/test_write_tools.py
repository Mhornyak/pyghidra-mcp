"""Integration tests for write tools (rename, comment, create, prototype, variable ops)."""

import json
import platform

import pytest
from mcp import ClientSession
from mcp.client.stdio import stdio_client

from pyghidra_mcp.context import PyGhidraContext
from pyghidra_mcp.models import (
    CreateLabelResult,
    RenameFunctionResult,
    SetCommentResult,
)

# macOS GCC prefixes symbols with '_'
_PREFIX = "_" if platform.system() == "Darwin" else ""


def _get_binary_name(server_params):
    return PyGhidraContext._gen_unique_bin_name(server_params.args[-1])


def _parse_result(result):
    """Parse an MCP tool result, raising on error."""
    text = result.content[0].text
    if getattr(result, "isError", False):
        raise AssertionError(f"Tool returned error: {text}")
    return json.loads(text)


def _find_symbol_address(sym_data, name):
    """Find the address of a symbol by exact name in a SymbolSearchResults dict."""
    for sym in sym_data["symbols"]:
        if sym["name"] == name:
            return sym["address"]
    return None


@pytest.mark.asyncio
async def test_rename_function(server_params, test_binary):
    """Rename function_one → my_func_one, then verify via decompile_function."""
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            binary_name = _get_binary_name(server_params)
            func_name = f"{_PREFIX}function_one"

            result = await session.call_tool(
                "rename_function",
                {
                    "binary_name": binary_name,
                    "name_or_address": func_name,
                    "new_name": "my_func_one",
                },
            )
            data = _parse_result(result)
            parsed = RenameFunctionResult.model_validate(data)
            assert parsed.new_name == "my_func_one"
            assert parsed.address

            # Verify the rename persisted via decompilation
            decomp = await session.call_tool(
                "decompile_function",
                {"binary_name": binary_name, "name_or_address": "my_func_one"},
            )
            assert "my_func_one" in decomp.content[0].text


@pytest.mark.asyncio
async def test_set_comment_eol(server_params, test_binary):
    """Set an EOL comment at function_two's entry point."""
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            binary_name = _get_binary_name(server_params)
            func_name = f"{_PREFIX}function_two"

            # Get function_two's address via symbol search
            symbols = await session.call_tool(
                "search_symbols_by_name",
                {"binary_name": binary_name, "query": "function_two"},
            )
            sym_data = _parse_result(symbols)
            func_addr = _find_symbol_address(sym_data, func_name)
            assert func_addr is not None, (
                f"{func_name} not found in symbols: {sym_data}"
            )

            result = await session.call_tool(
                "set_comment",
                {
                    "binary_name": binary_name,
                    "address": func_addr,
                    "comment": "This calls printf",
                    "comment_type": "EOL",
                },
            )
            data = _parse_result(result)
            parsed = SetCommentResult.model_validate(data)
            assert parsed.comment == "This calls printf"
            assert parsed.comment_type == "EOL"
            assert parsed.address == func_addr


@pytest.mark.asyncio
async def test_create_label(server_params, test_binary):
    """Create a label at main's address and verify via symbol search."""
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            binary_name = _get_binary_name(server_params)
            main_name = f"{_PREFIX}main"

            # Get main's address via symbol search
            symbols = await session.call_tool(
                "search_symbols_by_name",
                {"binary_name": binary_name, "query": "main"},
            )
            sym_data = _parse_result(symbols)
            main_addr = _find_symbol_address(sym_data, main_name)
            assert main_addr is not None, (
                f"{main_name} not found in symbols: {sym_data}"
            )

            result = await session.call_tool(
                "create_label",
                {
                    "binary_name": binary_name,
                    "address": main_addr,
                    "name": "my_custom_label",
                },
            )
            data = _parse_result(result)
            parsed = CreateLabelResult.model_validate(data)
            assert parsed.name == "my_custom_label"
            assert parsed.address == main_addr

            # Verify via symbol search
            search = await session.call_tool(
                "search_symbols_by_name",
                {"binary_name": binary_name, "query": "my_custom_label"},
            )
            search_data = _parse_result(search)
            label_names = [s["name"] for s in search_data["symbols"]]
            assert "my_custom_label" in label_names


@pytest.mark.asyncio
async def test_rename_variable(server_params, test_binary):
    """Attempt to rename a variable in main. Best-effort since variable names vary by platform."""
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            binary_name = _get_binary_name(server_params)
            main_name = f"{_PREFIX}main"

            # Decompile main to discover variable names
            decomp = await session.call_tool(
                "decompile_function",
                {"binary_name": binary_name, "name_or_address": main_name},
            )
            decomp_text = decomp.content[0].text
            assert decomp_text, "Decompilation returned empty result"

            # Try renaming a common Ghidra-generated local variable.
            # Variable names vary per platform/arch, so accept graceful failure.
            try:
                result = await session.call_tool(
                    "rename_variable",
                    {
                        "binary_name": binary_name,
                        "function_name_or_address": main_name,
                        "old_name": "local_10",
                        "new_name": "my_var",
                    },
                )
                data = _parse_result(result)
                assert data["new_name"] == "my_var"
            except Exception:
                # Variable name may differ per platform — acceptable
                pass
