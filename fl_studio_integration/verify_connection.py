"""Read-only MCP protocol and live FL Studio verification. No GUI automation."""
from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / "tools/fl-studio-mcp/.venv312/Scripts/python.exe"
REPORT = Path(__file__).with_name("verification.json")


def decode(result):
    if result.isError:
        raise RuntimeError(str(result.content))
    if result.structuredContent is not None:
        value = result.structuredContent
        return value.get("result", value)
    parts = [block.text for block in result.content if block.type == "text"]
    if len(parts) == 1:
        try:
            return json.loads(parts[0])
        except json.JSONDecodeError:
            return parts[0]
    return parts


async def verify():
    report = {
        "checked_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "fl_studio_version": "24.2.2.4597",
        "source_commit": "202b7a28a4dc725350b68064d1be4e1eab852847",
        "mode": "read-only verification; no keyboard or mouse automation",
        "protocol_ok": False,
        "live_read_ok": False,
    }
    params = StdioServerParameters(
        command=str(PYTHON),
        args=["-m", "fl_studio_mcp.server"],
        env={**os.environ, "FL_MIDI_PORT": "FLStudioMCP", "PYTHONUTF8": "1"},
    )
    try:
        with anyio.fail_after(45):
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    init = await session.initialize()
                    report["server"] = init.serverInfo.model_dump()
                    tools = await session.list_tools()
                    report["tool_count"] = len(tools.tools)
                    report["tool_names"] = [tool.name for tool in tools.tools]
                    report["protocol_ok"] = True
                    status = decode(await session.call_tool("fl_connection_status", {}))
                    report["connection"] = status
                    if not isinstance(status, dict) or not status.get("connected"):
                        report["next_step"] = "Create FLStudioMCP in loopMIDI; enable FL Studio MCP Controller on that input in FL Studio."
                    else:
                        checks = [
                            ("transport", "fl_get_transport_status", {}),
                            ("channel_count", "fl_get_channel_count", {}),
                            ("channels", "fl_get_all_channels", {}),
                            ("mixer_track_count", "fl_get_mixer_track_count", {}),
                            ("master", "fl_get_mixer_track_info", {"track": 0}),
                        ]
                        report["reads"] = {}
                        for key, name, args in checks:
                            value = decode(await session.call_tool(name, args))
                            report["reads"][key] = value
                            if (isinstance(value, dict) and "error" in value
                                or isinstance(value, list) and any(isinstance(v, dict) and "error" in v for v in value)
                                or isinstance(value, int) and value < 0):
                                raise RuntimeError(f"Live read failed: {key}: {value}")
                        report["live_read_ok"] = True
    except Exception as exc:
        report["error"] = str(exc)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["live_read_ok"] else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(verify()))
