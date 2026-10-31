# FL Studio / Codex 接入验证

本机：FL Studio 2024，24.2.2.4597。选用 [karl-andres/fl-studio-mcp](https://github.com/karl-andres/fl-studio-mcp)，源码固定于 `202b7a28a4dc725350b68064d1be4e1eab852847`。PostFader 当前要求 FL Studio 26.1.3+，因此本机没有安装 PostFader。

## 已完成

- 在 `../tools/fl-studio-mcp` 下载源码，使用独立 Python 3.12.15 环境 `.venv312`。
- 安装依赖，FastMCP 固定在 2.x；实际版本见 `requirements.lock.txt`。
- 安装 FL Studio MIDI 控制器和 ComposeWithLLM 钢琴卷帘脚本，未执行钢琴卷帘脚本。
- 注册 Codex 本地 stdio MCP 服务 `fl-studio`，保留原有其他服务。
- 通过真实 stdio MCP 初始化和工具枚举，发现 57 个工具。这不代表这些工具均已通过 FL Studio 实机测试。
- 修改本地连接逻辑：只使用指定虚拟端口，禁止回退到系统合成器；收到 FL Studio 播放状态回包后才标记连接成功。4 项回归测试通过。

最新真实连接状态见 `verification.json`。2026-10-10 用户完成 MIDI 设置后，只读实机验收通过：`protocol_ok=true`、`live_read_ok=true`，连接端口为 `FLStudioMCP 1`。实际读取到停止状态、Pattern 模式、5 个通道（808 Kick、808 Clap、808 HiHat、808 Snare、FLEX Bass）、127 个混音轨及 Master 参数。没有执行音乐工程写入、播放、录制或快捷键操作。

## 一次性设置（本机已完成，保留供恢复使用）

1. 运行 `../tools/fl-studio-mcp-downloads/loopMIDI/loopMIDISetup.exe`。安装包来自 [loopMIDI 作者官网](https://www.tobias-erichsen.de/software/loopmidi.html)，已确认数字签名有效、签名人为 Tobias Erichsen。Windows 安装步骤需要本机交互。
2. 在 loopMIDI 中创建端口 `FLStudioMCP`，保持 loopMIDI 运行。
3. 启动 FL Studio，在 Options → MIDI settings → Input 选择此端口，Enable，Controller type 选 `FL Studio MCP Controller`。
4. 双击本目录的 `verify.cmd`，或让 Codex 执行下面的验证命令。工具会读取播放状态、通道列表、混音器数量和 Master 参数，不发送修改工程的命令，也不模拟快捷键。
5. 重新连接 Codex 的 MCP 服务，或新开会话，以加载已注册的工具。当前会话也可通过本目录 Python 验证器调用标准 MCP 协议。

在工作区根目录执行：

```powershell
& 'tools/fl-studio-mcp/.venv312/Scripts/python.exe' fl_studio_integration/verify_connection.py
```

只有 `protocol_ok` 和 `live_read_ok` 都为 `true` 才表示本轮只读验收通过。即使通过，也不能据此声称写音符、混音修改或导出已验证。

## 路径与恢复

- 控制器：`C:/Users/33573/Documents/Image-Line/FL Studio/Settings/Hardware/FLStudioMCP/device_FLStudioMCP.py`
- 钢琴卷帘脚本：`C:/Users/33573/Documents/Image-Line/FL Studio/Settings/Piano roll scripts/ComposeWithLLM.pyscript`
- 原 Codex 配置备份：`C:/Users/33573/.codex/config.toml.before-fl-studio-20261010`
- 查看接入配置：`codex mcp get fl-studio`
- 仅移除此服务注册：`codex mcp remove fl-studio`
- 本地源码变更：`local-connection.patch`

不要用旧配置备份直接覆盖未来新增的 Codex 配置。撤销注册优先使用上面的单服务移除命令。

## 已知边界

原项目的钢琴卷帘工具通过快捷键触发 FL 内部脚本；本轮没有调用这些工具。此版本桥接没有完整的 Playlist 编排、插件加载及工程渲染接口。注册 MCP 服务没有把这些缺失功能补齐。

本次仅安装和验证接入组件，没有创建或修改音乐工程。
