# 组件来源与模型说明

Chord Studio 的安装分为应用、独立运行环境和可选模型。安装程序从下列上游下载依赖，检查固定文件的 SHA-256；不会上传用户音频。第三方代码、模型权重和音源各自保留原许可，应用仓库的公开可见性不替代这些许可。

## 基础识别与桌面

| 组件 | 用途与来源 | 许可说明 |
|---|---|---|
| Python / python-build-standalone | 由 [uv](https://github.com/astral-sh/uv) 自动配置的独立解释器 | Python 与其依赖的许可随解释器保留；uv 为 MIT / Apache-2.0 |
| Tauri / WebView2 | Windows 桌面窗口；[Tauri](https://github.com/tauri-apps/tauri)、[Microsoft WebView2](https://developer.microsoft.com/microsoft-edge/webview2) | Tauri 为 MIT / Apache-2.0；WebView2 使用微软运行时条款 |
| Visual C++ Runtime | Python 扩展和本地库使用的微软运行组件 | 只从微软下载，执行前检查 Authenticode 签名；已满足要求时复用 |
| FFmpeg 7.1 | 音频解码；[Gyan 官方构建镜像](https://github.com/GyanD/codexffmpeg/releases/tag/7.1) | 所选构建的 LICENSE、README 随工具安装；包含 GPL 组件，参见构建说明 |
| Sonic Annotator 1.7 | 冻结声学前端的 Vamp 宿主；[官方发布](https://github.com/sonic-visualiser/sonic-annotator/releases/tag/sonic-annotator-1.7) | 保留整个官方文件夹，包括 COPYING、CITATION、README 与运行库 |
| NNLS Chroma / Chordino | Matthias Mauch、Chris Cannam 的音高特征；[项目说明](https://isophonics.net/nnls-chroma) | GPL；从 [Vamp Plugin Pack 2.0](https://github.com/vamp-plugins/vamp-plugin-pack/releases/tag/v2.0) 提取经校验的插件及原 COPYING、README、CITATION、RDF 描述，不执行系统级插件安装程序 |
| madmom 0.16.1 | 节拍和冻结和弦前端；[代码与模型说明](https://github.com/CPJKU/madmom) | 代码与模型许可不同。随包模型使用 **CC BY-NC-SA 4.0**，包含非商业用途限制；安装后完整许可位于 `runtime/madmom/Lib/site-packages/madmom/models/LICENSE` |

NNLS 二进制与官方 Vamp Plugin Pack 2.0 内嵌资源一致。对应插件源码版本为 `4c5f214a75cb354f8d8c933e161377c5b4d83713`；构建入口、依赖版本和资源清单见官方发布标签的 [repoint-lock.json](https://github.com/vamp-plugins/vamp-plugin-pack/blob/v2.0/repoint-lock.json) 与 `nnls-chroma.pro`。安装器只提取功能所需文件，不重新发布整套插件。

## 分轨与 MIDI

| 功能 | 模型 / 实现来源 |
|---|---|
| 标准四轨、精细四轨 | [Demucs](https://github.com/facebookresearch/demucs)，`htdemucs` / `htdemucs_ft`；官方模型下载地址与完整摘要列于 `packaging/demucs-models.json` |
| 六轨、主唱／和声、鼓组、人声优先 | [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator) 使用的公开模型配置与权重；具体作者及文件名见 `packaging/model-packs.json` |
| 提琴组两个版本 | [mvsepless_resources](https://huggingface.co/noblebarkrr/mvsepless_resources)，固定仓库修订 `283025218e6447db47da1ad18df3c2f67ca8d66d` 的 BS-Roformer 弦乐模型 |
| 木吉他／电吉他 | MVSEP 53-stem 系列的吉他目标模型；安装清单保留下载地址、大小和完整摘要 |
| 旋律／伴奏吉他 | 项目既有 `guitar_role_v1` 适配权重与 Listra92 配置。该适配模型沿用上游架构和权重约束；不能仅凭适配而主张更宽泛许可 |
| 钢琴 MIDI | [Transkun](https://github.com/Yujia-Yan/Transkun)，V2 No Pedal Extension；项目代码 MIT，安装保留 Python 包原许可和模型资源 |
| 整曲 MIDI 重构 | [YourMT3](https://huggingface.co/spaces/mimbres/YourMT3)、ADT-str，以及 [Basic Pitch](https://github.com/spotify/basic-pitch) 等；每个模型的固定修订与文件摘要见 `packaging/model-metadata` 和模型清单 |
| MIDI 试听 | Frank Wen 的 FluidR3 GM（音源归属和 MIT 风格许可随 COPYING 保留），以及 [FluidSynth](https://github.com/FluidSynth/fluidsynth/releases/tag/v2.6.2)（LGPL，保留官方包的许可文件） |

**代码许可不代表模型权重的许可。** 部分公开权重缺少单独、清晰的用途授权，安装清单里的下载地址不构成商业授权。使用者应根据原作者说明确认用途；本项目不把这些第三方权重统一改称为 MIT，也不声称整套模型可无条件商用。

FluidR3 下载自 [Debian 上游源包](https://ftp.debian.org/debian/pool/main/f/fluid-soundfont/)，只安装 GM 音源、COPYING 与 README。试听音色是通用音源，不代表复刻原曲音色。

## 可重建环境

`packaging/requirements/` 保存固定版本和下载摘要。需要编译的少数 Python 扩展提供 Windows 预编译 wheel，让使用者不必安装编译器；wheel 的版本、大小和摘要见 `packaging/built-wheels.json`。包内上游许可和版权文本保持不变。所有 Python 依赖的完整许可可在各环境的 `Lib/site-packages/*.dist-info` 中查阅。

钢琴、分轨和重构环境共享安装目录内的一套 PyTorch/CUDA 库；不依赖开发者电脑的 Python 或其他磁盘目录。GPU 驱动由硬件厂商提供；没有可用 CUDA 时使用现有 CPU 路径，耗时会增加。

安装工具和模型包的固定文件都进行摘要校验。微软 Evergreen 组件会持续更新，因此使用官方地址和有效的微软数字签名校验，不伪称固定到某个旧版本。
