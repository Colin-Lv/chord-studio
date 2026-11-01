# 用分轨（stems）提升和弦识别：文献扫描与可复现性评估

> 检索日期：2026-11-01
> 范围：HPSS / 人声分离 / 低音分离 作为和弦识别前端；现代分离器（Demucs、BS-RoFormer 时代）的 stems 直接喂和弦模型；反向的"标签条件化分离"（score-informed）；评测与坑。
> 检索方式：arXiv API、OpenAlex、Crossref、Zenodo、出版方 PDF 直取（`apsipa.org` 的 PDF 可用），并用 `pdftotext` 抽取正文核对数字。
> **环境说明**：本会话的 `web_fetch` 对所有域名都返回 "resolves to a non-public IP address"（DNS 被劫持到内网），因此所有网页/PDF 改用 HTTP 直取 + 本地文本抽取完成。`web_search` 正常。这一限制不影响结论，但意味着凡是只有付费墙页面、我拿不到正文的数字，一律标 **未核实**。
> 标注约定：`未核实` = 我看到了该数字的二手转述或只看到摘要而没有正文表格；`✅` = 我直接从论文 PDF/HTML 正文里读到了该数字；`❌` = 明确不存在。

---

## 0. 结论先行（TL;DR）

1. **"把 stems 喂给和弦模型"这件事的直接、受控、真实数据证据非常少**。目前最贴近我们约束的一篇（冻结/预训练和弦模型 + 现代分离器 + 真实曲目）是 Mitoma & Furuya, APSIPA 2025：把 HTDemucs 的 vocals/bass/other 三轨幅度加倍后与 drums 重混（单轨消融里最优的是 `other`），triads WCSR 从 **75.52% → 75.72%（+0.20 个百分点）**，485 首真实曲目 ✅。这和"分离能大幅提升和弦识别"的直觉相反。
2. **文献里最稳的单点收益来自"低音信息"这一路，而不是"音质变干净"**：Sumi et al. ISMIR 2008 只用 bass pitch probability 做概率化融合，帧级准确率 **+8.1 点**（150 首 Beatles，真实数据）✅。但那是在 HMM/假设搜索内部融合，不是对既有后验的重加权。
3. **多任务/联合训练（Jointist、Cerberus）确实有效，但都需要训练**；Jointist 把预测出的 piano roll 作为额外通道喂给和弦模型，MajMin **0.798 → 0.812**（Isophonics）✅ —— 这是"用分离/转写结果改善和弦"，不是"用 stems 当音频输入"。
4. **合成数据上的增益不能外推**：同一组实验里，模型在 AAM（3000 首人工合成曲）上的指标普遍高于 Winterreise（真人演奏），后者方差也更大；且"在 AAM+Winterreise 上微调"反而不如"同语料从头训" ✅。
5. 对我们的三个可选动作：**(a) 换输入信号** 与 **(c) 低音根音先验** 可复现，但预期收益按文献量级是 **0.2–1 个百分点**级别，不是"翻倍"；**(b) 双跑后验融合** 没有直接文献，属于自研，必须自带嵌套交叉验证与置换检验。**需要训练、需要改特征、需要改标签集、需要标签当输入** 的路线全部不可复现。

---

## 1. 证据地图

| 路线 | 代表工作 | 证据强度 | 为什么 |
|---|---|---|---|
| HPSS → chroma 前端 | Ueda et al. ICASSP 2010 ✅（主张）/ 数字未核实 | 中 | 结论是"比传统 chroma 好"，但基线是 2010 年的 GMM/HMM，且未报告绝对差值（我拿不到正文） |
| 人声/伴奏分离 → 和弦 | TCSVM, Applied Sciences 2016；Mitoma 2025 的 vocals 消融 | 弱 | 前者只说"与 MIREX 2013/2014 参赛系统相当"（未核实具体数）；后者显示**增强人声反而降 0.77 点** ✅ |
| 低音 → 根音 | Sumi et al. ISMIR 2008 ✅ | 强（但系统老旧） | +8.1 点、150 首真实曲目、5 折交叉验证，数字可核 |
| 多任务联合 | Cerberus ICASSP 2020；Jointist 2022 ✅ | 中强 | 有效，但都以"可训练"为前提 |
| 现代分离器 stems → 冻结和弦模型 | Mitoma & Furuya APSIPA 2025 ✅ | 强（唯一直接证据） | 真实数据、485 首、有逐帧迁移表、有负例分析 |
| 合成语料训练 | Majchrzak & Mądziuk 2025 ✅ | 中 | 结论是"有限场景可用"，不是"可替代真实数据" |
| 标签/分数条件化分离 | Ewert et al. 2014 综述；Tunturi et al. 2025 ✅ | 强（作为方法） | 但推理时需要标签 → 对我们形成闭环依赖 |

---

## 2. 表 A：正向 —— 分离/分轨喂给和弦侧

| 论文（题目 / 作者·年） | 发表 | 喂给和弦侧的是什么 | 报告的增益（原文口径） | 数据集 | 真实/合成 | 证据 |
|---|---|---|---|---|---|---|
| *HMM-based approach for automatic chord detection using refined acoustic features*<br>[Ueda, Uchiyama, Nishimoto, Ono, Sagayama](https://doi.org/10.1109/icassp.2010.5495218) 2010 | ICASSP 2010 | HPSS 抑制打击乐后的**傅里叶变换 chroma + delta-chroma** → HMM | "HMM with the new features yields higher recognition rates (the best in MIREX 2008 audio chord detection task) than that with conventional features"；具体差值 **未核实** | MIREX 2008 audio chord detection 任务集 | 真实 | 主张 ✅ / 数字未核实 |
| *Harmonic/Percussive Separation Using Median Filtering*<br>[Fitzgerald](https://arrow.dit.ie/argcon/67) 2010 | DAFx 2010 | 中值滤波 HPSS 算法本身（被上面那类工作当前端用） | 不涉及和弦增益；只报告分离/重混听感 | 商业录音示例 | 真实 | ✅ |
| *Automatic Chord Recognition Based on Probabilistic Integration of Chord Transition and Bass Pitch Estimation*<br>[Sumi, Itoyama, Yoshii, Komatani, Ogata, Okuno](https://zenodo.org/record/1417221) 2008 | ISMIR 2008 | **bass pitch probability（BPP，PreFEst 估计）** 作为假设可靠性的一项 → 与 chroma GMM、和弦转移 N-gram 加权融合 | 帧级准确率 **73.4%**（150 首）；仅声学特征 → +BPP **+8.1 点**；再 +转移概率 **+11.8 点** ✅ | 12 张 Beatles 专辑中的 150 首、每首 1 分钟片段、Harte 标注、5 折交叉验证 | 真实 | ✅ |
| *Approximate Note Transcription for the Improved Identification of Difficult Chords*<br>[Mauch & Dixon](https://zenodo.org/record/1416597) 2010 | ISMIR 2010 | 先做**近似音符转写**再据此改造 chroma（让前端适配和弦模型，而不是反过来） | 声称能改善"难和弦"（如 sus、七和弦）的识别；绝对数字 **未核实** | MIREX 2009 chord detection 相关数据 | 真实 | 主张 ✅ / 数字未核实 |
| *Large-vocabulary Chord Transcription via Chord Structure Decomposition*<br>[Cho & Su](https://zenodo.org/record/3527891) 2019 | ISMIR 2019 | 把和弦标签**分解成 triad / bass / seventh 三个子任务**（即把"低音"变成独立输出头） | 大词表下优于单头基线；具体 WCSR **未核实** | 多数据集（Isophonics/Billboard/RWC 系） | 真实 | 主张 ✅ / 数字未核实 |
| *Chord Recognition Based on Temporal Correlation Support Vector Machine*<br>[Rao, Guan, Teng](https://doi.org/10.3390/app6050157) 2016 | Applied Sciences 2016 | **RPCA 分离人声**（低秩=伴奏，稀疏=人声），在低秩部分上算 LPCP chroma | "results ... comparable to the state-of-the-art methods that entered the MIREX in 2013 and 2014"；数字 **未核实**（MDPI 站点拒绝抓取） | MIREX'09 Audio Chord Estimation | 真实 | 主张 ✅ / 数字未核实 |
| *Simultaneous Separation and Transcription of Mixtures with Multiple Polyphonic and Percussive Instruments*（Cerberus）<br>[Manilow 等](https://doi.org/10.1109/icassp40776.2020.9054340) 2020 | ICASSP 2020 | 分离与多乐器转写**共享表示、联合训练**（三头：分离/音高/起音） | 联合训练在分离与转写两侧都优于单任务，且对未见混音泛化更好；具体数值 **未核实** | 多乐器混音（具体语料 **未核实**） | 未核实 | 主张 ✅ |
| *Jointist: Joint Learning for Multi-instrument Transcription and Its Applications*<br>[Cheuk 等](https://arxiv.org/abs/2206.10805) 2022 | ISMIR 2022 | 8 通道输入 = 音频谐波谱图 + **模型自己预测出的 piano roll**（不是 stems 音频） | 和弦 MajMin：Isophonics **0.798 → 0.812**；RWC-POP **0.766 → 0.785** ✅ | Billboard / Isophonics / RWC-POP，留一交叉（评测 Isophonics 或 RWC-POP） | 真实 | ✅ |
| *Accuracy Improvement of Automatic Chord Recognition with Source Separation Preprocessing*<br>[Mitoma & Furuya](https://doi.org/10.1109/apsipaasc65261.2025.11249321) 2025 | APSIPA 2025 | HTDemucs 四 stems，**把 vocals/bass/other 的幅度各自加倍后与 drums 重混**，再喂预训练 BTC | 大词表 WCSR：triads **75.52 → 75.72**；root 82.51 → 82.72；maj-min 81.59 → 81.84；tetrads 67.44 → 67.59 ✅ | 485 首：Isophonics 225 + Robbie Williams 65 + uspop2002 195 | 真实 | ✅ |
| *Improvement of Musical Chord Recognition Accuracy Using Source Separation and Remix Models*<br>[Yoshida, Mitoma, Furuya](https://doi.org/10.1007/978-3-032-30573-2_24) 2026 | 2026（Springer LNCS 系） | 同上思路 + "remix models"（用模型学重混比例而非手工加倍） | 摘要页被反爬拦截，**完全未核实** | 未核实 | 未核实 | ❌ |
| *Training chord recognition models on artificially generated audio*<br>[Majchrzak & Mądziuk](https://arxiv.org/abs/2508.05878) 2025 | arXiv 2025 | 训练语料换成 **AAM（3000 首人工合成曲）**，用 BTC 与 Harmony Transformer 做跨语料实验 | 定性：AAM 可"丰富小规模真人语料"甚至单独当训练集；具体 Root/MajMin/CCM 数字 **未核实** | AAM 3000 / Winterreise 48 / Billboard（890 首中 739 首用于 HT 训练） | **合成 + 真实**对比 | 主张 ✅ / 数字未核实 |

> 注：Mitoma & Furuya 的 PDF 直链是 <http://www.apsipa.org/proceedings/2025/papers/APSIPA2025_P307.pdf>，我已下载并用 `pdftotext` 抽取全文核对（下同）；DOI 链接为 2025 年收录版。

---

## 3. 表 B：反向 —— 标签/分数/乐器条件驱动分离

| 论文（题目 / 作者·年） | 发表 | 条件是什么 | 主张 | 数据 | 证据 |
|---|---|---|---|---|---|
| *Score-Informed Source Separation for Musical Audio Recordings: An Overview*<br>[Ewert, Pardo, Müller, Plumbley](https://doi.org/10.1109/msp.2013.2296076) 2014 | IEEE SPM 2014 | 乐谱（音高 + 时值 + 乐器） | 乐谱信息"guides the separation process, leading to significant improvements in terms of separation quality and robustness"（综述级结论，无单一数字） | 综述 | ✅ |
| *Score-Informed Source Separation for Music Signals*<br>[Ewert & Müller](https://doi.org/10.4230/dfu.vol3.11041.73) 2012 | Dagstuhl 2012 | 乐谱/MIDI | 把分数约束塞进 NMF 框架；给出 score-informed NMF 的通用做法 | 综述 + 示例 | ✅ |
| *Transcription Is All You Need: Learning to Separate Musical Mixtures with Score as Supervision*<br>[arXiv 2020](https://arxiv.org/abs/2010.11904) | arXiv 2020 | 用**转写结果（音符）当弱监督**去训分离 | 无需分离标签即可学分离 | 混合 | ✅（题名+摘要） |
| *Bespoke Neural Networks for Score-Informed Source Separation*<br>[arXiv 2020](https://arxiv.org/abs/2009.13729) | arXiv 2020 | 乐谱 | 为 score-informed 定制网络结构 | 混合 | ✅（题名+摘要） |
| *End-to-End Sound Source Separation Conditioned On Instrument Labels*<br>[arXiv 2018](https://arxiv.org/abs/1811.01850) | arXiv 2018 | **乐器标签**（比和弦更粗的条件） | 条件化端到端分离可行 | 混合 | ✅（题名+摘要） |
| *Score-informed Music Source Separation: Improving Synthetic-to-real Generalization in Classical Music*<br>[Tunturi, Diaz-Guerra, Politis, Virtanen](https://arxiv.org/abs/2503.07352) 2025 | arXiv 2025 | 乐谱（拼接进幅度谱 / 只用谱算掩码） | 合成数据（SynthSOD）训练、真实数据（URMP、Aalto anechoic orchestra）评测：score-informed 模型**优于基线但从合成到真实泛化差**；**score-only 掩码的合成→真实泛化明显更好** ✅ | SynthSOD（合成）→ URMP/Aalto（真实） | ✅ |
| *SCISSOR: Score-Conditioned Instrument Source Separation for Orchestral Recordings*<br>[arXiv 2026](https://arxiv.org/abs/2609.33265) | arXiv 2026 | 乐谱条件 | 面向管弦乐录音的分数条件乐器分离 | 管弦乐 | ✅（题名+摘要） |

**对我们的含义（详见 §6）**：这些工作把"标签"当成**推理时输入**。我们若把第一次和弦结果当条件再跑分离，就形成闭环；闭环的收益无法与"自证偏差"分离，除非有独立标注做校验。

---

## 4. 表 C：分离器与语料（判断泄漏与成本用）

| 对象 | 发表 | 关键事实 | 对我们的意义 | 证据 |
|---|---|---|---|---|
| [HTDemucs](https://arxiv.org/abs/2211.08553) | ICASSP 2023 | 时域+频域双分支 + 跨域 Transformer；**训练用 MUSDB + 内部 800 首**；输出 4 stems（vocals/bass/drums/other） | 内部 800 首身份不公开 → 无法排除与评测集重叠 | ✅（摘要明写 800 首内部数据） |
| [BS-RoFormer](https://arxiv.org/abs/2309.02612) | arXiv 2023（SDX'23 第一） | MUSDB18HQ（100 训/50 测）+ **in-house 500 首**（450 训/50 验）；仅 MUSDB18HQ 时平均 SDR **9.80 dB**；**`other` 轨由"混音减去 vocals/bass/drums"得到**；4 秒 hop、重叠段平均 | ① 内部 500 首同样不公开；② `other` 相减 ⇒ **stems 求和 ≈ 原混音**，做"改权重再重混"时不会引入额外残差 | ✅ |
| [Mel-RoFormer](https://arxiv.org/abs/2409.04702) | arXiv 2024 | **两步法**：先训人声分离，再把分离模型当 foundation model 微调人声旋律转写 | "分离器当上游预训练"是有效的，但需要训练 | ✅（摘要） |
| [MUSDB18](https://doi.org/10.5281/zenodo.1117372) | 2017 | 150 首（100/50），4 stems | 分离界的"MNIST"，规模小；无和弦标注 | ✅（BS-RoFormer 正文引用其规模） |
| [MoisesDB](https://arxiv.org/abs/2307.15913) | 2023 | 240 首、45 位艺人、12 类风格、**层级化 stems（不止 4 轨）** | 想做"bass 单独一轨 + 和弦"实验时，比 MUSDB 更合适 | ✅ |
| [AAM（Artificial Audio Multitracks）](https://arxiv.org/abs/2508.05878) | 数据集 | 3000 首算法生成曲，和弦分布比真实语料均匀得多 | 只适合做"合成→真实"的对照，不适合当唯一证据 | ✅ |

---

## 5. 逐条精读

### 5.1 HPSS 作为 chroma 前端（2010 前后）

- **主张**：流行音乐里打击乐会污染 chroma；先用 HPSS 压掉打击成分、强化谐波成分，再算 chroma + delta-chroma，HMM 识别率高于传统特征；该提交是 MIREX 2008 音频和弦检测任务的最好成绩（[Ueda et al. 2010](https://doi.org/10.1109/icassp.2010.5495218)）。
- **指标**：当时 MIREX 用帧级/时长加权的 chord accuracy（root/majmin 系）；论文摘要只给"更高"，**没有给差值**（未核实）。
- **数据集**：MIREX 2008 官方集（真实录音）。
- **判断**：这是"改前端特征"，对我们**不可直接复现**（我们的模型吃音频、不吃 chroma）。但可以近似成 (a)：**只把 HPSS 的谐波分量当输入音频**，这是一个 5 行代码就能做的 ablation，代价是丢掉瞬态 → 边界可能变糊。
- **HPSS 的算法基础**：[Fitzgerald 2010](https://arrow.dit.ie/argcon/67) 的谱图中值滤波（时间方向滤掉打击、频率方向滤掉谐波）。工程上 `librosa.effects.hpss` 就是这一族。

### 5.2 低音信息（最有历史分量的一条线）

- **Sumi et al. 2008**（[Zenodo](https://zenodo.org/record/1417221)）：
  - 做法：PreFEst 估计 **bass pitch probability（BPP）**，把"和弦-低音音的联合概率"作为假设可靠性三项之一（另两项：chroma GMM 似然、和弦转移 N-gram），在假设搜索里剪枝。
  - 指标：**帧级准确率（frame-rate accuracy）**，48 类和弦（maj/min/dim/sus4 × 12 根音）。
  - 数据：12 张 Beatles 专辑 150 首、每首 1 分钟、Harte 标注、5 折交叉验证（真实数据）。
  - 结果：平均 **73.4%**；"仅声学特征 → +BPP" **+8.1 点**；"声学 + BPP + 转移" **+11.8 点**。✅
  - 注意：这是**在模型内部做概率融合**，不是对已有后验做事后重加权；而且 BPP 是"概率分布"而非硬判根音（论文明确说非低音帧里 BPP 接近均匀，因此不会乱改）。这两点正是我们做 (c) 时该抄的：**软先验 + 冲突门控**，不要硬覆盖。
- **Cho & Su 2019**（[Zenodo](https://zenodo.org/record/3527891)）：把和弦标签拆成 triad / bass / seventh 三个子任务分别建模，说明"低音"在和弦标注体系里本来就是**独立可预测的一维**。这为我们"只改根音、不改性质"的先验提供了合法性，但也提醒：**如果模型的标签体系不含转位/斜杠和弦，低音先验只能改根音，不能表达 /E 这类信息**。
- **Mauch & Dixon 2010**（[Zenodo](https://zenodo.org/record/1416597)）：反向思路——先近似转写音符（含低音），再据此设计 chroma。属于"特征工程时代"的最优解之一，与我们的约束不兼容。

### 5.3 人声/伴奏分离

- **TCSVM 2016**（[MDPI](https://doi.org/10.3390/app6050157)）：RPCA 把谱图分解为低秩（伴奏）+ 稀疏（人声），在低秩部分算 LPCP；结论是"与 MIREX 2013/2014 参赛系统相当"。**数字未核实**（站点 403）。
- **反向证据更有价值**：Mitoma 2025 的逐 stem 消融里，**单独把人声加倍让 triads 从 75.52 掉到 74.75（-0.77）** ✅。这说明在他们的系统上，人声是**干扰**而非信息；那么"去人声"理论上应有小幅正收益——但注意，这是从"增强有害"推到"削弱有益"，**不是同一件事**（增益曲线可能非单调）。我们若做"instrumental only"，必须把"去人声+保留鼓"和"去人声+去鼓"分开测。

### 5.4 多任务 / 联合分离 + 和弦

- **Cerberus**（[ICASSP 2020](https://doi.org/10.1109/icassp40776.2020.9054340)）：在 Chimera 分离网络上加"转写头"，三头联合训练；作者称分离与转写"highly complementary"，联合训练两边都更好、对未见混音泛化更好。**这直接支持"分离与和弦/转写共享信息"，但前提是你能训练。**
- **Jointist**（[ISMIR 2022](https://arxiv.org/abs/2206.10805)）：更接近我们的场景——它把"乐器级 piano roll"当成额外输入通道，与音频谐波谱图拼成 8 通道，喂给 SpecTNT 做和弦/调性/小节。
  - 数字（Table 5，评测口径为 **frame-wise Major/Minor**）：Isophonics 0.798 → **0.812**；RWC-POP 0.766 → **0.785** ✅。
  - 关键差别：它喂的是**符号化中间表示**，不是分离后的音频；而且是**重新训练**的 SpecTNT。我们能复现的只是"多一路信息"这个思想，不能复现这条管线。
- **Omnizart**（[JOSS 2021](https://doi.org/10.21105/joss.03391)）：工具箱层面把和弦识别与转写放在一起，可作为"现成基线"参考。

### 5.5 现代分离器 + stems（2021–2025，唯一直接证据）

**Mitoma & Furuya, APSIPA 2025**（[DOI](https://doi.org/10.1109/apsipaasc65261.2025.11249321)，[PDF](http://www.apsipa.org/proceedings/2025/papers/APSIPA2025_P307.pdf)）——我把全文读了，细节如下：

- **基线**：BTC（[jayg996/BTC-ISMIR19](https://github.com/jayg996/BTC-ISMIR19) 的公开预训练权重），输入 CQT（22.05 kHz，C1 起 6 个八度，24 bins/octave，hop 2048），大词表 170 类。
- **分离**：HTDemucs 分出 vocals/bass/drums/other；把 **vocals、bass、other 都当"有音高乐器"**，各自幅度 **×2**，再与 drums 重混。
- **指标**：`mir_eval` 的 **WCSR（加权和弦符号召回，逐帧、按时长加权）**，报 triads/root/maj-min/tetrads 四项。
- **数据**：485 首真实曲目（Isophonics 225 + Robbie Williams 65 + uspop2002 195）。
- **结果**（Table I / II，%）：

  | 条件 | triads | root | maj-min | tetrads |
  |---|---|---|---|---|
  | 原始混音 | 75.52 | 82.51 | 81.59 | 67.44 |
  | 只加倍 vocals | 74.75 | 81.81 | 80.82 | 66.48 |
  | 只加倍 bass | 75.06 | 82.21 | 81.09 | 66.83 |
  | 只加倍 other | 75.67 | 82.66 | 81.78 | 67.45 |
  | **三者都加倍（提出方法）** | **75.72** | **82.72** | **81.84** | **67.59** |

- **逐帧迁移**（Table III，triads）：共 1,136,742 帧；"原对→新错" **11,390 帧**，"原错→新对" **13,675 帧**，净减 **2,285 帧**错误；作者做了符号检验，称显著。
- **负例（很重要）**：`Doctor Robert` 里 A:maj 段中吉他 riff 弹了一个 D 音，加倍 guitar 后 D 音更突出，模型把 A:maj **切成 A:sus4**。作者的结论是"增强和弦音多数时候有用，但也放大了孤立音"。
- **我的判断**：
  1. 这是**真实数据 + 现代分离器 + 预训练模型**，是当前最可信的一条证据，增益量级 **+0.2 点**。
  2. 单独加倍任一 stem 都**不如**原始混音（除 other 略高 0.15），三者一起才涨——说明效果主要来自**相对电平重排**，而不是"分离得更干净"。这直接指向我们的坑：**电平/响度控制没做好，就会把"增益"和"电平"混为一谈**。
  3. 他们把 drums 留着是刻意的（"完全去掉无音高乐器会丢失对曲式重要的节奏信息"）——我们现在的 H 是 `bass+other+guitar+piano`，**没有鼓**，这一点与文献的最优配置不同。

**Yoshida, Mitoma, Furuya 2026**（[Springer](https://doi.org/10.1007/978-3-032-30573-2_24)）：同一课题组的后续，标题说用"分离 + remix models"提升和弦准确率。摘要页被反爬（Crossref 也无摘要），**全部未核实**。如果后续要引用"用模型学重混比例"的说法，必须先去拿到正文。

### 5.6 合成数据能替代真实数据吗

- **Majchrzak & Mądziuk 2025**（[arXiv](https://arxiv.org/abs/2508.05878)）用 AAM（3000 首合成）对照 Winterreise（舒伯特，真人演奏）与 Billboard（真实流行）：
  - 指标分两层：**单和弦级**（Root、MajMin、Mirex、CCM）与**序列级**（WCSR/WAOR、加权准确率）。
  - 观察 1：在 AAM 上，**没在 AAM 上训过的预训练 BTC 反而比只训 Winterreise 的好** → 合成数据的"复杂度"被流行语料覆盖了一部分。
  - 观察 2：Winterreise 上整体更低、方差更大 → **真实数据更难**。
  - 观察 3：BTC 在 AAM+Winterreise 上**微调反而不如从零训**。
  - 具体数值都在图里，我没抄到表格 → **未核实**，但定性结论清楚：**合成语料可以补充，不能作为唯一证据**。
- **对我们的提醒**：任何"在合成 stems / 合成混音上验证出的分离增益"都要在真实曲目上重测，否则结论不可迁移。

---

## 6. 反向：标签条件化分离，以及它对我们管线的含义

### 6.1 文献说了什么

- **综述层面**：乐谱/标签提供音高、时值、乐器信息，"guides the separation process, leading to significant improvements in separation quality and robustness"（[Ewert et al., IEEE SPM 2014](https://doi.org/10.1109/msp.2013.2296076)）。这是这一支的权威入口。
- **弱监督版**：把转写结果当监督去训分离（[Transcription Is All You Need, 2020](https://arxiv.org/abs/2010.11904)）；为 score-informed 定制网络（[2020](https://arxiv.org/abs/2009.13729)）；用乐器标签做条件（[2018](https://arxiv.org/abs/1811.01850)）。
- **合成→真实的关键警告**：[Tunturi et al. 2025](https://arxiv.org/abs/2503.07352) 在 SynthSOD（合成）上训练、在 URMP 与 Aalto anechoic orchestra（真实）上评测，结论是 **score-informed 模型确实优于基线，但从合成到真实的泛化很差；只用分数算掩码的变体泛化反而更好** ✅。这条对我们是双重警告：score-informed 方法本身有"合成→真实"鸿沟，而**我们所有离线和在线实验的语料都是真实商业录音**，与论文的训练分布不同。
- **条件化的边界**：条件越强（完整乐谱）→ 分离越好；条件越弱（只有和弦标签）→ 信息量不足以约束音高分布，收益会显著缩水。**和弦标签只能告诉你"哪些音在响"，不告诉你"哪个乐器在响、什么时候起音"**，所以"用和弦条件化分离"在文献里几乎不是主流做法——主流是"用音符/乐谱条件化"。

### 6.2 对我们管线的含义

1. **开环可行、闭环危险**。我们只有冻结分离器 → 无法把标签喂进去 → 只能做开环（先分离、后识别）。若要做闭环（识别→分离→再识别），等价于用模型自己的输出当先验，**确认偏差**会把错误固化；而且一旦用评测集去挑"迭代几轮/阈值多少"，测试集就被间接用于选参，指标不再干净。
2. **"伪条件化"是可行的折中**：不做真条件化，而是用**预测结果去选择/加权已有 stems**。例如：只在与预测和弦"冲突"的帧上，才把 bass stem 提权；或按预测根音选择低音 chroma 的置信区间。这保留了"标签引导"的收益，又不需要可条件化的分离器。代价是：**一旦预测错，选择也错**，所以必须用"仅在冲突时动作"的门控来限制暴露面（与 Sumi 的软 BPP 思路一致）。
3. **不要期待"标签→分离→标签"能自我提升**：文献里 score-informed 的收益是**分离质量**的（SDR 类），不是下游和弦准确率的；没有任何一篇报告"用和弦标签条件化分离后和弦识别提升 X 点"。这条链条的最后一环**没有文献支撑**（未核实，且我认为不存在）。

---

## 7. 已知坑

### 7.1 分离 artefact 怎么伤 chroma

四条机制，按"是否会在 chroma 上留下系统性偏差"排序：

1. **频带选择性衰减（最毒）**。上采样层不当代来的 artefact 分两类：**tonal（加性高频噪声）** 与 **filtering（减性、衰减某些频带）**（[Upsampling Layers for MSS, EUSIPCO 2023](https://doi.org/10.23919/eusipco58844.2023.10289768)）。tonal 噪声对 chroma 影响有限（不落在音高格点上），**filtering 型会直接吃掉某个八度的能量 → chroma 里该音级系统性偏弱 → 根音/三音判错**。
2. **SDR 会骗人**。ICASSP 2022 的 benchmark 明确指出"the standard signal-to-distortion ratio metric can be misleading in some scenarios"，并用主观测听去校准替代指标（[论文](https://doi.org/10.1109/icassp43922.2022.9746530)）。所以"分离器 SDR 更高"不能推出"和弦更准"。
3. **谐波泄漏 → 虚假和弦音**。分离不干净时，伴奏/人声的谐波会漏进目标 stem；对 chroma 而言，这等价于**凭空多出几个音级**，最容易诱发 sus/add 类和弦。Mitoma 的 `A:maj → A:sus4` 就是这个机制的实证（增强吉他让孤立 D 音突出）✅。
4. **瞬态钝化 → 边界漂移**。时域分离器与 HPSS 都会平滑起音；和弦边界恰恰依赖"和声何时变化"，起音被钝化 → 边界抖动 → 段级指标掉、帧级指标可能反而涨（见 §7.4）。

### 7.2 harmonic stem 与低音基频：两个常被混淆的事实

- **HPSS 不会丢低音**。HPSS 的"谐波分量"按定义是**频率方向稳定、时间方向连续**的成分，贝斯正是典型谐波信号 → 它会留在 harmonic 里，代价是**起音被抹平**。所以"HPSS 的 harmonic stem 丢低音"这个说法**不成立**；成立的是"HPSS 让低音变糊"。
- **4-stem 分离器才会丢低音**。HTDemucs / BS-RoFormer 都把 `bass` 单独拎出来（4 stems：vocals/bass/drums/other）。如果我们只喂 `other`（或只喂"和声类 stem"而把 bass 漏掉），模型看到的是**没有根音的和弦** → 空根、错根、sus/6 和弦误判都会上升。
- **一个重要的实现差异**：BS-RoFormer 的 `other` 是**用混音减去 vocals/bass/drums** 得到的（[正文](https://arxiv.org/abs/2309.02612)）→ 四轨相加 ≈ 原混音，做"改权重再重混"时不会有残差；HTDemucs 没有这个保证。**如果我们的操作是"改权重再重混"，优先用求和一致的分离器**，否则重混信号本身就带有一次"再合成误差"。

### 7.3 评测泄漏

三种泄漏，按隐蔽程度排序：

1. **分离器训练集 ↔ 和弦评测集重叠**。HTDemucs 用 **MUSDB + 内部 800 首**（[摘要](https://arxiv.org/abs/2211.08553)）；BS-RoFormer 用 **MUSDB18HQ + in-house 500 首**（[正文](https://arxiv.org/abs/2309.02612)）。内部曲目身份**不公开**，而和弦评测常用的曲目（Isophonics / Robbie Williams / uspop2002 / Billboard 等）也是商业录音 → **无法排除重叠**。这不是"一定泄漏"，而是"无法证伪"，写论文/写结论时必须声明。
2. **同一批歌既训分离器又评和弦**。如果为了"领域适配"在评测集上微调分离器，就等于把测试集看过一遍。**任何在评测曲目上做过的分离器微调，都必须在该曲目上排除**。
3. **合成语料的"伪泛化"**。在 AAM/Slakh 这类合成混音上，分离器近乎完美（干声直接相加），下游增益会虚高；真实的混响、母带处理、声码器 artefact 都不在合成分布里（参见 §5.6 与 [Tunturi 2025](https://arxiv.org/abs/2503.07352) 的合成→真实鸿沟）。

**可操作的护栏**：① 记录并公开所用分离器的 checkpoint 与训练语料；② 至少在两种分离器（如 BS-RoFormer 与 HTDemucs）上复现同一结论；③ 把"分离器见过的数据"当成不可用评测集，宁可用小一点但干净的集合。

### 7.4 frame-level 与 segment-level 讲的是两个故事

- **定义差异**：帧级/时长加权指标（WCSR、加权准确率）算的是"**有多少时间占比被判对**"；序列级指标关心"**段落/边界是否对**"。前者的高分可以来自"大段对、边界烂"，后者的高分要求边界与标签同时正确。
- **`mir_eval` 的实际口径**：`mir_eval.chord` 提供的是 `root / majmin / thirds / triads / tetrads / mirex` 等**按时长加权的帧级**函数（[源码](https://github.com/craffel/mir_eval/blob/main/mir_eval/chord.py)）；它**不包含** over/under-segmentation 这类边界指标。也就是说：**Mitoma 2025 报的 75.52→75.72 全是帧级**，没有报边界。
- **同一模型可以"帧级涨、边界烂"**：Mitoma 的 Table III 是净赚 2,285 帧，但正文的 `A:maj → A:sus4` 例子说明它同时在**制造切分**。如果下游要显示"和弦轨道"，用户感知到的往往是**切得太碎**，不是"时间占比高了 0.2%"。
- **指标分类的权威说法**：Majchrzak & Mądziuk 把指标明确分成"single chord level"（Root/MajMin/Mirex/CCM）与"chord sequence level"（WCSR/WAOR = 预测正确段落的总时长；加权准确率 = 按时长加权的平均）（[arXiv 2508.05878 §2](https://arxiv.org/abs/2508.05878)）。
- **别忘了分割本身是个任务**：Harmony Transformer 把"和弦分割"和"和弦识别"做成多任务，理由正是"边界在和声状态之间没有显式定义"（[ISMIR 2019](https://doi.org/10.5281/zenodo.3527793)）。
- **评测协议的历史包袱**：Pauwels 等的二十年综述指出，ACR 的核心争议（词表、粒度、标注主观性）从 1999 年就存在（[ISMIR 2019](https://doi.org/10.5281/zenodo.3527738)）——所以跨论文比数字前先对齐词表与加权方式，否则"提升 2 个点"可能只是词表不同。

### 7.5 工程坑（含我们自己的补充，非文献结论）

- **电平**：Mitoma 是"手动把幅度加倍"，总响度也变了。若我们的融合实验里 mix 与 H 的 RMS/LUFS 不同，**任何差异都可能只是电平差异**。必须做响度对齐后再比较（并单独做一个"只改电平、不改内容"的对照）。
- **重采样**：分离器输出 44.1 kHz，前端可能吃 22.05 kHz（BTC 就是）。两次重采样 + 分离器内部的 STFT/ISTFT 会叠加 artefact。
- **chunk 边界**：BS-RoFormer 用 4 秒 hop、重叠段平均（[正文](https://arxiv.org/abs/2309.02612)）；长曲要分块，块边界处可能出现能量台阶 → chroma 的假边界。
- **相位/立体声**：时域分离器输出的相位不保证自洽；chroma 只用幅度，影响小，但重混时若左右声道相位不一致会产生梳状滤波 → 频带凹陷（回到 §7.1 的第 1 条）。
- **统计功效**：一首 4 分钟歌、约 1000 个分析帧，**1 帧 = 0.1%**；单曲上的 0.5 点差异 ≈ 5 帧，完全可能是解码抖动。**段级更糟**：一首歌可能只有 20–60 个参考段，1 段 = 2–5 点。
- **成本**：每首歌多一次分离（GPU 上数秒到数十秒）+ 多一次前端推理；若做网格搜索，成本按"曲目数 × 配置数"线性膨胀。

---

## 8. 对我们的可复现性

### 8.1 约束重述

我们的和弦模型是**冻结的、音频输入**的模型，输出为（根音 × 性质）的帧级分布（当前原型是 12×10=120 类 + 一个 N 概率）。因此只有三条动手路径：

- **(a) 换输入信号**：改喂什么音频（重混、去轨、HPSS、重采样、增益）。
- **(b) 双跑后验融合**：跑两次（不同输入），在**后验层**做加权/几何平均。
- **(c) 低音根音先验**：用 bass 轨的音高/ chroma 构造根音先验，重加权后验。

### 8.2 可复现性总表

| 技术 | 路径 | 可复现？ | 依赖 | 文献给的量级 | 主要风险 |
|---|---|---|---|---|---|
| Stem 加权重混（Mitoma） | (a) | ✅ 完全可复现 | 分离器 + 重混 + 电平控制 | **+0.20 pt** triads WCSR（真实，485 首）✅ | 孤立音被放大 → sus/add 误判；电平混淆 |
| 只喂 HPSS 谐波分量（Ueda 思路的音频版） | (a) | ✅ 可复现（librosa 即可） | 中值滤波 HPSS | 定性"优于传统特征"；数字未核实 | 瞬态钝化 → 边界漂移；对神经模型方向未知 |
| Instrumental only（去人声） | (a) | ✅ 可复现 | 分离器 | 无直接正证据；反推自"增强人声 −0.77 pt" ✅ | 增益可能非单调；去人声同时去了呼吸/齿音等线索 |
| 只喂 bass+other+guitar+piano（现 H） | (a) | ✅ 可复现 | 6-stem 分离器 | **无直接文献**；Mitoma 最优配置**保留鼓** | 丢鼓 → 边界线索减少；与文献配置不同，不能直接引用其增益 |
| 低音根音先验（Sumi 思想后置化） | (c) | ⚠️ 机制可复现，**增益不可迁移** | bass 音高/chroma + 冲突门控 | **+8.1 pt**（模型内概率融合，真实 150 首）✅ | 转位/经过音/踏板音 → 错误覆盖；软先验退化成硬覆盖 |
| 双跑后验融合 mix ⊕ H | (b) | ✅ 可复现，但**属自研** | 两套后验 + 权重选择 + 显著性检验 | 无直接文献（Jointist 的 +1.4 pt 是"特征拼接+重训"，不可类比）✅ | 两跑同源 → 误差相关，增益小于单跑最优；λ 过拟合 |
| 分离器微调 / 领域适配 | — | ❌ | 训练 | — | 且会造成 §7.3 的泄漏 |
| 和弦模型微调 / 重训 | — | ❌ 模型冻结 | 训练 | — | — |
| 联合多任务（Cerberus / Jointist） | — | ❌ | 训练 | +1.4 pt（Jointist，需重训）✅ | — |
| 学习型 chroma 前端（[Deep Chroma Extractor](https://arxiv.org/abs/1612.05065)） | — | ❌ | 改模型输入表征 | — | 我们吃音频 |
| 标签/分数条件化分离（score-informed、SCISSOR） | — | ❌ 开环不可用；闭环有自证偏差 | 可条件化的分离器 + 标签 | 分离质量提升（SDR 类），**下游和弦增益无文献** | 确认偏差 |
| 词表扩展 / 低音独立输出头（Cho & Su） | — | ❌ 冻结 120 类 | 重训 | — | — |
| 分离作为**训练期**增广（如 [EUSIPCO 2021 的节拍/小节工作](https://doi.org/10.23919/eusipco54536.2021.9616022)） | — | ❌ | 训练 | 未核实 | — |

**一句话**：能做的只有 (a) 和 (c)，(b) 是自研；预期收益按文献量级是 **0.2–1 点**，且很可能被"边界变碎"抵消。

### 8.3 三个候选的最小实验设计

**E1（输入替换，单因素）**
- 条件：`mix`（基线）/ `H`（现配置）/ `H+drums` / `instrumental`（去人声留鼓）/ `harmonic`（HPSS）/ `vocals-only`（负对照，应显著变差——用来证明实验有分辨力）。
- 控制：所有条件做 **RMS 或 LUFS 对齐**到 mix；同一前端、同一解码、同一随机种子。
- 指标：帧级 WCSR（root / majmin / triads）+ 段级（段内多数投票 + **边界 F1**，两个都报）。
- 统计：**逐曲配对 bootstrap**，报 95% CI；曲目数 ≥ 30（低于此不要下结论）。

**E2（后验融合）**
- λ 网格，但 **λ 必须在训练曲上选、在留出曲上报**（嵌套 CV）；同时报告 λ 在 split-half 上的一致率。
- 负对照：把 H 的后验**随机置换时间轴**后再融合——如果置换版也涨，说明涨的是"多一路平滑"而不是"分离信息"。

**E3（低音先验）**
- β 网格 + **三重门控**：① bass 帧能量高于阈值；② bass 音级**不在**当前预测和弦的音级集合内（冲突才动作）；③ bass chroma 熵低（置信）。
- 负对照：**随机根音先验**（同强度、随机音级）——必须显著差于真 bass 先验，否则说明 β 只是在做"标签平滑"。
- 消融：只对 root 重加权 vs 对全部 120 类重加权（后者会连带改变性质，容易过拟合）。

### 8.4 什么算"赢"（判据）

1. **主指标双报**：帧级 WCSR **与** 边界 F1 必须同时报；只涨帧级、边界跌 → 判定为"分割抖动"，不上线。
2. **效应量门槛**：≥ 0.5 个百分点，且配对 bootstrap p < 0.05；单曲实验（<30 段）不报段级差异。
3. **跨分离器稳健**：同一结论至少在两种分离器（BS-RoFormer / HTDemucs）或一种分离器 + HPSS 上复现；只在一种分离器上成立的增益视为"分离器 artefact 指纹"。
4. **负对照必须跑**：随机置换、随机先验、纯电平对照，三个都过了才算真增益。

### 8.5 与本目录原型的对照（非文献结论，基于 `prototype_fusion.py` / `fusion-result.json` / `split-half.json`）

1. **H 的构成与文献最优配置不同**：`build_harmonic()` 是 `bass+other+guitar+piano`，**不含 drums、不含 vocals**；Mitoma 的最优配置是"保留鼓、只调有音高轨"。因此**不能直接引用他们的 +0.20 pt** 作为我们的预期，必须自己测 `H+drums` 这一档。
2. **电平没对齐**：`build_harmonic()` 只在峰值 > 0.999 时削峰，没有做响度归一化 → mix 与 H 的响度差会混进融合结果。建议先补一个 RMS/LUFS 对齐版本再扫 λ。
3. **λ 不可信**：`fusion-result.json` 报 λ=0.7、β=0（985 格、单曲、238 秒）；但 `split-half.json` 里两个半集的最优 λ 分别是 **0.0** 和 **0.6** —— 说明在当前样本量下 λ 的估计极不稳定，**单点最优不能当结论**。
4. **"segment" 的口径要注意**：`segment_accuracy()` 是"参考段内多数投票是否等于该段标签"，**不是边界指标**，也不惩罚过度分割；它与文献里的 segmentation / over-under-segmentation 不同名不同义。若要和 Mitoma 的 WCSR 对齐，主指标应该用时长加权；若要谈"切得碎不碎"，得另算边界 F1。
5. **产物没留基线**：`fusion-result.json` 只存最优点，不存 `mix` 与 `harmonic` 的单跑数字 → 现在**无法从产物反推是否有增益**。建议把全网格 + 基线一起落盘，否则每次都要重跑。

### 8.6 建议顺序（成本从低到高）

1. **补齐度量与对照**（半天）：响度对齐、全网格落盘、加边界 F1、加随机置换负对照。——没有这些，后面所有结论都不可信。
2. **E1 输入替换**（1–2 天）：6 个条件 × 30 首，先看方向；重点是 `H` vs `H+drums` 和 `instrumental`。
3. **E3 低音先验**（1–2 天）：在 E1 的最优输入上做，带三重门控与随机先验对照。
4. **E2 融合**（2–3 天）：只有在 E1/E3 都出现稳定信号后才做；用嵌套 CV 选 λ，否则一定过拟合。
5. **止损条件**：若 E1 在所有输入变体上都 < 0.5 pt 且边界 F1 下降，则"换输入"这条路判定为无效，把预算转向 (c) 与解码/后处理（如和弦时长先验、Viterbi 平滑），不要继续堆分离器。

---

## 9. 未核实清单（明确列出，避免被当成事实引用）

| 项 | 状态 | 原因 | 怎么核 |
|---|---|---|---|
| Ueda et al. 2010 的具体识别率与相对提升 | 未核实 | IEEE 付费墙，无 OA 副本 | 找作者主页/ResearchGate，或 MIREX 2008 结果页 |
| TCSVM 2016 的准确率数字 | 未核实 | MDPI 对抓取返回 403 | 浏览器直接下载 PDF |
| Yoshida et al. 2026 的全部内容 | 未核实 | Springer 反爬，Crossref 无摘要 | 机构订阅或作者预印本 |
| Mauch & Dixon 2010 的具体增益 | 未核实 | 只读到摘要 | Zenodo 正文 PDF |
| Cho & Su 2019 的 WCSR 数字 | 未核实 | 只读到摘要 | Zenodo 正文 PDF |
| Cerberus 的具体 F1/SDR | 未核实 | 只读到摘要 | arXiv/ICASSP 正文 |
| Majchrzak & Mądziuk 2025 的表格数值 | 未核实 | 结果只在图里，未逐点抄录 | 读原文图 3/图 4 |
| Tunturi et al. 2025 的 SDR 数值 | 未核实 | 只读摘要 | arXiv 正文表格 |
| 分离器内部语料与我们的评测集是否重叠 | **不可核实** | 内部数据不公开 | 只能声明为限制，或用"训练集已公开"的分离器做洁净实验 |
| Jointist 的 +1.4 pt 是否可迁移到"stems 音频"场景 | 不适用 | 它的输入是符号 piano roll，不是音频 | 不能类比，只能作为"多一路信息有用"的旁证 |

---

## 10. 参考链接（全部）

**正向（分离 → 和弦）**

- Ueda, Uchiyama, Nishimoto, Ono, Sagayama. *HMM-based approach for automatic chord detection using refined acoustic features.* ICASSP 2010. <https://doi.org/10.1109/icassp.2010.5495218>
- Fitzgerald. *Harmonic/Percussive Separation Using Median Filtering.* DAFx 2010. <https://arrow.dit.ie/argcon/67>
- Sumi, Itoyama, Yoshii, Komatani, Ogata, Okuno. *Automatic Chord Recognition Based on Probabilistic Integration of Chord Transition and Bass Pitch Estimation.* ISMIR 2008. <https://zenodo.org/record/1417221>
- Mauch & Dixon. *Approximate Note Transcription for the Improved Identification of Difficult Chords.* ISMIR 2010. <https://zenodo.org/record/1416597>
- Cho & Su. *Large-vocabulary Chord Transcription via Chord Structure Decomposition.* ISMIR 2019. <https://zenodo.org/record/3527891>
- Rao, Guan, Teng. *Chord Recognition Based on Temporal Correlation Support Vector Machine.* Applied Sciences 2016. <https://doi.org/10.3390/app6050157>
- Manilow 等. *Simultaneous Separation and Transcription of Mixtures with Multiple Polyphonic and Percussive Instruments (Cerberus).* ICASSP 2020. <https://doi.org/10.1109/icassp40776.2020.9054340>
- Cheuk 等. *Jointist: Joint Learning for Multi-instrument Transcription and Its Applications.* ISMIR 2022. <https://arxiv.org/abs/2206.10805>
- Mitoma & Furuya. *Accuracy Improvement of Automatic Chord Recognition with Source Separation Preprocessing.* APSIPA ASC 2025. <https://doi.org/10.1109/apsipaasc65261.2025.11249321> ｜ PDF: <http://www.apsipa.org/proceedings/2025/papers/APSIPA2025_P307.pdf>
- Yoshida, Mitoma, Furuya. *Improvement of Musical Chord Recognition Accuracy Using Source Separation and Remix Models.* 2026. <https://doi.org/10.1007/978-3-032-30573-2_24>
- Majchrzak & Mądziuk. *Training chord recognition models on artificially generated audio.* arXiv 2025. <https://arxiv.org/abs/2508.05878>
- Wu 等. *Omnizart: A General Toolbox for Automatic Music Transcription.* JOSS 2021. <https://doi.org/10.21105/joss.03391>

**反向（标签/分数 → 分离）**

- Ewert, Pardo, Müller, Plumbley. *Score-Informed Source Separation for Musical Audio Recordings: An Overview.* IEEE Signal Processing Magazine 2014. <https://doi.org/10.1109/msp.2013.2296076>
- Ewert & Müller. *Score-Informed Source Separation for Music Signals.* Dagstuhl 2012. <https://doi.org/10.4230/dfu.vol3.11041.73>
- *Transcription Is All You Need: Learning to Separate Musical Mixtures with Score as Supervision.* arXiv 2020. <https://arxiv.org/abs/2010.11904>
- *Bespoke Neural Networks for Score-Informed Source Separation.* arXiv 2020. <https://arxiv.org/abs/2009.13729>
- *End-to-End Sound Source Separation Conditioned On Instrument Labels.* arXiv 2018. <https://arxiv.org/abs/1811.01850>
- Tunturi, Diaz-Guerra, Politis, Virtanen. *Score-informed Music Source Separation: Improving Synthetic-to-real Generalization in Classical Music.* arXiv 2025. <https://arxiv.org/abs/2503.07352>
- *SCISSOR: Score-Conditioned Instrument Source Separation for Orchestral Recordings.* arXiv 2026. <https://arxiv.org/abs/2609.33265>

**分离器与数据**

- Rouard, Massa, Défossez. *Hybrid Transformers for Music Source Separation (HT Demucs).* ICASSP 2023. <https://arxiv.org/abs/2211.08553>
- Lu, Wang, Kong, Hung. *Music Source Separation with Band-Split RoPE Transformer (BS-RoFormer).* arXiv 2023. <https://arxiv.org/abs/2309.02612>
- Wang, Lu, Chen. *Mel-RoFormer for Vocal Separation and Vocal Melody Transcription.* arXiv 2024. <https://arxiv.org/abs/2409.04702>
- Rafii 等. *The MUSDB18 corpus for music separation.* 2017. <https://doi.org/10.5281/zenodo.1117372>
- Pereira 等. *MoisesDB: A dataset for source separation beyond 4-stems.* arXiv 2023. <https://arxiv.org/abs/2307.15913>
- Korzeniowski & Widmer. *Feature Learning for Chord Recognition: The Deep Chroma Extractor.* arXiv 2016. <https://arxiv.org/abs/1612.05065>

**评测与坑**

- *On Loss Functions and Evaluation Metrics for Music Source Separation.* ICASSP 2022. <https://doi.org/10.1109/icassp43922.2022.9746530>
- *Upsampling Layers for Music Source Separation.* EUSIPCO 2023. <https://doi.org/10.23919/eusipco58844.2023.10289768>
- mir_eval 的 chord 模块（帧级/时长加权指标的函数清单）. <https://github.com/craffel/mir_eval/blob/main/mir_eval/chord.py>
- Chen & Su. *Harmony Transformer: Incorporating Chord Segmentation into Harmony Recognition.* ISMIR 2019. <https://doi.org/10.5281/zenodo.3527793>
- Pauwels, O'Hanlon, Gómez, Sandler. *20 Years of Automatic Chord Recognition from Audio.* ISMIR 2019. <https://doi.org/10.5281/zenodo.3527738>
- Humphrey & Bello. *Four Timely Insights on Automatic Chord Estimation.* ISMIR 2015. <https://doi.org/10.5281/zenodo.1417548>（延伸阅读；双栏 PDF 抽取质量差，本文未逐条核对其中观点）
- *Source Separation-based Data Augmentation for Improved Joint Beat and Downbeat Tracking.* EUSIPCO 2021. <https://doi.org/10.23919/eusipco54536.2021.9616022>
