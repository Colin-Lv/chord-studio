# 主调识别错误记录集合

> 后续阶段状态、KEY-021/022及新反例见 [第二阶段更新](D:/ChordStudio/Dialogue/docs/key_stage2_error_updates_20261101.md)。本页保留首次审计状态。

日期：2026-11-01。涵盖旧研究、安装版和上一轮新原型，不按作者归罪。KEY-019是未影响当前95例统计的潜在代码缺陷；KEY-020是本轮否定直接接入的研究假设，不能说成已经发生的生产故障。

状态必须区分：结论纠正、实验原型修复、生产待修。当前没有修改安装版。正式证据目录：`D:/ChordStudio/research/key-audit-20261031-r1`。

## 复核通则

1. 每项结论可追溯到输入、实际调用函数与代码版本。
2. 成功原因必须经单因素反事实或消融验证。
3. 语义单测与数据回放并行；高分不免除实现缺陷。
4. 新增失败必须与新增成功一起报告。
5. 不因模型不支持而断言音乐不可能；参考错误需独立证据。
6. 数据集暴露状态、作品分组、局部/全局目标在调参前固定。

## KEY-001 · 把未参与裁决的字段当成成功原因

**优先级/来源/状态：** P1 / 旧实现与报告 / 结论已纠正；生产路径保留

**证据位置：** `D:/ChordStudio/research/key-v3/LAYERS_REPORT.md; D:/ChordStudio/research/key-audit-20261031-r1/legacy_probes.json::bad_girl_causal_probe`

**发现：** 坏女孩实际为 C#m 经 guard_456 改 E；修改 idiom/frame_bonus 不改变 decide 结果。

**纠正：** 撤回“已不依赖456，是+1.20加成获胜”。

**防复发要求：** 成功归因必须提供调用链和单因素消融，区分展示字段与决策字段。

## KEY-002 · 把高频和弦集合称为有序进行

**优先级/来源/状态：** P1 / 旧实现 / 已定位；生产待修

**证据位置：** `C:/Users/33573/AppData/Local/Programs/Chord Studio/backend/app/key_layers.py::score`

**发现：** top 与 template 都是集合；顺序和重复次数没有进入模板匹配。

**纠正：** 仅可称和弦库存覆盖，不能证明4536或其他进行出现。

**防复发要求：** 加入同集合不同顺序的反例；匹配必须保留时间顺序及回归目标。

## KEY-003 · 小调判定依赖标签字符串

**优先级/来源/状态：** P1 / 旧实现 / 已复现；生产待修

**证据位置：** `C:/Users/33573/AppData/Local/Programs/Chord Studio/backend/app/key_decision.py::_inventory; D:/ChordStudio/research/key-audit-20261031-r1/legacy_probes.json::aliases`

**发现：** F-G-Am 保持 Am，F-G-A:min 改 C；音乐内容相同。

**纠正：** 使用解析后三音/五音等语义。

**防复发要求：** Am/A:min/Amin 等别名、等音拼写应给相同证据与结果。

## KEY-004 · 把跨空白与非属和弦计为终止

**优先级/来源/状态：** P1 / 旧实现 / 已复现；生产待修

**证据位置：** `C:/Users/33573/AppData/Local/Programs/Chord Studio/backend/app/key_layers.py::_cadences; D:/ChordStudio/research/key-audit-20261031-r1/legacy_probes.json`

**发现：** G-N-C 和 Gsus4-C 都记1次；目标调式也未核对。

**纠正：** 核对时序连续和双方和弦品质，区分邻接、回归与句尾终止。

**防复发要求：** 空白、未知标签、错误三音、挂留和弦须有反例；不把没有计到终止写成歌曲没有终止。

## KEY-005 · force 绕开主约束且未充分证明替代候选

**优先级/来源/状态：** P1 / 旧实现 / 已定位；生产待修

**证据位置：** `C:/Users/33573/AppData/Local/Programs/Chord Studio/backend/app/key_decision.py::decide`

**发现：** use_chord=(coverage_ok and not veto) or force。

**纠正：** 明确实际决策条件；候选受惩罚不等于另一候选正确。

**防复发要求：** 所有出口记录生效证据；先处理特征语义，再比较统一评分，避免增加嵌套门控。

## KEY-006 · 大调奖励损害总体后将门控当作当然解法

**优先级/来源/状态：** P1 / 旧实现与报告 / 结论已纠正；参数仍待独立验证

**证据位置：** `D:/ChordStudio/research/key-v3/LAYERS_REPORT.md`

**发现：** POP909总体87.57%降至82.07%，大调子集提升不足以证明方法有效。

**纠正：** 同时报告各类与总体，门控只是未验证提案。

**防复发要求：** 固定目标与权重后在作品隔离的新集评测；必须列出改对和改错。

## KEY-007 · 由特征缺失断言歌曲没有明确主音

**优先级/来源/状态：** P1 / 旧报告 / 结论已撤回；证据提取仍未完成

**证据位置：** `D:/ChordStudio/research/key-v3/FAILURE_ANALYSIS.md; D:/ChordStudio/research/key-audit-20261031-r1/melody-probe/results.json`

**发现：** 旧报告将低主和弦/无V-I扩展为人也无法判断；070在主和弦占比0时仍有偏向参考的旋律证据。

**纠正：** 改为“本方法尚未取得足够证据”，不要推断人类能力上限。

**防复发要求：** 区分未提取、未检出、实际不存在；加入旋律、结构和低音检查。

## KEY-008 · 用模型自己的调内指标判参考错误或宣告规则不会错

**优先级/来源/状态：** P1 / 旧报告 / 结论已纠正

**证据位置：** `D:/ChordStudio/research/key-v3/FAILURE_ANALYSIS.md; D:/ChordStudio/research/key-v3/REPORT.md`

**发现：** 调内占比更低被解释成标签有问题；456退步被说成标签口径冲突而非规则错误。

**纠正：** 只能列作核查假设，不能当作已证实归因。

**防复发要求：** 标签核验须独立听审/谱面/来源口径，不以待测模型作裁判。

## KEY-009 · 失败阈值、数量与归因范围错配

**优先级/来源/状态：** P2 / 旧报告 / 已核对原始计数

**证据位置：** `D:/ChordStudio/research/key-v3/FAILURE_ANALYSIS.md; D:/ChordStudio/research/key-audit-20261031-r1/legacy_probes.json::failure_counts`

**发现：** 95例中<10%为22例，<5%为9例；原文末尾把22例写成<5%。

**纠正：** 更正计数；95例全体统计不能直接证明76个关系调错误大多数的原因。

**防复发要求：** 报告表格由逐例记录生成，保留分母、筛选条件和样本ID。

## KEY-010 · 把最长局部调片段标签当成全曲人工主调

**优先级/来源/状态：** P1 / 旧评测口径 / 已审计；参考未改写

**证据位置：** `D:\ChordStudio\Dialogue\app\import_pop909.py::parse_key; D:/ChordStudio/research/key-audit-20261031-r1/legacy_probes.json::reference_scope`

**发现：** 909中152有多个调；导入取单个最长片段；180/843与累计最多调还不一致。

**纠正：** 报告为对导入参考的一致率；局部调和全曲主调分别评价。

**防复发要求：** 保存全部局部标签与生成方式；全曲主调需明确人工或自动参考口径。

## KEY-011 · 反复查看数据后继续按盲测或无过拟合解释

**优先级/来源/状态：** P1 / 历轮实验与解释 / 本轮已明确暴露状态；新集仍缺失

**证据位置：** `D:\ChordStudio\Dialogue\docs\key_coherence_review_20261031.md; D:/ChordStudio/research/key-coherence-v4/replay_initial.json`

**发现：** 权重在看到虚拟及147例后调整；无歌名特例不能证明没有选择偏差。

**纠正：** 全部作为已暴露回归集；保留所有失败尝试。

**防复发要求：** 预先固定实验协议；按作品隔离新集；禁止将历史 sealed 名称当作未看过的证据。

## KEY-012 · 重复作品、逻辑回放与模型泛化混为一谈

**优先级/来源/状态：** P2 / 结果解读风险 / 本轮已校正范围

**证据位置：** `D:/ChordStudio/research/key-audit-20261031-r1/inputs.json; D:/ChordStudio/research/key-audit-20261031-r1/comparison.json`

**发现：** 两个虚拟项目是同曲；110/147仅生产逻辑在历史 raw.lab 的回放；147以小调为主。

**纠正：** 分别标注作品数/版本数、原始输入与全链路范围；有限域模型比较不推广成普遍结论。

**防复发要求：** 结果必须带数据域、版本、独立作品数和输入来源。

## KEY-013 · 持续副属和弦切碎导致总分改变

**优先级/来源/状态：** P1 / 上一轮新原型 / 已修复并通过反例测试

**证据位置：** `D:/ChordStudio/research/key-audit-20261031-r1/tests_before.txt; D:\ChordStudio\Dialogue\tests\test_key_coherence.py::test_secondary_dominant_subdivision_preserves_all_scores`

**发现：** E7两秒-Am两秒在C下-0.4315；E7拆两段后-1.8060。

**纠正：** 计算上下文前合并相邻同构和弦，不跨空白，不修改输入。

**防复发要求：** 所有24候选总分须对纯切分不变；只测回归次数不够。

## KEY-014 · 普通副属判定遗漏五音和七音性质

**优先级/来源/状态：** P1 / 上一轮新原型 / 已修复并通过反例测试

**证据位置：** `D:\ChordStudio\Dialogue\tests\test_key_coherence.py::test_major_seventh_is_not_a_secondary_dominant`

**发现：** Emaj7-Am 被判为普通副属解决。

**纠正：** 限定常规属结构和大/小三和弦目标；复杂变音属保留未建模状态。

**防复发要求：** 与E7正确例配对测试，不把“根音下行五度”直接等同属解决。

## KEY-015 · 非法时间和残缺声学输入被静默接受

**优先级/来源/状态：** P1 / 上一轮新原型 / 已修复并通过测试

**证据位置：** `D:/ChordStudio/research/key-audit-20261031-r1/tests_before.txt; D:\ChordStudio\Dialogue\app\key_coherence.py`

**发现：** 非有限时间、重叠、非正时长未拒绝；候选缺失补极小值、等音重复覆盖。

**纠正：** 严格输入约定；音频通道为空或完整24类，拒绝NaN/Inf/负分/重复键。

**防复发要求：** 边界值、非法值与缺失值测试；数据失败应显式报错，不产生看似有效的调性。

## KEY-016 · 诊断项被注释写成已生效的鲁棒处理

**优先级/来源/状态：** P2 / 上一轮新原型 / 注释已纠正；建模能力待补

**证据位置：** `D:/ChordStudio/research/key-audit-20261031-r1/before/key_coherence.py; D:\ChordStudio\Dialogue\app\key_coherence.py`

**发现：** outside_pitch_share按音符加权，但排名的兼容性仍对整个和弦分类。

**纠正：** 撤回扩展音不会使整和弦失效的注释含义。

**防复发要求：** 每项说明应对应实际分数路径；诊断、排序与显示分别标识。

## KEY-017 · 手工分数和领先差被当作可靠信心

**优先级/来源/状态：** P1 / 历轮评分解释 / 本轮明确限制；尚未校准

**证据位置：** `D:/ChordStudio/research/key-v3/REPORT.md; D:\ChordStudio\Dialogue\app\key_coherence.py`

**发现：** 旧报告用confident描述手调margin；当前声学分数取log后仍非校准概率。

**纠正：** 分数只表示当前排序；不将归一化/对数变换冒称概率校准。

**防复发要求：** 独立校准及可靠性评估前禁止输出概率或自信标签；记录权重来源和调参暴露。

## KEY-018 · 可变current_run、静默来源回退和覆盖输出损害复现

**优先级/来源/状态：** P1 / 上一轮实验工具 / 已修复入口与证据快照

**证据位置：** `D:/ChordStudio/research/key-audit-20261031-r1/before/evaluate_key_coherence.py; D:\ChordStudio\Dialogue\app\audit_key_revision.py`

**发现：** 原入口读取当前run并覆写固定replay.json，只哈希部分依赖，缺文件静默跳过。

**纠正：** 新目录冻结输入及依赖，显式来源，记录排除，禁止覆盖比较结果。

**防复发要求：** 复现需输入哈希和代码哈希同时匹配；新快照不冒充同输入复现。

## KEY-019 · 关系调错误分类未核对调式和方向

**优先级/来源/状态：** P2 / 旧失败分类代码 / 潜在缺陷已确认；当前95例分类未受影响

**证据位置：** `D:/ChordStudio/research/key-v3/failure_analysis.py::interval_class`

**发现：** 仅按根音差3/9即称关系调，未验证大小调相反及正确方向；当前95例重新核对无误分类。

**纠正：** 函数定义需核对完整关系大小调条件，不能据此虚报现有76例统计已经错误。

**防复发要求：** 同调式小三度关系、错误方向与真正关系大小调须分开测试。

## KEY-020 · 把休止前音代理当作乐句终止与可靠主音证据

**优先级/来源/状态：** P2 / 下一步研究假设 / 已实验否定直接接入；保留负结果

**证据位置：** `D:/ChordStudio/research/key-audit-20261031-r1/melody-probe/protocol.json; D:/ChordStudio/research/key-audit-20261031-r1/melody-probe/results.json`

**发现：** 休止前音代理在68旧失败中24偏参考、38偏旧预测、4平手、2缺失。

**纠正：** 尚不能把它叫乐句终止识别，也不能说已修正24首。

**防复发要求：** 诊断候选对不冒称24类准确率；先验证句界与声部，再进行固定协议的增量实验。

## 维护方式

新增问题继续编号；关闭时必须附可执行反例、修复版本与同输入验证结果。只改报告不能把生产缺陷标为已修复；只通过样本也不能把研究假设标为已证实。历史记录保留，补充更正不覆盖原实验。

交接主文：[dsh 交接报告](D:/ChordStudio/Dialogue/docs/dsh_key_handoff_20261101.md)。
