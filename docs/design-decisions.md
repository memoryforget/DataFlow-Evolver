# 设计决策：交付物定位、奖励锚点、数据池与防泄露

> 状态：**设计已定；奖励锚点 / baseline 作 root / Relative Gain / 数据池准入已接线（M7，2026-06-26）**
> 最后更新：2026-06-26
> 本文沉淀关于「交付什么、奖励怎么定、要不要数据池、如何防泄露、如何免按任务调参」的全部讨论结论，指导实现与 paper 写作。
> 实现进度的勾选清单在 [`../TODO.md`](../TODO.md) §M7，本文不重复维护。

---

## 0. 总览（一句话结论）

1. **主贡献 = 双反馈 MCTS 自动发现的「数据工程 pipeline」**；数据（含 data pool）是证据与可选成品，不是优化目标。
2. **奖励 = Relative Gain（相对强基线的改进量）**，per-node；与 data pool **解耦**。
3. **data pool 是可选副产品**（成品交付 / 沉没成本回收 / 诊断驱动的累积覆盖），不参与奖励、默认关闭。
4. **防泄露**：benchmark 切 dev/test，loop 只用 dev、test 终评一次 + 源语料去污染。
5. **超参尺度无关化**：以「相对基线 / 分位数 / 占比」替代绝对阈值，使**同一套配置跨领域不用重调**。

---

## 1. 交付物定位：Pipeline 为主，data 为证据

| | Pipeline 为主（采纳） | Dataset 为主（未采纳） |
|---|---|---|
| 贡献性质 | 方法 / 可复用产物（语料 → 数据 的函数） | 静态样本 |
| 泛化叙事 | 可证明「换语料 / 换 benchmark 仍有效」→ 加分 | 易被质疑「只是对着 benchmark 攒了一份数据」 |
| 与 MCTS 目标一致性 | **完全一致**（搜索目标 = pipeline 产数据的下游分） | 不一致（见 §4.4 目标—产物错配） |
| 可复现性 | 高（交付 recipe 即可） | 一般 |

**paper 叙事**：「提出双反馈 MCTS，自动发现数据工程 pipeline；作为证据，其产出数据训练的模型在 held-out benchmark 上更强（**打败人工强基线**），且 pipeline 可迁移到新领域。」

**两种使用模式（同一方法的两个出口，非两个贡献）：**
- **部署最优单 pipeline**：便宜、可迁移到新语料的持续生成器；
- **收割多样性池**：以算力换性能上限。
- 二者构成一条 **成本 ↔ 性能 权衡曲线**，作为对方法的特性刻画（characterization）。

---

## 2. 奖励与阈值：以强基线为锚（核心——免按任务调参）

### 2.1 问题：依赖分布的绝对阈值不通用

框架要用于**任意 SFT 领域**。若每个依赖分数分布的阈值（池 τ、review 门、per-pipeline 上限）都要按任务人为微调，就违背通用性。解法：**把绝对阈值改写成「以强基线为锚」或「相对/占比」的尺度无关规则**。

### 2.2 强基线是什么

`workspace/_agent_root/dataflow/statics/pipelines/api_pipelines/text_sft_synthesis_pipeline.py`：

| 步骤 | 算子 | 作用 |
|---|---|---|
| 1 | `CondorGenerator`（`num_samples=3`, `CondorQuestionPrompt`, 起点 `empty.jsonl`） | **从零合成** SFT 样本（知识标签驱动，不依赖输入语料） |
| 2 | `CondorRefiner`（`CondorRefinePrompt`） | 精炼 instruction / output |
| 3 | `AlpagasusFilter(min_score=3, max_score=5)` | **LLM-as-judge 打分 1–5，保留 3–5** 的质量过滤 |

产物 schema `instruction / input / output`（Alpaca SFT）；published 方法组合（Condor + Alpagasus），**自带质量过滤**——可信的人工强基线。

### 2.3 核心思想：每任务跑一次基线，得到该任务的尺度

- 每个新任务/领域，**先用基线 pipeline 跑一遍**（固定 `seed=42`、只跑一次）→ 得 `baseline_dev_score`。
- 这个数**就是该任务的尺度**。所有相对规则都挂在它上面，于是同一套配置在数学题、代码题、通用 QA 上都讲得通——无需按任务重调。

### 2.4 基线的三个角色（两内部 + 一外部）

baseline 在框架里扮演**三个不同角色**，勿混为一谈：

**① 搜索根（root）+ ② 奖励锚点 —— 两者在 baseline 作 root 时合一**
- **root**：baseline pipeline 作 node_0（持有真实 config、`visits=1`、`baseline_dev_score` 入账），MCTS 从**强起点变异**。对比：当前实现是**虚拟空根**（`TreeManager.root` 的 `config=None`），第一层从零冷启动。
- **奖励锚点**：`reward = node_dev_score − baseline_dev_score`（Δ=0，>0 即打败基线）。当 baseline 即 root 时，这等于「**比起点进步多少**」——root 自身奖励为 0，概念最自洽。
- **池准入 τ**：`admit ⇔ node_dev_score ≥ baseline_dev_score + Δ`。**无魔数**，门槛由基线自标定。

> 关键技术点：**作为纯奖励平移，锚点不改变搜索**——对所有节点减同一常数，不改变 UCB 的兄弟排序与 argmax。故"依赖好基线"这条**只对 root（warm start）成立，对奖励锚点不成立**（见 §2.6 caveat 5）。

**warm / cold 开关（重要，为消融保留）**
- `baseline.enabled=true`（暖启动）：baseline 作 root + Relative Gain 奖励，求最佳效果与最快收敛。
- `baseline.enabled=false`（冷启动）：退回虚拟空根、**奖励用绝对分 `node_dev`**（见 §2.8），run 内部**完全不碰 baseline**——这是证明"方法不依赖好基线"的消融模式，必须保留为一等公民（见 §5.2）。

**③ 外部对照（论文结果表）—— 不止跟 baseline 比**
- baseline **每个领域评一次即可**（跨 run 复用），**一鱼三吃**：暖启动的 root + Relative Gain 的零点 + 对照表里的一行。**冷启动 run 不依赖它**；它仍要评，是因为论文需要这三项中的任一项。
- **起跑线 / 独立对照行**：baseline（Condor+Alpagasus）本身就是已发表强方法，列为对照之一，展示"我们比它提升多少"。
- **独立 SOTA 行（主结论）**：还须打败其它**独立强方法**（Self-Instruct / Evol-Instruct(WizardLM) / Condor 单用 / Alpagasus 单用 等），否则"只赢自己起点"=循环论证，审稿人不认。
- 全部在 **held-out test** 上报告（配合 §3 防泄露）。

### 2.5 这如何免按任务调参（全超参尺度无关化）

| 超参 | 原（绝对、需按任务调） | 改成（尺度无关） |
|---|---|---|
| 奖励 | 绝对 dev 分 | **相对基线**（Relative Gain） |
| 池 τ | 绝对分 0.6 | **≥ 基线 + Δ**（Δ=0） |
| per-pipeline 上限 | 绝对行数 | **占池比例**（如 ≤30%）；或靠覆盖度去重直接消掉 |
| review 门 | 绝对分 0.6 | **兄弟节点分位数 + 灾难下限**（schema≈0 才硬砍） |

> 残余元参数只剩 `Δ` 与回退参照选择——这些是**跨任务稳定**的量，不是按任务猜的绝对分。逃不掉的不是"按任务调"，而是「少数尺度无关元参数 + 每任务一次基线标定」，与"通用框架"claim 相容。

### 2.6 客观 caveats（写作时照实交代）

1. **同口径对比**：基线与节点须在**同一 dev benchmark、同一训练设置、同一 schema** 下评测。基线输出 `instruction/input/output`，若任务 schema 不同（如 `question/answer`）需字段映射。
2. **范式差异**：基线 **corpus-free**（从零合成）vs 我们 **corpus-grounded**（语料 grounded）。同属 SFT 但输入范式不同，写作时说明，别假装同范式。
3. **换领域时基线须在同一目标领域产数据**：否则对比被"领域覆盖差异"confound 而非"方法好坏"。
4. **基线锚是 Phase-2（训练后 dev 分）工具，不是 Phase-1（review）工具**：review 在训练前无 dev 分，其相对化走「兄弟分位数 + 灾难下限」，勿混淆。
5. **审稿命门「你们依赖一个好基线才有效」**：
   - 此批评**只打到 warm start（baseline 作 root）**，打不到奖励锚点（§2.4 技术点：纯平移不改变搜索）。
   - 更好的起点帮搜索是**所有优化的通性**，非本方法独有弱点。
   - **堵嘴方式 = §5.2 的独立性消融**：冷启动（空根）若也能打败独立 SOTA（只是慢），即证明 baseline 是**加速器而非拐杖**，能力来自双反馈 MCTS 本身。
   - ⚠️ 该消融**承重**：若冷启动打不过 SOTA，则批评成立、贡献退化为"好初始化+轻量搜索"。

### 2.7 已锁定决策

- **Δ = 0**；paper 用 **Relative Gain（相对增益）** 表述，>0 即打败基线。
- **基线固定 `seed=42`、只跑一次**得稳定 `baseline_dev_score`，不做多次平均。
- **任务范围锁定 SFT（仅换领域）**：Condor+Alpagasus 基线对所有目标任务适用，**无 root/parent 回退分支**。
- **baseline 作 root（warm start）= 推荐默认配置，但保留为开关**：冷启动（虚拟空根）必须作为一等模式留存，用于 §5.2 独立性消融。不把框架硬编码到 baseline。
- **对照不止 baseline**：主结果须打败独立 SOTA，baseline 仅作"起跑线"行。
- **变异起点 = 父节点/基线的 pipeline.py 代码注入（2026-06-26 接线确定）**：`build_pipeline_prompt` 新增 `parent_code` 参数，`_step` 把 `parent.config.code` 注入生成 prompt 的【变异起点】块，要求 Agent「在它的基础上做针对性变异」。
  - 暖启动：首层父 = baseline root，其 `config.code` = 适配后的基线源码（Condor+Refiner+Alpagasus）→ 第一层即「对强基线的变异」。
  - 冷启动：虚拟空根 `config=None` → `parent_code=None` → 不注入该块 → 退回从零冷启动，**行为与历史完全一致**。
  - 深层节点（任意模式）：父 = 上一节点，注入其生成代码 → 沿用「在父基础上演化」的统一语义。
  - 选型理由：复用既有 `PipelineConfig.code` 字段，无需读盘；warm/cold 与深浅层用同一条代码路径，零特判。

### 2.8 冷启动下的奖励与门槛（无 baseline 时）

冷启动 run 不依赖 baseline，奖励/门槛按尺度无关的方式自洽：

- **奖励 = 绝对分 `node_dev`**（锚点退化为 0）。因锚点是纯平移（§2.4 技术点），**冷启动用绝对分与暖启动用 Relative Gain 在搜索/选点上等价**——只差一个解释性常数。
- **✅ penalty 一致性（2026-06-26 已接线 `RewardPolicy.penalty`）**：原 `main.py` 硬编码 `PENALTY_REWARD=0.0` 已移除。
  - 绝对分模式（冷启动）：节点分 ≥0，penalty=0 即最差 → 正确。
  - Relative Gain 模式（暖启动）：能跑但逊于基线的节点奖励为负，会比崩溃节点（0）更低 → **排序错乱**。故暖启动把 penalty 也相对锚点设（`penalty = −baseline_dev`，即对应绝对 0），保证失败恒为最差。
  - 落地：penalty 随 `reward_mode` 由 `RewardPolicy` 统一给出，`_step` 的 4 处失败分支均用 `self.reward_policy.penalty`。
- **池门槛**：无基线时 τ 退回尺度无关的相对规则——**超过运行中位数 / top-X% 分位数**（见 §2.5 表 / §4.3），而非绝对 `score_threshold`。
- **（可选，非默认）parent-relative 奖励** `node_dev − parent_dev`：奖励"这次变异比父节点强多少"，信用分配更密；但**它不是全局平移、会改变搜索行为**，风险更高，列为后续探索。第一层父为空根 → 退回绝对分。

---

## 3. 防数据泄露纪律

**泄露 A — 搜索对 benchmark 过拟合**：MCTS 反复用 benchmark 当奖励 → 在「刷这个 benchmark」；再用同一 benchmark 报告 → 分数虚高。等价于"反复用验证集调超参却当测试集报告"。
**泄露 B — 内容泄露**：生成数据的源语料与 benchmark 测试题重叠 → 等于把答案喂进训练集。

**解法（不放弃 Phase-2 真实反馈）：**
1. **benchmark 切 dev / test**：
   - `dev`（held-in）：**只**给 MCTS 当奖励信号，允许被刷（`trainer.dev_eval_cmd`）；
   - `test`（held-out）：**全程对 loop 不可见**，仅在选出 best recipe 后跑**一次**用于报告（`trainer.test_eval_cmd`）。
   - **dev/test 切分是训练脚本内部职责**，框架把 benchmark 当黑箱、只决定调哪条命令（不传 split、不介入切分）。
2. **去污染（decontamination）**：生成源语料与 `test` 题目做 n-gram 重叠过滤，剔除重叠项。
3. **（可选）多 seed / 多 split 降方差**。

**为何不只靠 ReviewAgent**：它是 Phase-1 快筛，只查表面 schema+相关性，**不知道数据能否让模型变强**。只靠它会退化成 ANDES 式「纯合成+启发式打分」，丢掉真实下游反馈这一核心卖点。**正解 = 保留 Phase-2 但评测放 dev、test 严格隔离**。

> **Phase-1 失败也要反馈(已接线,2026-06-23)**：节点被 ReviewAgent 打回时：
> 1. **同节点立即自修正环**（`review_self_correction_retries`，默认 **5** 次）：把 `review_result.issues` + **各 step 中间产物的行数概览** 喂给 Agent SDK，让它修复后重新执行+重审，过审才进 Phase-2，限次后才扣 0 分；
> 2. 利用 **DataFlow 每个算子落一个 step cache jsonl** 的特性：自修正 prompt **告知 Agent 这些中间产物存在**（`{node_dir}/cache/`，按执行顺序 step1=第1个算子产出…，可对照 `pipeline.py` 的 `forward()`），让它**自己用 Glob/Read 查看中间数据**、逐 step 追踪质量从哪一步（哪个算子）开始变坏——既含行数骤减/变空，也含行数没变但内容劣化（字段变 null/空、跑题等）。框架不预算行数/样例，交由 Agent 判断（保持简单）。
> 3. 此外 `review_issues` 也会注入"下次在该节点下扩展"的生成 prompt（跨节点指引），与同节点重试互补。
> 至此**执行环与评审环对称**：执行崩溃喂 traceback（限 `self_correction_retries`），评审失败喂 issues+中间产物（限 `review_self_correction_retries`）。
> 注：ReviewAgent 当前是 **0.6 绝对阈值的硬门**，其相对化（兄弟分位数+灾难下限，见 §2.5）仍为设计、未接线。

---

## 4. Data Pool（可选副产品，与奖励解耦）

### 4.1 真实价值（均与「是否赢过单 pipeline」无关）

1. **可直接使用的成品**：用户要能喂训练的数据，而非一条还得自配 SDK/key/算力去跑的 pipeline。
2. **沉没成本回收**：几十个节点已花了真金白银的预算；无池则除最优节点外全部丢弃，有池则变现。
3. **承载学习轨迹（最 framework-intrinsic）**：双反馈环是「按诊断逐轮补短板」的增量过程。单节点只是轨迹上一个快照；**只有池**装得下「所有诊断驱动补丁之并集」=模型真正需要的覆盖图。这正是区别于 ANDES（产完即丢、不累积）的核心。多样性是**诊断驱动的针对性覆盖**，不是随机多样性。

### 4.2 何时不值得做池

- 任务窄而同质（单 pipeline 已全覆盖，池只加冗余）；无可靠质量门控（池被噪声拖累）；严格走"pipeline 即方法"想保持聚焦时可缓做。

### 4.3 准入策略（待实测定参）

- **Pipeline 级**：必须 `review.passed`；且 `node_dev_score ≥ baseline_dev_score + Δ`（Δ=0，见 §2.4；基线关闭时回退到绝对 `data_pool.score_threshold`）。研究级可选「边际贡献」（加入后池 dev 分是否提升，贪心子模），但每候选需在 `池∪节点` 重训，贵。
- **行级**：默认「准入 pipeline 全部行 − 去重」；现阶段无可靠逐行质量模型，激进逐行过滤引偏差。
  - **去重**：`content_hash` 精确去重起步（确定性、零依赖）。**嵌入近邻去重明确推后**——它与 bad-case 上采样冲突（失败模式数据往往故意相似，会被误删）。
  - **命中失败模式的行不丢、应上采样**（bad-case-aware 的精髓）。
  - **per-pipeline 占比上限**（`data_pool.max_rows_per_pipeline`，尺度无关），防单 pipeline 淹没池。

### 4.4 与奖励解耦 + 目标—产物错配

- **奖励 per-node**（训练只用当前节点数据，见 §4.5），池**只收割、不参与奖励**。
- 与「每个节点把自己那片做到最好、池把各片加总」是**互补**关系——**前提是变异确实覆盖新 gap**；否则池全是冗余、价值落空。故池价值兑现依赖去重 + 覆盖度追踪 + 足够 exploration。
- **错配警告**：若既用 per-node 奖励搜索、又把 union 池当头牌成果，则优化器优化单 pipeline 质量、却交付 union，二者不保证一致（高多样性但平庸的 pipeline 会被误杀）。**本项目通过 §1「pipeline 为主」定位规避**。若未来改主打 pool，须把奖励改成**边际贡献**（训练 `池∪节点`），接受非平稳奖励与更高算力。

### 4.5 Stage-2 训练范围：只用当前节点数据（保持现状）

- 现状（`main.py`）：`trainer.run(exec_result.dataset_path, ...)` —— 只喂当前节点 jsonl，不是池。
- 在"pipeline 为主"定位下**正确且应保持**：奖励=「这条 pipeline 的数据有多好」，与实际部署方式一致。

---

## 5. 实验设计与对照

### 5.1 对照集（主结果表，全部在 held-out test 上）

| 行 | 作用 |
|---|---|
| **起跑线**：DataFlow 人工 baseline（Condor+Alpagasus）原样 | 展示"我们比出发点提升多少" |
| **独立 SOTA**：Self-Instruct / Evol-Instruct(WizardLM) / Condor 单用 / Alpagasus 单用 等 | **主结论**：发现的 pipeline 不止赢起点，还赢领域 |
| **本方法（warm start）** | baseline 作 root 的最佳配置 |
| **本方法（cold start）** | 空根、绝对分奖励——**独立性消融**（见 §5.2） |
| **消融行**（见 §5.2） | 隔离各组件贡献 |

> 只跟"自己出发的 baseline"比 = 循环论证。主张必须建立在打败**独立** SOTA 上。
> baseline **每个领域评一次**即可，跨 run 复用（既是此表的起跑线行，也是暖启动 root 与 Relative Gain 零点）；冷启动 run 不依赖它。

### 5.2 承重消融（回应"依赖好基线"的命门，见 §2.6 caveat 5）

1. **独立性消融（最关键）**：冷启动（`baseline.enabled=false`，虚拟空根）vs 暖启动（baseline 作 root）。
   - 期望：冷启动**最终也打败独立 SOTA**（只是更慢/更多迭代）→ 证明 baseline 是加速器非拐杖。
   - ⚠️ 若冷启动打不过 → 贡献退化为"好初始化+轻量搜索"，需如实承认。
2. **搜索贡献消融**：单次 LLM "设计一条 pipeline"（无 MCTS）、随机搜索（同预算）vs 本方法 → 隔离 MCTS/UCB 的作用。
3. **双反馈消融**：去掉 Phase-2（只 ReviewAgent）vs 完整双反馈 → 证明真实下游反馈的价值。

### 5.3 池 vs 单 pipeline（待检验假设，非预期结论）的受控项

1. **等量对比（最高优先级）**：池下采样到与单 pipeline 同 example/token 数（或反之）。否则「池更好」只能归因于**数据量更大**而非多样性 → 实验会被毙。
2. **显式多样性指标**：嵌入离散度 / n-gram 熵 / 领域覆盖率，证明池确实更多样。
3. **池质量门控 + 去重**：否则池可能被噪声拖累反而更差（quality–diversity trade-off）。
4. **test 全程隔离 + 源语料去污染**（§3）。
5. **诚实报告**：池没赢同样有信息量（「单 pipeline 已足够，池边际收益被噪声抵消」）。

---

## 6. 配置映射

| 配置项 | 含义 | 当前默认 |
|---|---|---|
| `baseline.enabled` | warm start：baseline 作 root + Relative Gain 奖励 + τ≥基线；关=冷启动（虚拟空根，供独立性消融 §5.2） | false（**已接线**；开启会先跑一次基线消耗预算） |
| `baseline.reward_mode` | `absolute`（冷启动，奖励=node_dev）/ `relative_to_baseline`（暖启动，Relative Gain） | absolute（**实际模式由 enabled 决定**：暖→relative，冷→absolute；此项仅文档化意图） |
| `baseline.margin` | Δ：超过基线多少算正向 | 0.0 |
| `baseline.seed` | 基线随机种子，只跑一次 | 42 |
| （隐含）失败 penalty | **随 `reward_mode` 切换**：absolute→0；relative→`−baseline_dev`（见 §2.8） | **已接线**（`RewardPolicy.penalty`） |
| `trainer.dev_eval_cmd` | dev 评测（驱动 MCTS） | 空 → **已接线**回退 `train_eval_cmd` |
| `trainer.test_eval_cmd` | test 终评（仅一次，手动） | 空 |
| `data_pool.enabled` | 收割数据池 | false |
| `data_pool.score_threshold` | **冷启动**时的绝对 τ 回退（暖启动改用 `baseline_dev+margin`，见 `DataPool.admission_threshold`） | 0.0（以 review.passed 为真实门槛） |
| `data_pool.max_rows_per_pipeline` | per-pipeline 上限（拟改占比） | 0（不限） |
| `data_pool.dedup` | `content_hash`（精确）/ `embedding`（未实现） | content_hash |

> 已锁定但仍待实测重设的硬约束：**一旦有真实 dev 分布，须据基线 + 分位数确认 Δ 与占比上限**。

---

## 7. 落地状态

- **已完成（脚手架/文档）**：本设计文档；`core/data_pool.py`（接口 + 内容哈希去重 + 上限）；`configs/default.yaml`（`baseline` / `data_pool` / `trainer.dev|test_eval_cmd` 段，默认关闭）；README dev/test + 去污染 + 数据池说明。
- **已接线（M7，2026-06-26，受 `baseline.enabled` 开关控制）**：
  - `core/baseline.py`：`RewardPolicy`（绝对分 / Relative Gain + penalty 随模式切换）+ `adapt_baseline_source`（基线源码最小可行适配：api_url/model_name/key_name/empty.jsonl/cache_path）。
  - `main.py`：暖启动启动时跑一次基线（seed=42，复用 `SubprocessExecutionWrapper` + `DF_PIPELINE_API_KEY` 注入 + `dev_eval_cmd` 评测）得 `baseline_dev_score`；`TreeManager.set_baseline_root` 把基线作 root（visits=1、reward=0）；回传奖励走 `RewardPolicy.reward`，失败 penalty 走 `RewardPolicy.penalty`；第⑧步按 `node_dev ≥ baseline_dev+margin`（暖）/ `score_threshold`（冷）收割数据池并落盘。
  - 变异机制：`parent_code` 注入生成 prompt（见 §2.7）。
  - dev 评测：`trainer.dev_eval_cmd` 留空回退 `train_eval_cmd`。
  - 冷启动（`enabled=false`）：不跑基线、虚拟空根、绝对分奖励、penalty=0——行为与历史一致（27 个既有测试全绿）。
  - 单测：`tests/test_baseline.py`（warm/cold 路径、Relative Gain、penalty 切换、池门槛、源码适配）。
- **待实测后实现**：行级 bad-case 上采样、嵌入去重、去污染脚本、双评测受控实验、可选边际贡献奖励。见 [`../TODO.md`](../TODO.md) §M7。
- **注**：`best_leaf()` 仍只在 root 的子节点中挑选（不含 baseline root 自身）。暖启动若所有子节点均逊于基线，返回的是「最不差的子节点」而非基线本身——属次要语义点，留待实测时按需调整。
