# 设计决策：交付物定位、Init node、输入契约、评分口径与防泄露

> 状态：**2026-07-31 架构重整完成**——删除 `baseline` 与 `data_pool` 两个模块，引入 Init node、
> `input/` 输入契约、评分策略（`EvaluationStrategy`）、外部数据搜索与去污染硬指标。
> 本文沉淀「交付什么、奖励怎么定、数据从哪来、如何防泄露、如何免按任务调参」的结论，指导实现与 paper 写作。
> 实现进度清单在 [`../TODO.md`](../TODO.md)，本文不重复维护。

---

## 0. 总览（一句话结论）

1. **主贡献 = 双反馈 MCTS 自动发现的「数据工程 pipeline」**；数据是证据与副产品，不是优化目标。
2. **Init node（node_0）= 树根 = 整棵树的变异起点与参照系**；奖励就是节点自己的分，不做任何平移。
3. **输入范式由 `input/` 目录内容决定**：有语料 = corpus-grounded 合成，空 = from-scratch 合成。
4. **外部数据可搜可取**（HuggingFace），但必须落到共享 `data_links/`、留溯源、过去污染；
   溯源不假设 Agent 守规矩——每个节点独立记录工具调用与代码里的外部数据引用（§3.3）。
5. **防泄露是硬指标**：benchmark 切 dev/test，loop 只用 dev；合成数据与 test 做 n-gram 去污染，超阈值直接判不通过。
6. **超参尺度无关化**：以「相对 node_0 / 分位数 / 占比」表述，使同一套配置跨领域不用重调。

---

## 1. 交付物定位：Pipeline 为主，data 为证据

| | Pipeline 为主（采纳） | Dataset 为主（未采纳） |
|---|---|---|
| 贡献性质 | 方法 / 可复用产物（语料 → 数据 的函数） | 静态样本 |
| 泛化叙事 | 可证明「换语料 / 换 benchmark 仍有效」→ 加分 | 易被质疑「只是对着 benchmark 攒了一份数据」 |
| 与 MCTS 目标一致性 | **完全一致**（搜索目标 = pipeline 产数据的下游分） | 不一致（优化单 pipeline 却交付数据并集） |
| 可复现性 | 高（交付 recipe 即可） | 一般 |

**paper 叙事**：「提出双反馈 MCTS，自动发现数据工程 pipeline；作为证据，其产出数据训练的模型在
held-out benchmark 上更强，且 pipeline 可迁移到新领域。」

### 1.1 为什么删掉 data pool（2026-07-31）

原设计有一个跨节点累积的高质量数据池（`core/data_pool.py`），定位是"可选副产品"。删除理由：

- **与主张不一致**：定位是 pipeline 为主，却维护一个数据交付物，等于给自己开了第二条叙事线。
  一旦池成为卖点，就出现**目标—产物错配**：优化器优化的是单 pipeline 的分，交付的却是并集——
  高多样性但平庸的 pipeline 会被奖励误杀，而池的价值恰恰依赖它们。要消掉这个错配就得把奖励改成
  「边际贡献」（在 `池∪节点` 上重训），非平稳奖励 + 数倍算力，代价远超收益。
- **准入规则是复杂度的主要来源**：门槛、分位数、跨源去重、per-pipeline 占比 water-filling……
  一整套只为一个默认关闭的副产品服务的机制，还牵连 checkpoint 与配置。
- **想要并集随时可得**：所有节点的数据都在各自 `node_N/` 目录里，事后拼一份是几行脚本的事，
  不需要在主循环里长期维护一套准入策略。

**节点数据各存各的**，不设全局池——这也是 DataMaster 的做法。

---

## 2. Init node：起点、参照与消融开关

### 2.1 原设计的问题：一个 baseline 承担三个角色

被删除的 `baseline` 模块让一条"人工强基线流水线"同时充当：① MCTS 的根（暖启动）、
② 奖励锚点（`reward = node_dev − baseline_dev`）、③ 论文对照表里的一行。三者被同一个开关
（`baseline.enabled`）绑在一起，于是：

- 想换个起点（比如用自己已有的 pipeline 起步）就得改基线模块，而这会同时改掉奖励口径；
- 奖励锚点其实是**纯平移**（对所有节点减同一常数，不改变 UCB 的兄弟排序与 argmax），
  也就是说它**根本不影响搜索**，只影响数字的可读性——为一个不影响搜索的量维护一套
  `RewardPolicy`（还得处理 penalty 随模式切换、否则失败节点排序错乱）不划算；
- 角色 ③ 本就是论文层面的事：跑一次别人的方法、把分数写进对照表，与框架内部无关。

### 2.2 新设计：root 就是一个普通实验节点

**Init node = node_0 = `TreeManager.root`**，持有真实 `PipelineConfig`，和其余节点走**完全同一条**
评分路径（`Orchestrator._evaluate_node`），没有特殊评分口径、没有特殊奖励。

- **奖励 = 节点自己的分**（Tier-1 复合分或 dev 真分），失败恒为 `PENALTY_REWARD=0`。
  两种口径的分数都非负，所以 0 天然是最差，不需要随模式切换的 penalty。
- **"相对起点提升了多少"** = `score − node_0_score`，在报告层算即可。因为整棵树都是 node_0 的
  后代，这个差值的含义比原来的 Relative Gain 更直白，且不必在搜索里维护任何锚点状态。
- **起点跑不出分就直接终止**：没有可变异的基础，继续搜索只是在空转烧预算。

### 2.3 三档启动模式（消融的一等公民）

| 模式 | 配置 | 语义 |
|---|---|---|
| 参考实现改写 | `init.reference_pipeline: <path>` | 把一份已有实现**忠实**改写成本框架契约下的流水线：不改数据处理逻辑、不增删步骤、不顺手优化。用途是把外部方法搬到同一评测口径下做对照 |
| 指令构建 | `init.instruction: "..."` | 按一句自然语言指令设计初始流水线 |
| 自主设计 | 两项都留空 | Agent 自行设计一条**最小可行**流水线 |

两项均可被 `--init-code` / `--init-instruction` 命令行覆盖：消融要在同一份 YAML 上换档，
改配置文件既易漏改，也让"这次跑的是哪一档"不可追溯。

**为什么给了参考实现仍要过一次 LLM，而不是直接拷进去跑**：参考实现不满足本框架的契约
（写到 `{node_dir}/pipeline.py`、`first_entry` 指向 `InputCorpus.entry_path`、cache 落节点目录、
serving 配置按环境变量名注入），那次改写在做实事。曾经的 `adapt_baseline_source` 走的是另一条路
——正则机械改写源码，确定性好但对着别人写的代码极脆，已随 baseline 模块删除。
DataMaster 在同一问题上做了同样的选择：它的 `--initial-code` 也只作为 prompt 参考，由 Initial Agent
"忠实重构"成 template+dataloader 两件套，prompt 里明写 *preserve the original algorithm* /
*MUST NOT optimize the model, add new data*。

Init node 的 prompt 里写死了一条硬约束：**稳定跑通优先于精巧**。整棵树都从它变异出去，
起点跑不稳等于整轮搜索报废；复杂度本就该由后续节点按评审反馈逐步加上去。

### 2.4 变异起点：父节点代码注入

`build_pipeline_prompt` 的 `parent_code` 把父节点已验证可运行的 `pipeline.py` 注入子节点 prompt
的【变异起点】块，要求「在它的基础上做针对性变异」。有了 Init node 之后，**首层节点的父就是
node_0**，因此除 node_0 自身外这段恒存在——不再有"冷启动首层无父可继承"的分支。

同时注入的还有父节点的 `rationale`（设计意图，避免子节点误改父节点刻意的取舍）、
`parent_review`（Tier-1 五维归因，指明该攻哪一维）与兄弟节点已试方向的负面清单。

### 2.5 搜索边界：训练代码冻结，只搜数据侧

**搜索空间必须只含数据侧**，否则奖励归因不成立：一个节点分高，说不清是数据变好了还是训练配方
碰巧更合适，而我们主张的是「发现数据工程 pipeline」。

DataMaster 用文件权限把这条边界钉死：每个节点有 `code_{id}_template.py`（训练+验证+写 submission）
和 `code_{id}_dataloader.py`（数据处理）两份代码，Initial 节点两份都能写，Black/Red 节点**只能写
dataloader**——它们挂的 MCP 工具（`for_datanode.py`）里 `write_code` 物理上就拒绝写 template，
template 由父节点复制继承、只读。

我们的切法相同，边界还更靠外：训练+评测是框架外的一个脚本（`trainer.dev_eval_cmd`），
连 Init node 都改不了；每个节点变异的只有 `{node_dir}/pipeline.py` 与 `operators/`。
Agent 根本没有接触训练代码的入口，不需要额外加权限约束。

代价是我们丢了 DataMaster 天然具备的一样东西：它的 template 是一份**每节点的磁盘文件**，
事后总能查出某个分数是哪份训练代码跑出来的；我们的训练配方是一行配置 + 一堆 `SFT_*` 环境变量，
中途被改不留痕迹。所以补了 `run_manifest.json`（§8）——把命令、影响分数的环境变量与脚本内容
摘要写进每个节点的训练输出目录，让"跨节点可比"从假设变成可核验的记录。

#### 2.5.1 冻结训练配方留下的洞：数据量仍在买算力

冻结训练代码并不等于冻结了训练**预算**。总优化步数是 `epochs × N / global_batch`，`epochs`
是常量而 `N` 是节点自己产的——于是数据条数直接决定算力预算。一个多产三倍数据的节点白拿三倍
梯度步数，它分高了说不清是数据更好还是单纯训得更久，§2.5 想守的奖励归因照样不成立。更糟的是
这条捷径比"改进数据"好走得多：MCTS 会如实地学会多产数据，而这与数据质量无关。

`generation.target_samples` 对 pipeline 只是**建议值**——经 `DF_TARGET_SAMPLES` 注入，算子可以
不读它，过滤步骤会让实际产出低于它，增强步骤会让它翻倍。所以"同规模"只能在数据进训练脚本前
切齐：`core/train_set.py` 定量物化，`CappedTrainer` 包在 `self.trainer` 外面，使搜索期打分与
终选 Tier-2 两条独立的训练路径**由构造处一次保证**都受约束（漏掉任一条，那批节点就在不同算力
预算下打分，且不会报错）。抽样确定性、保序，与 dev 集共用 `core/rows.py` 的同一套原语。

切齐管不到反方向：产出**不足** cap 是同一个问题的另一半（欠训），补不出数据来，只能记进
`train_set.manifest.json` 的 `short` 字段并告警，让它可见而不是安静地混进分数。

DataMaster 面对的是完全相同的局面——template 冻结、Red node 下载外部数据后训练集可任意膨胀——
但两端都没有约束：prompt 里只有一句 "Control Data Scale"，给的理由还是训练时间而非可比性，
代码层没有任何条数上限或自动 subsample。这是我们比它多守的一条。

想做"更多数据是否更好"的消融时，`generation.enforce_target_samples: false` 显式把数据量放回
搜索空间——关键在于这是一个要主动做的选择，而不是默认状态。

---

## 3. 输入契约与数据来源

### 3.1 `input/`：范式由目录内容决定，不设开关

`core/input_corpus.py` 扫描 `project.input_dir`：

- **目录里有数据文件** → **corpus-grounded 合成**：入口（`first_entry_file_name`）指向真实语料，
  prompt 要求流水线在这份语料之上加工（抽取/改写/增强/蒸馏/过滤/配对），产出必须能追溯到输入。
- **目录空或不存在** → **from-scratch 合成**：入口指向 0 字节 `empty.jsonl`（仅为满足 DataFlow 的
  `first_entry` 形参），流水线凭模型知识从零合成。

不做成配置项，是因为「配了 grounded 却没语料」「放了语料却按 scratch 跑」两种组合都只可能是误配：
前者必崩，后者白白浪费用户准备的数据。让目录内容作为唯一事实来源，就没有可漂移的第二处状态。

**data preview 注入 prompt**：入口文件的行数、字段（取样本并集，稀疏字段也要露出）与前 3 行原文
（长字段截断）都进生成 prompt。Agent 不必先写一个探针流水线去看数据长什么样。预览是**尽力而为**的：
格式解析失败只少一段上下文，绝不让整轮搜索起不来；唯一会启动即报错的是"显式指定的入口文件不存在"。

多个数据文件时按「格式优先级（jsonl > json > csv > parquet）+ 路径字典序」确定性地选入口并**告警**，
可用 `project.input_entry` 显式指定——静默挑一个再让人去日志里猜是哪个，比多打一行告警糟糕得多。

### 3.1.1 产出 schema 由下游脚本决定，不由 LLM 现编（2026-08-02）

`input/` 的 schema 是任意的（用户给什么是什么），下游要的是另一套字段，中间靠算子做映射——
这条 prompt 里一直写着。缺的是**「另一套字段」到底由谁说了算**。

原先由 `TaskProfiler` 让 LLM 现编 `target_schema`，配置里没有任何锚点。而真正的契约写在训练
脚本的取字段逻辑里。当前 PTB 官方路径的桥接脚本兼容多个别名，但框架权威契约仍应固定：

| | instruction 侧 | output 侧 |
|---|---|---|
| `posttrainbench_sft_eval.py` | `instruction` / `question` / `prompt` / `input` / `query` | `generated_cot` / `output` / `answer` / `response` / `completion` / `solution` / `code` |

于是有一条完整的静默错误链：Profiler 给任务编出 `{problem, solution}`，流水线照做，
ReviewAgent 拿**同一份编出来的 schema** 去量 `schema_score` 和 `null_rate`，双双满分放行；
但训练桥接按自己的字段表转换数据时可能一条都读不到，抛"数据集为空或字段不匹配"。框架把它记成
「训练脚本非零退出」，节点判死。**GPU 烧完才暴露，归因还是错的**；部分不匹配更隐蔽，
不合规的行被转换逻辑静默丢掉，只表现为条数变少。

现在 `trainer.dataset_schema` 是权威：写进 Profiler prompt（让模型的 quality_criteria 也围绕
真字段展开），解析后覆盖模型输出（prompt 说了不许改，模型仍可能改），续跑时对 checkpoint 里的
旧 schema 同样覆盖。它随后自动流到三处消费点——生成 prompt 的目标 schema、评审 prompt、硬指标
的 `required_fields`——因为它们本来就都读 `profile.target_schema`。

再配一道纯代码的硬门槛：关键字段空缺率 ≥ `review.null_threshold`（默认 0.5）直接判不通过，
并把要求的字段名写进 issues。字段名整体对不上时空缺率≈1，一定触发；个别行缺字段则仍按比例扣
`tier1`，不毙节点。放在评审阶段而不是训练阶段，是因为**这一步不花 GPU 也不花 LLM**，而且评审
自修正循环拿到"要求 [instruction, output]"就能直接改对。

这条和 DataMaster 的解法方向相反，值得说清楚。它不定数据 schema，定的是**代码接口**：
`base_dataloader.py` 由系统提供、运行时拼在最前面，契约是 `setup()` 设好 `train_data`/`test_data`、
`get_data()` 返回二元组；**具体形状由 Initial 节点的 LLM 连同 template 一次生成**，天然自洽，
之后 template 冻结，后续节点改坏形状就跑不起来——结构上穿帮，不需要额外校验。

我们学不了这条，因为训练脚本是框架外一份**早就存在的固定脚本**，它吃什么字段已经定死，Init node
没有资格"定义"它。所以方向反过来：把脚本的既有契约显式声明出来，让它成为权威。代价是配置里多一
处需要与脚本保持同步的信息——留空即退回旧行为（并告警），毕竟不是每个下游都有可枚举的固定字段。

### 3.2 外部数据搜索（默认关闭）

`agent_sdk.external_data: true` 时挂载两台 in-process MCP：`dataset-search`（取数，
`llm/mcp_datasets.py`）与 `data-discovery`（找线索，`llm/mcp_discovery.py`，可用
`agent_sdk.discovery: false` 单独关掉）。

**动机**：光靠模型知识从零合成，数据多样性的上限就是模型自身分布；而真实公开语料是搜索空间里
一整个维度——"换一个数据源"往往比"再调一版 prompt"的增益大得多。

四条纪律写死在**取数**工具的行为里，而不是只写在 prompt 里：

1. **落盘集中**：一律进共享 `data_links/`，按 `(hub, dataset_id, config, split)` 跨节点复用同一份
   文件。键里带 hub：同名数据集在 HF 与 ModelScope 上未必是同一份内容。
2. **可溯源**：每个文件旁落一份 `.meta.json`（来源 / dataset_id / config / split / URL /
   license / 行数 / 字节 / 是否完整 / 时间）。数据从哪来是论文与去污染审计都要回答的问题，事后
   从文件名猜不出来。license 独立取一次落盘（复用 inspect 阶段的结果），因为 Agent 在
   `inspect_dataset` 里看到的 license 只存在于它的上下文里，交代许可时看的是磁盘上的边车。
3. **默认取全量 split，不做"前 N 行"截断**（见 §3.2.1）。
4. **只取单个 split，不整仓库拉**：整仓库会把 `test` split 一起拖下来，那正是 §4 花力气防的
   泄露入口。落盘仍受 `dataset_search.max_download_mb` 字节上限保护，触顶截断并在 meta 标明。

依赖（`huggingface_hub` / `datasets` / `modelscope` / `requests`）惰性导入，网络不通或缺包只会让
**对应那一个工具**返回一条可读说明，不影响主循环——这是可选增强。

**默认关闭**的理由：是否允许引入外部数据属于实验设定的一部分（影响结论的可比性），必须显式打开。
且开启后 benchmark 泄露风险显著上升——Hub 上大量数据集本身就是各类 benchmark 的镜像或衍生，
故开启时**务必同时配好去污染参考集**（§4.2）。

#### 3.2.0 两个 hub、两类工具：为什么这么切

**ModelScope 作为第二个 hub**，加它的理由不是"多一个源"，而是**国内网络下 HF 常常直连不通**。这
时可选项只有配镜像或换源，而 ModelScope 上中文语料的覆盖本来也更好。三个工具与 HF 侧一一对应，
落盘实现完全共用（`_download(hub, args)`）——字节上限、原子落盘、溯源字段、复用提示这几条纪律，
任何一个 hub 漏掉都等于开了个后门，而"两份实现慢慢长歪"是这类重复最典型的下场。两个 hub 只在
`_row_iter()` 一处分叉。

ModelScope 检索走 SDK 的 `HubApi.list_repos("dataset", search=...)`。最初这里手写了一条社区流传的
`PUT /api/v1/dolphin/datasets`，实测直接 404；SDK 那条不仅是官方接口，还一次就把 license / tags /
tasks / downloads / gated 都带回来，等于 HF 侧 search + inspect 两步的信息量——所以 ModelScope 组
只有三个工具而非四个。

**脚本型数据集不支持，且这是有意的**：ModelScope 上仍有不少 legacy 的 script dataset（`modelscope/gsm8k`
就是），加载它们要求 `trust_remote_code=True`，也就是在编排器进程里执行仓库里的任意代码。我们不开
这个开关，而是把错误翻成"换一份 parquet/jsonl 存储的镜像仓库"——同一份语料在 ModelScope 上通常有
多个 owner，换一个就能拿到。

**`data-discovery` 与取数分开**，因为它们回答的是不同的问题。Hub 的关键词搜索**不支持语义匹配**，
"想要小学数学应用题的 CoT 语料"这种意图，往往得先在论文/博客/GitHub 里看到某个数据集被用过，才
知道它叫什么。三个工具（Serper 网页 / GitHub 仓库 / arXiv 论文）**只发现、不下载**，返回里统一带
一句"拿到 id 后回 `download_dataset` / `download_ms_dataset`"——任何真正取数的动作都必须经落盘工具
留溯源，否则 §3.3 那套记录就漏了。

坦白说，相比 SDK 内置的 `WebSearch`/`WebFetch`，这三个的增量有限：结构化结果 + 统一的下一步指引，
仅此而已。DataMaster 有对应的四台 MCP（HF / GitHub / Google / Scholar），但**其中只有 HF 能真正
下载数据集**，另外三台同样只返回线索。所以两边"可取数的源"数量其实是我们 2 : 它 1，多出来的是
发现渠道的形式化程度。凭据缺失（Serper 是付费服务）按**工具**降级而非整台 server 罢工——少配一个
key 不该让 arXiv 和 GitHub 也用不了。

#### 3.2.1 为什么默认下载整个 split（对齐 DataMaster，但换了实现）

早先的实现默认只流式取前 5000 行。三个问题：

1. **它替搜索做了一个决策**。取多少条、怎么采样、按什么过滤，本就是**流水线该做的事**，也正是
   被搜索的对象。固化在工具里，等于在搜索空间里先砍掉一维，而且是 MCTS 无从推翻的一刀。
2. **"前 N 行"不是随机样本**。数据集若按难度 / 来源 / 时间排过序，头部就是有系统性偏差的一段，
   而 Agent 会拿它当"这个数据源的样子"来判断适不适用。
3. **复用键不含 `max_rows`，会静默改变数据量**。node_3 拉 5000 行、node_7 想拉 50000 行，命中缓存
   直接拿回 5000 行。而"加大外部数据量"恰是一个自然的变异方向——node_7 会在数据量根本没变的
   情况下拿到分数，MCTS 据此得出"加数据没用"的错误结论，日志里还看不出来。

改成默认全量后，第 1、2 条消失，第 3 条退化成"缓存文件是否完整"，由 meta 的 `complete` 字段和
复用提示显式回答。

DataMaster 也是全量下载（`hf_hub_download` 遍历整个仓库），但实现上我们没有照抄：

| | DataFlow-Evolver | DataMaster |
|---|---|---|
| 范围 | 单个 split → jsonl | 整个仓库所有文件 |
| 读取方式 | 流式边读边写，内存与数据集大小无关 | 逐文件下载；连 `get_dataset_sample` 取 5 行都先 `load_dataset` 整个 split |
| 上限 | `max_download_mb` 字节上限，触顶截断并标明 | 无行数/大小上限，仅 600s 超时 |
| 落盘安全 | 先写 `.part` 再原子改名 | 直接写，已有文件可能被覆盖 |

差异来自领域：DataMaster 做 MLE-Bench 的 Kaggle 任务，图像/CSV 竞赛数据必须完整、还常需要仓库里
的非表格资产，整仓库下载是对的；我们做文本 SFT 语料合成，一个 split 的 jsonl 就是流水线的输入，
拉整仓库只会额外把 test split 拖进来。

上限用**字节**而非行数：一行的大小在不同数据集之间能差几个数量级，行数上限对"会不会拉爆磁盘"
这个真正要防的问题几乎没有约束力。

#### 3.2.1.1 SDK 签名漂移：stub 掉的单测看不见

`search_datasets` 曾经在**每一次调用**上都报 `TypeError: list_datasets() got an unexpected keyword
argument 'direction'`——`huggingface_hub` 1.x 移除了该参数，而我们还在传。单测全绿，因为它们把
`_hf_api()` 整个 stub 掉了，测的是我们自己的封装契约，恰好把"我们调得对不对"这一层挡在了外面。

补法不是去打真网络（慢、不稳、CI 里跑不了），而是拿**真实已安装的 SDK 签名** bind 一次参数：

```python
inspect.signature(hf.HfApi.list_datasets).bind(None, search="q", sort="downloads", limit=5)
```

零网络、毫秒级，却正好覆盖 stub 留下的那个洞。四处调用（HF 的 list/info、`load_dataset`、
ModelScope 的 `list_repos`/`MsDataset.load`）都加了，SDK 缺失时 `importorskip` 跳过。

#### 3.2.2 连接配置为什么必须走 `dataset_search` 段

`dataset-search` 与 `data-discovery` 都是 **in-process MCP**（`create_sdk_mcp_server`），工具的
handler 跑在编排器自己的 Python 进程里，而 `agent_sdk.env` 那一段是发给 Agent SDK **子进程**的
（见 `_agent_env`）。
把 `HF_TOKEN` 放进 `agent_sdk.env` 不会生效，失败表现却是"搜索返回网络错误"，很难联想到是变量
没送到。故连接信息单列 `dataset_search` 段，由 `configure()` 在启动时写进 `os.environ`。

时机也是硬约束：`huggingface_hub` 在 **import 期**就把 `HF_ENDPOINT` 读成模块常量，导入之后再改
环境变量无效。所以本模块所有 hf 导入都在函数内部（惰性），`configure()` 在 `Orchestrator.__init__`
里调用，早于任何一次工具调用。

### 3.3 溯源不能建立在「Agent 守规矩」上（`core/provenance.py`）

上面三条纪律只约束**走工具**的那条路。而 Agent 的内置工具是全放开的（`claude_code` 预设，
无白名单），它完全可以直接在流水线里写 `load_dataset("openai/gsm8k")`、`hf_hub_download(...)`
或 `requests.get(url)`——数据在执行时才拉进来，`data_links/` 里什么都不留。第三种尤其自然：
Agent 本来就熟 `datasets` 库，写进流水线比"先调工具、再引绝对路径"省事。

去污染与溯源回答的是两个不同的问题，都得能答：

| 问题 | 由谁回答 | 层次 |
|---|---|---|
| 这个节点的产出有没有混进 benchmark 题？ | `core/decontamination.py` | 数据层，与来源无关 |
| 这些数据是从哪来的、许可是什么？ | `core/provenance.py` | 来源层 |

第一个问题即使来源完全无痕也能答（扫的是产出内容），所以**泄露风险本身是兜住的**。但检出污染
之后要问"这批题从哪个源进来的、还有哪些节点用了同一个源"，就必须有来源记录。因此每个节点目录
落两份：

- **`agent_tools.jsonl`**：Agent 每次运行调过的工具（含参数，长串截断），按 `phase` 标出是生成
  还是第几轮自修正。**按尝试追加**而非只记最终结果：失败那次的下载已经落了盘，重试不会把文件
  收回去，只记最后一次会漏掉它们。
- **`data_provenance.json`**：评分收尾对节点代码（`pipeline.py` + `operators/**.py`）做一遍静态
  扫描，把 `data_links/` 引用归到本节点名下（附 `.meta.json`），并把绕过工具的 `load_dataset` /
  `hub_download` / `hf:`·`ms:` 远程 URI / 数据类 URL 挖出来标成 `unmanaged: true`。**成败都扫**——
  "拉了不该拉的数据集导致污染扣分"恰恰只发生在失败节点上，只在成功路径记录等于漏掉主要场景。

`storage_uri` 那一类是补进来的，值得单独说：DataFlow 的 `FileStorage` 原生支持把
`first_entry_file_name` 写成 `hf:owner/name:config:split` 或 `ms:owner/name:split` 直接拉远程数据，
而我们自己的 `dataflow-dev` 技能 knowledge base 正好教了这个写法。它比 `load_dataset` 更顺手（一个
字符串就完事），却完全不长得像"下载"，前三条正则一个都不命中——是**最可能被走、也最容易漏掉**的
绕行路径，而且漏掉时 `unmanaged` 会安静地保持 false。扫描器与 prompt（含 `external_data` 关闭时的
那条禁令）现在都专门覆盖它。这也说明一件事：溯源的盲区不在"Agent 会不会故意绕"，而在"我们自己的
工具链提供了几条我们没想到的合法捷径"。

URL 判定要区分数据源与 LLM serving：每个节点都会调 serving API，若 `https://` 一律算外部数据
引用，`unmanaged` 就恒为真、失去分辨力。判据取"域名是公开数据托管站，或路径以数据文件扩展名
结尾"，其余归入 `other_urls` 仅作参考。

**已知限度**（`data_provenance.json` 里的 `method: "static_scan"` 就是在标它）：正则扫不到运行时
拼出来的 dataset id 与 URL（f-string、变量拼接）。这不是加几条正则能补上的，真要堵死得靠沙箱
断网。定位是"把绝大多数情况变得可查"，不是"保证无遗漏"。

对应地，prompt 侧两段纪律成对出现：工具开启时明说"所有外部数据必须经这四个工具获取，不要直接
`load_dataset`/`requests`/curl"；**工具关闭时同样要说**"本轮不得联网拉取任何外部数据"——工具没
挂载不等于 Agent 没法联网，少了这段，"本轮不引入外部数据"就只存在于配置里，prompt 对 Agent
只字未提。

---

## 4. 防数据泄露

**泄露 A — 搜索对 benchmark 过拟合**：MCTS 反复用 benchmark 当奖励 → 在「刷这个 benchmark」；
再用同一 benchmark 报告 → 分数虚高。等价于"反复用验证集调超参却当测试集报告"。
**泄露 B — 内容泄露**：合成数据与 benchmark 测试题重叠 → 等于把答案喂进训练集。

### 4.1 dev / test 隔离（针对泄露 A）

- `dev`（held-in）：**只**给 MCTS 当奖励信号，允许被刷（`trainer.dev_eval_cmd`）；
- `test`（held-out）：**全程对 loop 不可见**，仅在选出最终模型后跑一次用于报告（`trainer.test_eval_cmd`）。

### 4.1.1 dev 题集由框架物化（2026-07-31）

原先 dev 题集是在评测子进程内部现抽的（`math_train.py` 里 `random.Random(42).sample(rows, 500)`），
抽样口径、随机种子、条数全在框架之外。改为由 `core/dev_set.py` 在启动时按 `(source_sha, seed, size)`
物化成 `workspace/eval/dev.jsonl` + manifest，并以 `--dev-set` 显式下传。三条理由，按重要性排：

1. **可去污染**。这是最要紧的一条。dev 是搜索**直接优化的目标**，把 dev 题蒸馏进合成训练数据
   比混 test 更能刷分——test 全程不参与 loop，混进去在搜索期一分钱好处都没有，混 dev 则立刻涨分。
   而我们的 dev 恰恰抽自 `MATH train` 这种 Hub 上遍地都是的公开语料，开了外部数据搜索后 Agent
   拉到原始题库再蒸馏 CoT 是很自然的一步。但题集只短暂存在于子进程内存里时，**这个洞根本无从
   检测**：没有文件可以拿去当去污染参考集。物化之后它自动并入索引（不依赖用户记得配）。
2. **可比性**。奖励是节点之间比出来的，前提是所有节点评在同一批题上。抽样发生在每个节点的评测
   子进程里，就意味着"同一批题"只是一条约定：谁把 `MATH_TRAIN_SAMPLE_SIZE` 调一下，前后节点
   就在不同题集上比，MCTS 的排序被静默污染，不报错也不留痕。物化把它从约定变成事实，口径还进
   checkpoint，续跑时对不上直接抛错。
3. **一处切分、处处复用**。每个评测脚本各写一套抽样必然漂移——MATH 用 `seed=42` 抽 500，
   PostTrainBench 用 `--dev-limit 150`，已经是两个机制了。物化后脚本退化成"在这个文件上评测"。

> **落地时踩过一次**：物化下传了，但 `load_math_train` 的内置抽样没有为 `--dev-set` 让路，
> 拿到题集后照样按 `MATH_TRAIN_SAMPLE_SIZE` 再抽一次。因为两边默认都是 500，`len > 500`
> 不成立，恰好从未触发；把 `dev_set.size` 改成 800 就会静默评 500 条，而 manifest 记的是 800。
> 「框架已经物化了」和「脚本确实不再抽样」是两件事，后者要脚本侧配合才成立。
>
> 第一版修法是「显式传了题集就不抽样」，但这只是让两套机制**不冲突**，平行的抽样档位还在。
> 最终改成**脚本侧完全没有抽样能力**：`load_math_train` 的 `data_path` 变成必填，没有默认
> 题库、没有 `MATH_TRAIN_SAMPLE_SIZE`，`--mode dev` 缺 `--dev-set` 直接报错。代价是
> `dev_set.source` 留空时这条链路跑不了 dev（旧行为是脚本自己挑 500 条），这是想要的：
> 有退路的约定迟早会被走回去，而每一次都不报错。

这一条**采纳的是 DataMaster 提出的纪律，但换了执行层**，两者的差别值得说清楚。

DataMaster 把规则写在 prompt 里，理由很直白——"All nodes must be evaluated on the same validation
set so that metrics accurately reflect improvements"（`prompts/tools/dataloader_en.md`），并要求 Agent
优先读 `input/val.csv`、禁止随机切。**但代码库里没有任何地方生成这个文件**：MLE-Bench 的
`prepare.py` 只产出 `train.csv` / `test.csv` / `sample_submission.csv`，全库检索也找不到 `val.csv`
的写入端。而同一份 prompt 紧接着给了兜底——"If `input/val.csv` does not exist: You may only split
train/val from the original competition training set"，模板 dataloader 就照此实现了
`train_test_split(..., random_state=42)` 的回退，更多 `initial_code/*.py` 干脆不检查直接切。

也就是说，那条纪律在 DataMaster 里是**约定**，没有强制机制；文件缺席时（这是默认情况）每个节点各切
各的，正是上面第 1 条要防的漂移。我们把它落成了代码：框架启动时物化、`--dev-set` 显式下传、口径进
checkpoint、对不上直接抛 `DevSetChanged`。**Agent 全程碰不到这份题集**——它只写数据流水线，无权决定
模型在哪些题上被评。

还有一层结构性差异：DataMaster 的 val 是从任务自带训练集里切出来的，而我们没有可切的训练集（训练
数据是每个节点现合成的），所以我们固定的是**评测子集**，不是 train/val split。

`source` 留空只表示**框架不物化**，不代表评测脚本就该自己挑题。当前 PTB 官方默认配置就是留空：
官方 `evaluate.py` 读死自己的题池，验证反馈通过 `--limit N` 给出，终评通过 `--limit -1`
给出。留空这个选项存在，是因为不是每个下游任务都有一份可枚举的 dev 文件；但"框架不管"与
"脚本随便"之间没有默认通路，得由脚本作者显式决定并写进它自己的契约。
换口径不静默改写而是抛 `DevSetChanged`：已跑节点的分数出自旧题集，混在一棵树里比较就是错的。

**benchmark 本身仍是黑箱**：框架只决定"评哪批题、调哪条命令"，不介入判分逻辑与 test 切分。

### 4.2 去污染硬指标（针对泄露 B，已接线）

`core/decontamination.py` + `ReviewAgent`：把合成数据与 benchmark 测试集做**归一化后的 13-gram
精确重合**检测，得 `contamination_rate`。

- **为什么必须是硬指标**：搜索的目标函数就是下游分数，而把测试题混进训练集是提高这个分数最省力的
  路径。只要泄露不被扣分，MCTS 迟早会找到它——这不是 Agent"不听话"，是我们的奖励函数在鼓励它。
- **两级处置**：按比例打折 `tier1 *= (1 - contamination_rate)`（少量疑似重合可能是假阳，
  不该一票否决），同时设硬门槛 `max_rate`（默认 2%），超过即 `passed=False` 并把命中片段作为
  issue 喂回 Agent——不给证据，它下一轮还会这么干。
- **口径细节**：归一化（小写、去标点、压空白）让"改标点就绕过"失效；n-gram 而非整句，因为改写过的
  题目仍会保留长片段；短文本（token 数 < n）退回整串精确匹配，否则最容易泄露的短题反而检不出；
  CJK 按字切（先给每个 CJK 字符两侧补空格再统一按空白切），否则中文整句会变成一个 token、
  n-gram 恒为空。
- **合规性**：去污染只做命中判定，不读 test 的分数、不把 test 内容回流给任何 Agent。

### 4.3 Phase-1 反馈环（评审失败也要有反馈）

节点被 ReviewAgent 打回时：

1. **同节点立即自修正环**（`review_self_correction_retries`，默认 5）：把 `issues`（具体证据原文）
   喂给 Agent，修复后重新执行 + 重审，过审才进 Phase-2，限次后才判失败。
2. 利用 DataFlow **每个算子落一个 step cache jsonl** 的特性：自修正 prompt 告知 Agent 中间产物位置
   （`{node_dir}/cache/`，按执行顺序对照 `pipeline.py` 的 `forward()`），让它自己逐 step 追踪质量
   从哪个算子开始变坏。框架不预算行数/样例，交由 Agent 判断。
3. 自修正 prompt 另注入三样：`failure_signals` 标准化标签（与 `issues` 分开保存，不混进同一列表）；
   `history`＝本节点前几轮的「针对什么 → 观察到什么变化」摘要（硬指标 delta 标为"确定变化"，
   阈值 0.02；五维与 tier1 标为"参考变化"，两轮抽样不同、judge 波动本身约 ±0.1，不足以归因）；
   以及 `decision.json` 同步要求（只改代码会让算子构成与设计意图停留在旧版本）。
4. **证据落盘**：每次评审追加一行 `node_N/review.jsonl`（含抽样原文与 LLM 原始响应，供事后核验
   tier1 是否可信、离线换权重重算）；每轮自修正追加一行 `node_N/fix_history.jsonl`。均不进
   checkpoint：节点在一次迭代内是原子的，续跑不会接着跑半途的自修正循环。
5. **跨节点反馈走 `parent_review`**：父节点的 Tier-1 归因（五维、最弱维点名、硬指标、命中信号、
   issues 原文）注入子节点生成 prompt，**无论父节点评审是否通过都注入**——搜索期这是唯一存在的
   质量信号。（曾设想"打回后把 issues 注入下次扩展"，但评审未过的节点会置 `dead`、永远不会被
   `select()` 选为父节点，那是死代码，已删。）

至此**执行环与评审环对称**：执行崩溃喂 traceback（限 `self_correction_retries`），
评审失败喂 issues + 信号 + 历史修复轨迹并告知中间产物位置（限 `review_self_correction_retries`）。

---

## 5. 评分口径（EvaluationStrategy）

`search.reward_source` 选择一个策略对象（`core/evaluation.py`），而不是散落各处的 `if`：

| | `train` | `review` |
|---|---|---|
| 搜索期奖励 | 真实 SFT + dev 评测的**真分** | ReviewAgent **Tier-1 复合分**（代理分） |
| 搜索期数据规模 | `generation.target_samples`（全量） | `generation.probe_samples`（探针，注入 `DF_TARGET_SAMPLES`） |
| 搜索期是否占 GPU | 是 | 否 |
| 诊断（bad case） | 有 | 无（没有真实 bad case 可分析） |
| 收尾 | 不需要 | **必须**：对 top-K 放大重产 + 真实 SFT + 评测，把代理分兑现成真分 |
| 时间预算预留 | 0 | `tier2_top_k × tier2_per_run_sec + final_eval_reserve_sec` |

把这些绑在一个对象里，是因为它们必须同时成立：注入了探针规模却按训练分打分、或按 Tier-1 打分却
不做收尾兑现，都会得到一个看起来能跑、结论却不成立的运行。散成若干处 `if reward_source == ...` 时，
加第三种口径就意味着去每一处补分支，漏一处不会报错。

策略**无状态**，协作者（trainer / diagnostician / 规模配置）在调用时从编排器读取：编排器始终是这些
对象的唯一持有者，策略只表达"这一口径怎么做"。

**未知口径名直接抛异常**：拼错的 `reward_source` 若静默回退成默认值，一次跑几小时的实验会安静地
跑成另一种实验，只有逐行看日志才可能发现。

---

## 6. 实验设计与对照

### 6.1 对照集（主结果表，全部在 held-out test 上）

| 行 | 作用 |
|---|---|
| **起跑线**：Init node（node_0）自身 | 展示"搜索比起点提升多少"；用参考实现改写模式时，这一行就是那份外部方法在同一口径下的分 |
| **独立 SOTA**：Self-Instruct / Evol-Instruct(WizardLM) / Condor / Alpagasus 等 | **主结论**：发现的 pipeline 不止赢起点，还赢领域内已发表方法 |
| **本方法** | 搜索得到的最优 pipeline |
| **消融行**（见 §6.2） | 隔离各组件贡献 |

> 只跟"自己出发的起点"比 = 循环论证。主张必须建立在打败**独立** SOTA 上。

### 6.2 承重消融

1. **起点独立性消融（最关键）**：自主设计起点 vs 参考实现改写起点。
   - 期望：自主设计起点**最终也打败独立 SOTA**（只是更慢/更多迭代）→ 证明强起点是加速器而非拐杖。
   - ⚠️ 若自主设计打不过 → 贡献退化为"好初始化 + 轻量搜索"，需如实承认。
   - 注：起点只影响搜索**从哪出发**，不影响奖励口径（新设计里奖励就是节点自己的分），
     所以这个消融干净地只隔离"起点质量"这一个变量。
2. **搜索贡献消融**：单次 LLM「设计一条 pipeline」（无 MCTS）、随机搜索（同预算）vs 本方法
   → 隔离 MCTS/UCB 的作用。
3. **双反馈消融**：只用 Phase-1（`reward_source=review` 且不做收尾）vs 完整双反馈
   → 证明真实下游反馈的价值。
4. **输入范式消融**：`input/` 有语料（grounded）vs 空（from-scratch）→ 刻画方法对输入语料的依赖。
5. **外部数据消融**：`external_data` 开/关 → 量化"允许 Agent 找真实数据"带来的增益，
   并配合去污染率报告说明增益不是来自泄露。

### 6.3 客观 caveats（写作时照实交代）

1. **同口径对比**：起点与节点须在同一 dev benchmark、同一训练设置、同一 schema 下评测。
2. **范式差异**：若对照方法是 corpus-free（从零合成）而我们是 corpus-grounded（语料 grounded），
   写作时说明，别假装同范式。
3. **换领域时对照方法须在同一目标领域产数据**：否则对比被"领域覆盖差异"confound 而非"方法好坏"。
4. **Tier-1 是代理指标**：`reward_source=review` 下搜索期排序依据是 Tier-1，它与下游真分的相关性
   须用收尾阶段的 `(tier1, dev_score)` 样本对实测报告，不能默认成立。
5. **ReviewAgent 的 `passed` 仍是绝对阈值硬门**（schema/relevance 0.6），其相对化（兄弟分位数 +
   灾难下限）仍为设计、未接线。

---

## 7. 配置映射

| 配置项 | 含义 | 默认 |
|---|---|---|
| `project.input_dir` | 输入语料目录；有数据=grounded 合成，空=from-scratch | `./input` |
| `project.input_entry` | 多个数据文件时显式指定入口（相对 `input_dir`） | 空（自动选） |
| `project.data_links_dir` | 外部数据集下载目录（跨节点共享） | 空 → `<workspace>/data_links` |
| `init.reference_pipeline` | Init node 参考实现路径 → 忠实改写模式 | 空 |
| `init.instruction` | Init node 自然语言构建指令 | 空（两项都空=自主设计） |
| `search.reward_source` | 奖励口径：`train` / `review`；未知值直接报错 | `train` |
| `generation.target_samples` | 全量产出条数（train 口径搜索期 / review 口径收尾重产） | 800 |
| `generation.probe_samples` | review 口径搜索期探针条数 | 未设=沿用 target |
| `search.tier2_top_k` | review 口径收尾按 tier1 取 top-K 做真实 SFT+评测 | 3 |
| `search.time_budget_sec` | 整轮（搜索+收尾+终评）总墙钟预算；0=只按 max_iterations 停 | 0 |
| `agent_sdk.external_data` | 挂载 dataset-search（+ data-discovery），允许搜索/下载外部数据集 | false |
| `agent_sdk.discovery` | external_data 打开时是否连带挂 data-discovery（只找线索那台） | true |
| `dataset_search.endpoint` / `.token` / `.cache_dir` | HF 连接信息 → `HF_ENDPOINT` / `HF_TOKEN` / `HF_HOME`；**不能放 `agent_sdk.env`**（§3.2.2） | 空（沿用环境） |
| `dataset_search.max_download_mb` | 单份数据集落盘上限；触顶截断（留 split 头部）并在 meta 标明 | 2048 |
| `dataset_search.modelscope_endpoint` | ModelScope 站点（镜像/自建） | `https://modelscope.cn` |
| `discovery.serper_api_key` | `search_web` 用；留空=仅该工具不可用，另两个照常 | 空 |
| `discovery.github_token` | 留空=匿名搜索（GitHub 限流 10 次/分钟） | 空 |
| `trainer.dev_set.source` | dev 题集源文件；留空=不物化，而数学链路的评测脚本会因拿不到题集拒绝跑 dev | `data/math_train.jsonl` |
| `trainer.dev_set.size` / `.seed` | 固定抽样条数（0=全量）与种子；三者共同构成 dev 口径 | 500 / 42 |
| `review.decontamination.reference_files` | benchmark 测试集文件；留空=不检测（dev 集自动并入） | `[]` |
| `review.decontamination.ngram` | n-gram 长度；短文本退回整串精确匹配 | 13 |
| `review.decontamination.max_rate` | 污染率上限，超过即 `passed=False` | 0.02 |
| `trainer.dev_eval_cmd` | dev 评测（驱动 MCTS） | 必填 |
| `trainer.test_eval_cmd` | test 终评（review 口径下由编排器在终选后自动跑一次） | 空 → 由 dev 命令派生 |

---

## 8. 落地状态（2026-07-31）

**已接线**

- `core/input_corpus.py`：`input/` 契约、范式判定、data preview 注入 prompt。
- Init node：`Orchestrator._run_init_node` + `PipelineAgent.generate_initial` + 三档启动模式；
  `TreeManager.best_node()` 把 root 纳入终选候选（没有任何变异超过初始流水线时，就该交付它）。
- `core/evaluation.py`：`TrainScore` / `ReviewScore` 策略 + 未知口径名报错。
- `llm/mcp_datasets.py`：外部数据集搜索/预览/下载 + `data_links/` 溯源 sidecar（默认关闭）。
- `core/decontamination.py`：n-gram 去污染索引 + `ReviewAgent` 打折与硬门槛。
- `core/provenance.py`：节点级 `agent_tools.jsonl`（工具调用轨迹）+ `data_provenance.json`
  （外部数据来源静态扫描，绕过工具的直连引用标 `unmanaged`），见 §3.3。
- `core/dev_set.py`：dev 题集物化 + manifest + 自动并入去污染参考集 + checkpoint 口径核对（§4.1.1）。
- 训练留痕：每次训练在 `<out_dir>/run_manifest.json` 记命令 / 影响分数的环境变量 / 脚本内容摘要。
  训练配方散在框架之外（命令行 + `SFT_*` 环境变量），跑到一半被改动会让节点分数不可比，而树里
  看不出任何差别——DataMaster 靠"template 作为每节点磁盘文件"天然留痕，我们的训练脚本是全局的，
  只能靠指纹补上。
- 断点续跑 checkpoint `SCHEMA_VERSION=3`（移除 reward_policy / pool 两段，root 即 Init node；
  另记 dev 集口径，续跑不一致即报错）。

**已删除**

- `core/baseline.py`（`RewardPolicy` / `adapt_baseline_source` / `set_baseline_root`）——见 §2.1。
- `core/data_pool.py` 及其准入策略——见 §1.1。

**待实测后推进**

- ReviewAgent `passed` 门槛的相对化（兄弟分位数 + 灾难下限）。
- Tier-1 与下游真分的相关性实测（用收尾阶段的样本对）。
- 外部数据消融与去污染率的联合报告。
