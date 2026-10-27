# 混合检索 RAG 系统 · 架构文档

> 版本：v1.8（活跃文档，随实施演进）
> 日期：2026-10-25
> 决策来源：五轮选型沟通（2026-09-06 ~ 09-09）+ 本机环境探查 + 三 Agent 对抗性审查（2026-09-09）+ M1/M2 实施后架构对齐沟通（2026-09-15）+ 存储层服务化修订（2026-09-17）+ 范围调整（2026-09-18：放弃 E0~E4 对照实验，优先 RAG 系统可用性，评测简化为单列 RAGAS）
> v1.0 → v1.1 变更：① Python 环境改用 miniconda py312；② 架构图改纯文字描述；③ 评测定版——LLM 合成 Q&A + RAGAS（替换原"评测搁置"决策）
> v1.1 → v1.2 变更（对抗性审查后修订）：① M4 查询集定版手写 15~20 条，G4 裁判校准移入 M5，Gate 重编号；② 语料收口 50~60 份（本地模式 ≤2 万点舒适区）；③ sparse 实现路径定版——自写 jieba → BM25 → {indices, values} 映射，不用 fastembed 内置 BM25；④ 评测方法学三项风险（金标准偏倚/同家族裁判/可证伪性）记入 §9，M5 实施前定对策；⑤ 技术勘误：reranker 显存、RAGAS embedding/来源映射坑、M5 工时上调
> v1.2 → v1.3 变更（M1/M2 实施落地后对齐）：① **ED6 修订**——索引入库从"LlamaIndex 全包"改为 **qdrant-client 直连**，LlamaIndex 仅保留切块职责（原因见 ED6）；② **存储层改为三个命名向量**（dense + bm25 + bge_sparse），新增 ED9；③ **对照实验矩阵裁剪**——E0~E3 定为主线（共用一份索引），E4 降级为可选，砍掉切分粒度对照与元数据过滤开关对照（原因见 §7.2）；④ 切块参数按实际产物定版 **768/80**（原 512/50）；⑤ §3.1 元数据 schema 对齐 chunk 产物实际字段；⑥ §5 目录结构改**扁平化**（src 下每模块单文件，不建子目录）；⑦ 语料实测 **60 份 / 12.07 万 chunk**，远超原 ≤2 万点预估，§9 相应风险升级为高
> v1.3 → v1.4 变更（存储层服务化）：**向量库从 Qdrant 本地模式（`QdrantClient(path=...)`，SQLite 持久化）切换为服务端模式（Docker 容器 + `QdrantClient(host=, port=)`）**。原因为本机已安装 Docker Desktop 并部署 qdrant 容器（版本 1.19.1），ED3"本机无 Docker"的前提不再成立，且实测 4 万点集合本地模式加载偏重。① **ED3 修订**——记录由本地模式改服务端的决策演变；② 连接参数集中到 `config.yaml` 的 `qdrant` 段（host/port/api_key），新增 `index_documents.get_qdrant_client()` 全项目统一建连；③ 存储持久化移交容器卷，`data/qdrant_storage/` 目录废弃（不再由代码读写，历史数据保留作备份）；④ §9 收敛"本地模式单进程持锁死锁"风险（服务端模式不再有 `.lock`），新增"容器需先启动"运维前置。检索算法、三命名向量口径、payload schema、RRF 融合**均无变化**
> v1.4 → v1.5 变更（范围调整：放弃对照，优先可用性）：① **方向大调**——E0~E4 对照实验矩阵（原 §7.2）降级为 §10 演进路线，不再是主线交付；三命名向量（dense/bm25/bge_sparse）与客户端 RRF 实现照存，将来恢复对照零改造成本；② **M4/M5 合并为单一评测里程碑**——单列 E3（完整混合链路），50 道合成题（自写轻量生成器，不用 RAGAS TestsetGenerator），RAGAS 四指标（ContextRecall/ContextPrecision/Faithfulness/AnswerRelevancy），免人工金标准标注；③ **裁判解耦**——评测裁判 LLM 走硅基流动 DeepSeek（与生成侧百炼 qwen 异家族），AnswerRelevancy embedding 走硅基流动 BGE-M3 API（评测进程不加载 torch）；④ **文档漂移修正**——§1 语料数据更新为实测 4.06 万 chunk / 平均 482 字符 / table 占比 52.3%（title 已并入前缀占比 0），§9 清理已消解风险（粒度过细、title 占比高、本地模式持锁），v1.4 changelog"RRF 无变化"与 M3 状态"客户端 RRF"矛盾修正；⑤ **§1 成功标准改写**——从"E0 vs E3 对照表"改为"RAG 链路端到端可用 + RAGAS 四指标初步报告"；⑥ G3/G4 门、手写查询集、跨公司对比型查询、pooling 标注、文本→点映射全部删除
> v1.5 → v1.6 变更（BM25 口径迁移：idf 交 Qdrant 服务端）：① **稀疏 BM25 分工重划**——df/idf 等随语料变化的全局统计改由 Qdrant 服务端维护（`bm25` 命名稀疏向量挂 `Modifier.IDF`，Qdrant ≥1.10 特性，容器 1.19.1），客户端入库只存 tf 饱和+长度归一值（k1/b 不变，avgdl 冻结为 `sparse.avgdl_frozen=248.54`）；**新增/替换文档零全局重算、不再全量重传**——原入库端阶段 B 两遍统计、`bm25_stats.json` 持久化、"检索端必须复用同一份 idf 表"约束全部取消（§3"Qdrant 不自己算 BM25"论断修订：idf 服务端管、k1/b 长度归一仍客户端）。查询端逻辑不变，仍是每命中词权重 1.0；② **增量入库能力就绪**——日常 `--codes <code>` 只动新报告，整份替换加 `--replace`（按 stock_code+fiscal_year 删旧点再入库），§1 非目标移除"增量入库"、§10 演进项 2 标注已具备；③ ED9 补记：bm25 挂 IDF 后与 bge_sparse（不挂）不再同构，恢复 E4 需把 idf 加权差异计入；④ §9 登记已知风险：删点/覆盖后服务端 df 递减滞后到 segment 合并（qdrant/qdrant#6690、#6735），小规模自愈；⑤ 迁移需一次性 `python src/index_documents.py --recreate --skip-encode` 重建（npz 缓存不动 GPU），前后等价性用 `_bm25_migration_check.py` dump/compare 验证
> v1.6 → v1.7 变更（M3 收尾：接口断裂修复 + 联调）：① **retrieve.py 重构收编**——2026-09-30 的精简版造成 ui.py/llm_client.py 调用断裂（v1.5 口径的 exp/flt/rerank 返回结构被删），本次以精简版为基础**合并回实验组能力**：`retrieve(query, exp, flt, rerank) -> list[dict]`（EXP_ROUTES 表驱动 E0~E4，每项含 payload 11 字段/各路分数/rerank_score），阶段耗时移入 `r.last_timings_ms`，新增 `build_filter()`（codes/年份区间→Filter）、`_load_dotenv()`（config/.env 兜底）、`n_points`；② **重排默认按实验组定**——E3 默认重排，E0/E1/E2/E4 默认不重排（对照变量是召回路），显式 rerank 可覆盖；重排失败仍硬抛不降级（§9 对应风险行已改口径）；③ **rerank 响应解析修复**——openai SDK 3.3.0 对 `cast_to=dict` 抛 ValueError（construct_type 对裸 dict 走 get_args 解包失败，2026-10-17 实测），改 `cast_to=dict[str, Any]` 新旧版本均兼容；④ **UI/生成联调完成**——ui.py 对齐新接口（重排失败展示错误不回退、meta 行含 encode/召回/rerank/llm 计时），gradio 6.30.0 安装并实测启动（127.0.0.1:7860 探活 200）；⑤ **LLM 生成侧遗留**——token plan key 对 qwen3.8-flash 返回 403 AccessDenied.Unpurchased（模型未购/套餐不含），生成降级路径按设计工作（返回原因不抛异常），待账号侧解决后零代码改动即可恢复；⑥ 语料统计勘误——data/chunks 实测仍为 4.06 万块（40,609），collection 实测 40,192 点，差 417（重切块后残点/未重灌部分），以 collection 为权威运行口径
> v1.7 → v1.8 变更（M4 评测启动：LLM 切换 + 评测方案定版）：① **LLM 生成侧切换**——百炼 qwen3.8-flash（token plan 403 遗留）→ **智谱 glm-5.3-flash**（coding plan 端点 `https://open.bigmodel.cn/api/coding/paas/v4`，key=GLM_API_KEY），端到端联调通过（检索+重排+生成带引用）；② **评测裁判定版**——DeepSeek 官网 `deepseek-flash`（config 新增 judge 段，key=DEEPSEEK_API_KEY），硅基流动职责收窄为仅 rerank；③ **评测方案改版（用户定版）**——弃 RAGAS（实装 0.4.3 import 即挂：依赖的 `langchain_community.chat_models.vertexai` 已被 langchain-community 0.4.2 移除；降级 ragas 0.3.9 需连带降 langchain 全家桶、有波及 gradio 6.30 的连锁风险），**自写五检索指标：Recall@5 / Precision@5 / HitRate@5 / MRR / NDCG@5**，金标准=出题源 chunk（唯一命中口径，用户定版），另加 DeepSeek 裁判 5 分制答案侧参考分；④ **evaluate.py 落地**——gen/run/score 三子命令单文件（分层抽样 seed=42 可复现、题型配额 term18/paraphrase18/numeric14、run 断点续跑、score 分题型分层+最差 10 题归因 → data/eval/report.md），小样本 3 题全链路验证通过；⑤ **环境变量兜底增强**——retrieve._load_dotenv() 现兼读 Windows Machine 注册表（WorkBuddy 会话不继承机器级环境变量的已知问题，GLM/DeepSeek/硅基流动 key 三段统一兜底）；⑥ §7 评测章与本 changelog 同步，RAGAS 相关依赖注意/风险行改写为历史口径
> 文档体系：**本文档为唯一活跃架构文档**，自包含可独立阅读；`01-技术选型对比.md`（v1）与 `01-技术选型对比-v2.md` 降为历史留档，仅供追溯选型过程
> 实施状态：**M0~M3 全部落地**（M3 2026-10-17 收尾：E0~E4+过滤+重排实测通过，gradio 6.30 UI 可用；LLM 生成 2026-10-25 切智谱 glm-5.3-flash 后联调通过，403 遗留解除）；**M4 评测正式跑批完成（2026-10-25，50 题）**——Recall@5=0.58 / MRR=0.433 / NDCG@5=0.470 / 裁判 3.72；分题型：numeric 0.857 > term 0.722 ≫ paraphrase 0.222（改写题召回是最大短板）；待质量闸人工抽核 10 题；**v1.6 BM25 口径迁移已生效**（collection 实测 bm25 挂 Modifier.IDF、单 segment）；语料口径：collection 40,192 点（chunks 产物 40,609，差 417 重切块残留，以 collection 为准）

---

## 1. 项目定位

**一句话定位**：面向上市公司年报的中文混合检索 RAG 系统——搭建 dense + sparse 双路召回 + RRF 融合 + 重排的**可用端到端链路**，以 RAGAS 四指标产出初步质量报告；E0~E4 对照实验（量化混合检索增益）降级为演进路线（§10），三命名向量与客户端 RRF 照存，将来恢复对照零改造成本。

| 维度 | 内容 |
|---|---|
| 语料 | 巨潮资讯网上市公司年报 PDF，**60 份实测**（20 家 × 3 年，覆盖金融/家电/白酒/医药/汽车/能源/科技等行业）。切块产物实测 **4.06 万 chunk**（v1.7 核验：data/chunks 40,609——text 1.94 万 / table 2.12 万 / title 0——标题已并入后续 text 前缀；平均 482 字符/块，p50=525 / max=1522。**collection 现存 40,192 点，与产物差 417**（2026-09-19 重切块后残留，以 collection 为运行口径，如需对齐可 `--recreate --skip-encode` 重建））。**表格块占 52.3%**，表格解析质量是评测可信度的关键前提（见 §9） |
| 目标（当前版） | ① 跑通全链路（下载→解析→切块→索引→检索→生成→UI），交付**可用**的 RAG 系统；② 以 50 道合成题 × RAGAS 四指标产出 E3（完整混合链路）的初步质量报告 |
| 非目标 | E0~E4 对照实验（降级为 §10 演进）、查询改写/HyDE、多轮对话、GraphRAG、生产化部署、人工金标准标注（pooling/nDCG）。（"增量入库"v1.6 起从非目标移除——服务端 IDF 口径下已具备） |
| 成功标准 | M3 里程碑：检索链路端到端可用（UI 能问、答案带引用溯源）；M4 里程碑：RAGAS 四指标初步报告（ContextRecall/ContextPrecision/Faithfulness/AnswerRelevancy）+ 最差 10 题归因 |
| 目标权重 | **能用优先**，评测其次，对照实验后置 |

## 2. 系统架构与流水线

### 2.1 系统架构（分层）

```
┌─ 交互层 ── ui.py（Gradio 7860：提问 / 实验组切换 / 元数据过滤 / 引用溯源面板）
├─ 检索层 ── retrieve.py（Retriever：E0~E4 实验组路由 / 客户端 RRF / 云端重排，失败硬抛）
│            llm_client.py（Prompt 组装 + GLM 生成，降级不断链）
├─ 索引层 ── download_pdf.py（巨潮接口下载）→ parse_pdf.py（PyMuPDF block 三分类解析）
│            chunk_documents.py（SentenceSplitter 768/80 分治切块）→ index_documents.py
│            （BGE-M3 编码 + 三命名向量入库；公共口径函数 tokenize / term_dim / get_qdrant_client）
├─ 评测层 ── evaluate.py（gen 分层出题 / run E3 断点续跑跑批 / score 五指标 + 裁判报告）
├─ 存储层 ── Qdrant 服务端容器（collection `annual_reports`：dense / bm25 / bge_sparse
│            三命名向量 + payload 11 字段，40,192 点）
│            data/*（raw PDF / parsed / chunks / embeddings npz 编码缓存 / eval 评测产物）
├─ 基础层 ── config/config.yaml（qdrant / embedding / sparse / retrieval / rerank / llm /
│            judge / eval 八段）+ config/.env（key 兜底）+ check_env.py（环境自检）
└─ 外部服务 ─ 智谱 GLM（coding plan：生成 + 出题）／ DeepSeek 官网（评测裁判）／
             硅基流动（rerank）／ 巨潮资讯网（年报语料源）
```

> 分层说明：检索层与索引层**不共享运行时**（离线一次/增量 vs 在线每次提问），仅通过 index_documents 的口径函数与 Qdrant collection 衔接；评测层是检索层的消费者（复用 `retrieve.Retriever`），不独立建检索实现。

### 2.2 流水线（三条链路）

**① 离线索引链路**（一次性 + 增量；`src/` 前四个模块顺序执行，各自可断点续跑）：

```
巨潮接口 -> download_pdf.py（多行业采样 60 份 PDF + manifest）
        -> parse_pdf.py（PyMuPDF：block 三分类 title/text/table + section_path 标题栈，人工抽查）
        -> chunk_documents.py（切块 768/80：title 并入前缀 / text 走 splitter / table 行级带表头
                               + 元数据注入：公司/代码/年份/章节/页码/块类型）
        -> index_documents.py 阶段A（BGE-M3 一次 encode 同时产出 dense + learned sparse，
                                      落盘 npz 缓存，重跑免 GPU）
        -> index_documents.py 阶段B（jieba 分词 + tf 饱和/长度归一值（idf 交服务端 Modifier.IDF，
                                      v1.6 无全库统计环节，天然支持增量）+ payload 11 字段
                                      -> Qdrant upsert，point id = uuid5(chunk_id) 幂等重入）
```

**② 在线检索链路**（每次提问；E3 主线，含重排）：

```
ui.py 提问 -> retrieve._encode（BGE-M3 编码 dense + jieba 分词 bm25 查询向量，命中词 1.0）
           -> Qdrant 双路召回（各 top50；payload filter 可同时下推两路，ED7）
           -> 客户端 RRF 融合（Σ 1/(60+rank)，保留分路排名明细）
           -> 前 25 条送硅基流动 /v1/rerank（bge-reranker-v2-m3；失败硬抛不降级）
           -> top5 -> llm_client 组装 Prompt（chunk 正文 + 编号引用元数据）
           -> 智谱 GLM 生成（约束引用出处）-> ui.py 展示（答案 / 命中列表各路分数 / 阶段计时）
```

**③ 评测支线**（M4，独立于主线；详见 §7）：

```
data/chunks 分层抽样（seed=42，公司×年份×块类型）-> evaluate.py gen（GLM 出题 50 道：
        术语 18 / 改写 18 / 数字 14）-> 质量闸（人工核 10 题含 5 表格题）
        -> evaluate.py run（E3 跑批：contexts + answer + 延迟，断点续跑）
        -> evaluate.py score（自写五检索指标 Recall@5/P@5/HitRate@5/MRR/NDCG@5，
           金标准=源 chunk 唯一命中 + DeepSeek 裁判答案侧参考分 -> 分题型分层 + 最差 10 题归因）
        -> data/eval/report.md
```

**检索路与实验组（v1.5：对照降级，能力保留）**：当前主线只跑 **E3 完整混合链路**（dense + bm25 双路 + RRF + rerank）。E0~E4 对照实验矩阵降级为 §10 演进路线——但三命名向量（dense/bm25/bge_sparse）全部照存，客户端 RRF 实现保留分路排名明细，将来恢复对照零改造成本（`retrieve.py --exp E0/E1/E2/E4` 多跑几遍即可）。

### 2.3 一次查询的完整生命周期

1. 用户在 Gradio 输入问题（如"贵州茅台 2023 年毛利率多少"），可选实验组（默认 E3）/ 股票代码 / 年份区间
2. 检索模块构造双路查询向量：dense（BGE-M3 编码查询）+ bm25（jieba search 分词 → 每命中词权重 1.0，idf 由服务端 Modifier.IDF 打分时乘——v1.6；旧"查 idf 表算查询权重"口径废止）
3. Qdrant 服务端并行执行两路检索（各 top 50）；**RRF 融合为客户端实现**（M3 定版：偏离原服务端 FusionQuery 方案，公式等价，为分路归因与恢复对照保留明细）
4. 融合序前 25 条送硅基流动 rerank API 重排（**默认仅 E3 走重排**，E0/E1/E2/E4 对照组不重排；失败硬抛不降级）
5. 重排后 top 5 chunk 正文与元数据（公司/年份/章节/页码）组装进 Prompt
6. 智谱 GLM（glm-5.3-flash）生成答案，Prompt 约束必须引用出处
7. Gradio 展示：答案面板 + 检索命中列表（含各路分数）+ 引用溯源面板 + 阶段耗时（encode/召回/RRF/rerank/llm）

> 注：元数据过滤（如"只在某公司某年份查"）由 payload filter 在 step 3 两路上同时下推，与 RRF 融合正交可叠加（ED7）。

## 3. 各环节技术选型

| 环节 | 选型 | 核心理由 | 备选/降级路径 |
|---|---|---|---|
| 数据获取 | 巨潮 `hisAnnouncement/query` 接口，按股票代码 + 年报类别（`category_ndbg_szsh`）+ 时间检索 | 官方接口、返回 PDF 直链、可批量 | 接口变更则手工下载，PDF 本地化后依赖即解除 |
| 采样策略 | 多行业分散：实测落地 **20 家 × 3 年 = 60 份**（金融/家电/白酒/医药/汽车/能源/科技等） | 行业术语密度差异是验证 BM25 价值的天然实验变量；同年多公司支持跨公司对比型查询 | — |
| PDF 解析 | **PyMuPDF 先行** | 轻量、快；先验证正文质量 | **Gate**：M1 抽查财务表格不过关 → 升级 MinerU/Marker（版面还原，保表格） |
| 切块 | LlamaIndex `SentenceSplitter(**768/80**)` + 块类型分治 | 框架执行、策略自定、可复现；title 整块不切、table 按 raw_table 行级切分（每段带表头，单行不拦腰截断）、text 走 splitter | 结构感知切分（按年报章节）、small-to-big 为后续实验变量；**切分粒度对照降级为演进**（见 §10） |
| 元数据 | 见 §3.1 schema | 过滤检索 + 引用溯源的刚需 | — |
| Embedding | **BGE-M3 本地**（FP32，见 §6 GPU 现状） | dense+sparse+ColBERT 三表示；1024 维；8192 上下文；**一次 `encode(return_dense=True, return_sparse=True)` 同时产出 dense 与 learned sparse**，故存 bge_sparse 零额外前向成本 | **降级开关**：硅基流动云 API 同模型（免费，代码内预留切换） |
| 稀疏表示 | **双稀疏并存**：① `bm25` = jieba search 分词 + 客户端 tf 饱和/长度归一（k1=1.5/b=0.75，avgdl 冻结 248.54）+ **服务端 Modifier.IDF**（v1.6：df/idf 随点增删自动维护，增量入库零全局重算；旧"两遍全库统计 + idf 表持久化"口径废止）；② `bge_sparse` = BGE-M3 learned sparse（一次 encode 白给，不挂 modifier）。自写 {indices, values} → Qdrant `SparseVector` | 零训练、可解释、与 dense 互补；中文必须 jieba 分词——**不用** LlamaIndex 内置 fastembed `Qdrant/bm25`（英文向，中文效果差）；BM25 分工（v1.6）：**idf 交服务端**（Qdrant ≥1.10 支持，语料一变即自动更新），**k1/b 长度归一留客户端**（服务端仍不支持这两个参数，v1.3"权重必须全客户端算好"中仅 idf 部分被推翻） | 分词模式已定版 jieba_search（G2）；E4 对比 learned sparse 时只换查询的 vector name 不重建索引，**但 v1.6 后两路 modifier 不对称，对照需计入 idf 差异（ED9）** |
| 向量库 | **Qdrant 服务端**（Docker 容器 qdrant/qdrant，`QdrantClient(host=..., port=...)`），**qdrant-client 直连**（不经 LlamaIndex 封装） | v1.4 修订：本机 Docker Desktop 就绪后由本地模式切服务端，多命名稀疏向量 + hybrid + RRF 能力不变，解除单进程锁与点数软上限约束 | 数据随容器卷持久化；若弃用 Docker → qdrant.exe 单二进制（免 Docker），只改 config.yaml qdrant 段 |
| 融合 | RRF（k=60），Qdrant 服务端 `prefetch + FusionQuery` | 免调参、鲁棒、绕开量纲问题 | 加权归一化否决（见 ED2）；k 不调（标注查询 <200 条） |
| 重排 | **bge-reranker-v2-m3 云 API**（硅基流动，免费） | 避开 4~6G 显存双模型 OOM；cross-encoder 典型提升 Precision@5 10~30% | 本地 FP16（约 2G，显存富余时） |
| 生成 | **智谱 glm-5.3-flash（v1.8 切换）**，OpenAI 兼容层（coding plan 端点） | 原 token plan qwen3.8-flash 403 未购权益，切换后联调通过；配置化可随时切回百炼/DeepSeek 等 | — |
| 交互 | Gradio Web UI | 几十行出完整界面，适合展示对照结果 | — |
| 编排框架 | **LlamaIndex（职责收窄至切块）**；**索引入库与检索改用 qdrant-client 直连**（v1.3 修订，理由见 ED6） | 切块沿用 SentenceSplitter 成熟能力；但索引侧需要"1 dense + 2 sparse 三命名向量"、查询侧需要按 vector name 指定走哪路 + prefetch 级 filter 下推，LlamaIndex 的 Qdrant 集成假设"1 dense + 1 sparse"表达不了，实验组（E0~E4）差异恰在这些层面 | LangChain（否决理由：优势在 Agent/工具链，本项目用不上；对照实验常与框架默认值打架） |

### 3.1 元数据 Schema

v1.3 对齐 chunk 产物实际字段（`data/chunks/*.json` 的 `metadata` + `chunk_id`），入库时映射为 Qdrant payload：

| payload 字段 | 类型 | 来源 | 用途 |
|---|---|---|---|
| `company_name` | str | chunk 文件名 `<code>_<year>_<公司名>_chunk.json` 推导 | 按公司过滤 |
| `stock_code` | str | `metadata.stock_code` | 同上（代码级精确匹配） |
| `fiscal_year` | int | `metadata.year`（产物中为字符串，入库转 int） | 按年份过滤（跨年对比的前提） |
| `section` | str | `metadata.section`（section_path 以 " > " 拼接） | 按章节过滤 + 引用溯源 |
| `page_num` | int | `metadata.page_num` | 引用溯源（答案标注出处页码） |
| `chunk_idx` | int | `chunk_id` 末段序号（`<code>_<year>_<5位序号>`） | 同文档内块序号（引用去重、增量入库去重键） |
| `chunk_type` | str | `metadata.block_types[0]`：`title` / `text` / `table` | 块类型（表格块单独标记，检索时可区分处理） |
| `chunk_id` | str | chunk 产物原值 | 全局唯一 ID → UUID5 生成 Qdrant point id（幂等重入） |
| `source_file` | str | `metadata.source_file` | 溯源到 PDF 原件 |
| `is_ocr` | bool | `metadata.is_ocr`（本项目恒 false） | 质检标记 |
| `text` | str | chunk 的 `content` | **M3 组装 Prompt 依赖**（正文必须随向量入库，不能只存元数据） |

生成方式：从下载清单（公司、代码、年份）自动生成 + 解析时从标题栈提取章节 + 切块时记录页码，**不手工填**。兜底：无标题时 `section` 置 `"unknown"`，不阻塞入库。

> 注：产物中 `block_types` 是列表（当前每 chunk 恒单元素），payload 取首元素存为 `chunk_type` 字符串，便于 Qdrant keyword 过滤。

## 4. 关键工程决策记录

| # | 决策 | 理由 | 代价/风险 |
|---|---|---|---|
| ED1 | 混合检索（dense + BM25） | 互补矩阵：dense 强在语义泛化/口语化提问，BM25 强在精确术语/编号/长尾词——年报场景（公司名、财务科目、数字指标）正是 BM25 高价值区 | 索引与查询成本 ×2（两路表示）；**v1.3 补充**：实际存 3 个命名向量（见 ED9），存储成本约 ×2.3，但查询成本仍 ×2（检索只走两路） |
| ED2 | RRF 融合，不做加权归一化 | BM25 分数无上界依赖语料统计，余弦在 [-1,1] 且相关文档聚集 0.7+，两分布无法直接归一化；RRF 只看排名彻底绕开 | 丢失分数差距信息（可接受） |
| ED3 | Qdrant 本地模式起步 → **（v1.4 修订）Qdrant 服务端（Docker 容器）** | 原决策依据"本机实测无 Docker"；2026-09-17 本机装好 Docker Desktop 并部署 qdrant 容器（1.19.1），前提不再成立，切换只动连接层（`get_qdrant_client()` 统一建连，检索算法/向量口径零变化） | 新增运维前置：用任何索引/检索脚本前容器必须在跑；历史 `data/qdrant_storage/` 保留为本地模式时代的备份，代码不再读写 |
| ED4 | Reranker 走云 API | BGE-M3(~2G) + reranker(~2G FP16，实测上界) + CUDA 开销 ≈ 5G，4~6G 显存同跑必 OOM；硅基流动免费且延迟 100~300ms 可接受 | 依赖网络与第三方可用性 |
| ED5 | Embedding 按本地写 | 索引阶段一次性批处理，显存够用；**GPU 已实测可用**（2026-09-15：GTX 1650，torch 2.10.0+cu130，原 NVML Unknown Error 已解除）；fp16 实测慢 4.7 倍故固定 fp32 | GPU 故障时需切云（代码预留开关，同模型零迁移成本）；全量编码已在 M2b 完成（4.06 万 chunk × 3 表示，实测耗时见 §8 M2b 状态） |
| ED6 | **（v1.3 修订）** 解析自控 + 切块参数化 + **索引入库 qdrant-client 直连** | 解析质量决定上限（年报表格是最大坑）需人工把关；切分策略是实验变量需自定；**原"向量化框架全包"的前提已被推翻**——LlamaIndex 的 Qdrant 集成假设"1 dense + 1 sparse"，表达不了本项目"1 dense + 2 sparse 三命名向量"，查询侧也无法按 vector name 指定走哪路、无法在 prefetch 级下推 filter；而 E0~E4 实验组的差异恰恰全在这些层面，硬套框架等于到处打洞绕过它 | 解析环节占 M1 一半工时；索引/检索代码需自己写 payload 组装与查询构造（换来的是与 Qdrant 官方 hybrid 文档一一对应、实验矩阵完全可控） |
| ED7 | 元数据是第二层精度来源 | payload filter（公司/年份/章节）与 RRF 融合正交，可叠加使用 | schema 设计需前置（M2 定稿） |
| ED8 | 评测定版：LLM 合成 Q&A + RAGAS（**v1.5 修订**） | 冷启动无标注的最佳实践。**v1.5 变更**：不用 RAGAS TestsetGenerator（合成模块坑多），改自写轻量生成器出题；只用 RAGAS 指标模块算 ContextRecall/ContextPrecision/Faithfulness/AnswerRelevancy；不算检索侧 Recall@K/nDCG（需人工金标准，已放弃）；单列 E3 无对照（见 §7） | 合成质量依赖解析质量（表格乱则答案错）；裁判 v1.5 走 DeepSeek 异家族 + 10 题人工复核；Ragas 0.2→0.3 API 大改需锁版本 |
| ED9 | **（v1.3 新增，v1.5 保留）** 双稀疏并存：`bm25` + `bge_sparse` 同时入库，主线检索只用 `bm25` | BGE-M3 的 learned sparse 与 dense 出自**同一次前向传播**（`encode(return_dense=True, return_sparse=True)`），存第三份向量零额外编码成本、存储增量仅约 5%（稀疏向量每 chunk 几十个非零维度）；换来 E4 实验**不需要重建索引**——只把查询的 vector name 从 `bm25` 换成 `bge_sparse`。**v1.5：E4 降级为 §10 演进路线，但 bge_sparse 向量照存**——将来恢复对照零成本，此即 ED9 的直接收益 | collection schema 多一个命名向量；若将来完全不做 E4，这部分存储是沉没成本（可接受，~5% 增量）。**v1.6 注**：`bm25` 路已挂服务端 Modifier.IDF、`bge_sparse` 不挂，两稀疏路不再同构——恢复 E4 时"idf 加权有无"是对照变量之一，需单独评估，或给 bge_sparse 同挂 modifier 后一并重建再比 |

## 5. 目录结构

**v1.3 定版：`src/` 下扁平化——每个模块一个 `.py` 文件，不建子目录。** 原设计按模块分子目录（`src/parse/`、`src/index/`…），但本项目每模块职责单一、单文件足够容纳（300~600 行），子目录只会增加 import 路径长度与 `sys.path` 处理成本。

```
混合检索RAG/
├── docs/                      # 文档（architecture.md 活跃；选型对比 v1/v2 留档）
├── src/
│   ├── download_pdf.py        # 巨潮下载：接口调用、标题甄别、限频重试、manifest（原 download/ 合并）
│   ├── parse_pdf.py           # ✅已落地 PDF 解析：PyMuPDF 抽取、block 三分类、section_path 标题栈
│   ├── chunk_documents.py     # ✅已落地 切块：SentenceSplitter、表格行级切分、元数据注入
│   ├── index_documents.py     # ✅已落地 向量化+入库：BGE-M3 dense & learned sparse 编码、
│   │                          #   jieba 分词+tf 饱和值（idf 交服务端 Modifier.IDF，v1.6）、
│   │                          #   Qdrant 建表、PointStruct 组装、npz 缓存、CLI
│   │                          #   （--codes 增量入库 / --replace 整份替换 / --recreate）
│   ├── retrieve.py            # ✅已落地 M3 检索：E0~E4 实验组（EXP_ROUTES 表驱动）、
│   │                          #   payload 过滤下推、客户端 RRF、云端重排（失败硬抛）、
│   │                          #   last_timings_ms 阶段计时、build_filter、_load_dotenv
│   ├── evaluate.py            # ✅已落地 M4 评测：gen/run/score 三子命令、分层抽样出题、
│   │                          #   E3 断点续跑跑批、自写五检索指标+DeepSeek 裁判参考分、
│   │                          #   分题型分层报告+最差 10 题归因（小样本验证过，待 50 题正式跑批）
│   ├── ui.py                  # ✅已落地 M3 Gradio 界面：提问、实验组切换、过滤、引用面板
│   └── check_env.py           # ✅已落地 环境自检（M0 验收项）
├── data/
│   ├── raw/                   # 年报 PDF 原件（60 份，含 manifest.json）
│   ├── parsed/                # ✅解析产物：<code>_<year>_<公司>.json
│   ├── chunks/                # ✅切块产物：<code>_<year>_<公司>_chunk.json
│   ├── chunk_debug/           # 切块调试输出（SentenceSplitter node 全属性）
│   ├── embeddings/            # ⏳编码缓存：<code>_<year>.npz（dense + learned sparse）
│   └── bm25_stats/            # 🗄 v1.6 停用：历史全库 idf 表（avgdl_frozen=248.54 的出处），代码不再读写
    #   注：v1.4 起向量数据存于 Qdrant 容器卷，代码不再维护 data/qdrant_storage/；
    #       该目录若存在仅为本地模式时代的历史备份，不参与运行
├── model/
│   └── BGE-M3/                # 本地模型权重（开源时用户自行下载放置）
├── config/                    # config.yaml + .env（key 引用，不入库）
└── requirements.txt
```

**约定（沿用既有代码风格）**：各文件自行以 `ROOT = Path(__file__).resolve().parents[1]` 定位项目根并解析相对路径，不额外抽 `config.py`；`config.yaml` 由各模块按需 `yaml.safe_load`。

**导入顺序硬约束**：`FlagEmbedding` 必须先于 `jieba` 导入，否则 Windows 上触发 DLL segfault（`index_documents.py` 顶部固化）。

**编码缓存与入库分离**（在单文件内以缓存目录实现）：dense/sparse 编码结果落盘 `data/embeddings/*.npz`，改 payload schema 或调试入库时可跳过 GPU 重编码（4.06 万 chunk 全量编码已完成，缓存避免重跑 GPU）。

## 6. 配置与环境

**Python**：miniconda py312 环境（本机既有 conda 环境名 `py312`，Python 3.12.7；开源仓库不记录本机安装路径）。

**依赖清单**：`llama-index`、`llama-index-vector-stores-qdrant`、`qdrant-client`、`pymupdf`、`jieba`、`gradio`、`FlagEmbedding`（BGE-M3 本地推理）、`openai`（智谱/DeepSeek/硅基流动统一 OpenAI 兼容层）、`requests`、`python-dotenv`。（v1.8：ragas/datasets 已从清单移除——评测改自写指标，零第三方评测依赖）
**依赖注意**：① fastembed 仅在使用 LlamaIndex 内置 BM25（`Qdrant/bm25`）时需要，本项目自写 jieba 映射，不强依赖（若 `llama-index-vector-stores-qdrant` 传递引入则随包安装即可）；② v1.8 弃 RAGAS——实装 0.4.3（最新）import 即挂（依赖被 langchain-community 0.4.2 移除的 `chat_models.vertexai`），降级 0.3.9 需连带降 langchain 全家桶（core 1.6.2/openai 1.6.1 → 0.3.x）有波及 gradio 6.30 风险，定版自写判分；将来恢复 RAGAS 待其官方适配后再评估。

**API key（三个，分开管理）**：

| 平台 | 用途 | 端点 |
|---|---|---|
| 智谱 GLM（coding plan） | LLM 生成 + 出题（glm-5.3-flash） | OpenAI 兼容 `open.bigmodel.cn/api/coding/paas/v4` |
| DeepSeek 官网 | 评测裁判（deepseek-flash） | OpenAI 兼容 `api.deepseek.com/v1` |
| 硅基流动 | rerank（bge-reranker-v2-m3，免费） | `/v1/rerank` |

**config.yaml 结构示意**（数值为当前默认取值；§2.2 流水线正文中的 top 数值为可调范围，M3 联调后已定版）：

```yaml
paths:
  raw: data/raw
  parsed: data/parsed
  chunks: data/chunks          # v1.3 新增：切块产物目录（M2 实际产物位置）
  # v1.6：bm25_stats 项移除——df/idf 统计改由 Qdrant 服务端 Modifier.IDF 维护

qdrant:                    # v1.4：服务端模式取代原 paths.qdrant 本地目录
  host: 127.0.0.1          # Docker 容器端口映射到本机
  port: 6333               # REST 端口（gRPC 6334 暂不使用）
  api_key: ""              # 容器未设 QDRANT__SERVICE__API_KEY 则留空

embedding:
  provider: local          # local | siliconflow（降级开关）
  model: model/BGE-M3      # 本地模型目录（相对项目根）
  hf_model_id: BAAI/bge-m3 # 备用 HF id（本地目录缺失时经 HF_ENDPOINT 镜像下载）
  batch_size: 16           # OOM 则调小（GTX 1650 4G 显存偏紧）
  device: cuda             # cuda | cpu
  use_fp16: false          # 实测 GTX 1650（TU117 无 Tensor Core）fp16 反而慢 4.7 倍

sparse:
  method: bm25             # bm25 | bge_m3_sparse（E4 后定版；两者均入库，见 ED9）
  tokenizer: jieba_search  # jieba_search | jieba_exact（G2 已定版 search）
  k1: 1.5
  b: 0.75
  avgdl_frozen: 248.54     # v1.6 新增：BM25 的 idf 交服务端，avgdl 冻结为常量（实测出处）

retrieval:
  top_k_per_route: 50      # 两路各召回
  rrf_k: 60                # 不调（<200 标注查询）
  rerank_top_n: 25         # 送重排数量
  final_top_n: 5           # 进 Prompt 数量

rerank:
  provider: siliconflow
  model: BAAI/bge-reranker-v2-m3

llm:
  provider: zhipu          # OpenAI 兼容，可切 bailian/deepseek 等（v1.8 切智谱）
  model: glm-5.3-flash

judge:                     # M4 评测裁判（v1.8 新增，与生成侧异家族）
  provider: deepseek
  model: deepseek-flash

chunking:
  chunk_size: 768          # v1.3 定版：与实际切块产物一致（原 512）
  chunk_overlap: 80        # v1.3 定版（原 50）
  table_max_chars: 800     # 表格按字符截断（非 token），每段带表头
```

**GPU 现状（v1.3 更新）**：2026-09-15 实测 `torch.cuda.is_available() == True`，设备 `NVIDIA GeForce GTX 1650`，torch 2.10.0+cu130。原 NVML Unknown Error 已解除。实测环境版本：FlagEmbedding 1.4.2 / qdrant-client 1.19.0 / jieba 0.42.1 / llama-index-core 0.14.24 / numpy 1.26.4。

**fp16 结论（实测翻车，已定版关闭）**：GTX 1650 的 TU117 核心无 Tensor Core，fp16 走软件模拟，编码 8.4min vs fp32 2.6min（慢 4.7 倍），故 `use_fp16: false` 固定。

**导入顺序硬约束**：`FlagEmbedding` 必须先于 `jieba` 导入，否则触发 DLL segfault（已在 `check_env.py` 固化约定）。

**Qdrant 服务（v1.4）**：Docker Desktop 中的 `qdrant` 容器（镜像 qdrant/qdrant 1.19.1，REST 端口映射 127.0.0.1:6333）。跑索引/检索/评测/UI 前须确认容器在运行（`docker ps` 或 `python src/check_env.py` 的服务探活项）。数据持久化在容器卷（D:\Program Files\Qdrant\data），删容器重建卷不丢数据；`docker start qdrant` 即可恢复服务。

## 7. 评测体系（v1.8 定版：单列 E3 × 自写五检索指标 + 裁判参考分）

> v1.5 曾定版 RAGAS 四指标方案；v1.8（2026-10-25）用户定版改自写指标——起因：实装 ragas 0.4.3（最新版）import 即挂（其依赖的 `langchain_community.chat_models.vertexai` 已被 langchain-community 0.4.2 删除），降级 ragas 0.3.9 需连带降 langchain 全家桶、有波及 gradio 6.30 的连锁风险，走架构 §9 预案"自写判分"。本节描述当前主线：对 **E3 完整混合链路**跑 50 道合成题，算五检索指标 + 裁判参考分，产出报告。**不做人工金标准标注、不做对照实验**（对照恢复见 §10）。

### 7.1 指标定义与金标准口径（v1.8）

金标准 = **出题源 chunk（唯一命中口径，2026-10-25 用户定版）**——top5 中只有源 chunk 本身算命中，邻块/同章节块不算（严格口径，指标会低估但可复现、零标注成本）：

| 指标 | 定义 | 说明 |
|---|---|---|
| Recall@5 | top5 含源 chunk 的题目比例 | 主指标 |
| Precision@5 | 命中数/5 | 单金标下限 0、上限 0.2（解读注意） |
| HitRate@5 | 同 Recall@5 | 单金标时数值等价，并列呈现 |
| MRR | 1/首次命中排名（未命中 0） | 位置敏感主指标 |
| NDCG@5 | 1/log2(rank+1)（IDCG=1，单金标折减） | 位置折减 |
| 裁判评分（参考） | DeepSeek 5 分制评 answer vs reference_answer | 答案侧参考，不进主表，不依赖金标准 |

### 7.2 评测流程（src/evaluate.py 单文件，三个子命令）

**① `gen`——出题（自写轻量生成器，glm-5.3-flash）**
分层抽样源块（公司×年份×块类型，seed=42 可复现，每份年报≤2 题保多样性，长度 80~900 字符），GLM 逐块按题型出题。题型配额锁死：术语精确 18 / **语义改写 18**（prompt 明令换措辞）/ 数字提取 14（源块优先 table）。产物 `data/eval/eval_set.json`（含 question/reference_answer/source_chunk_id/source_meta）。

**② 质量闸（人工，1 小时）**
人工核 10 题（含 5 道表格题）。表格错误集中则数字提取型整体剔除并在报告声明。

**③ `run`——E3 跑批（断点续跑）**
复用 `retrieve.Retriever.retrieve(exp="E3")`（含重排，重排失败硬抛中断整批——即"不许带病入表"），GLM 生成失败不阻塞（answer_degraded 记录，检索指标不受影响）。每题存 contexts(top5 明细)/answer/retrieve_ms/timings_ms → `data/eval/run_results.json`。

**④ `score`——五指标 + 报告**
总体 + 分题型分层（term/paraphrase/numeric）+ 最差 10 题（MRR 升序）归因素材 + DeepSeek 裁判答案侧参考分 + P50/P95 延迟 → `data/eval/report.md`。

### 7.3 指标解读的诚实声明

- **绝对值会偏乐观**：合成题天然贴近来源 chunk，指标度量的是"检索找回出题素材的能力"，不是"检索找回所有真正能支持答案的证据的能力"。这是免标注方案的固有代价，接受它、不假装无偏。
- **分层差分才有信息量**：改写型 vs 术语型的 Recall@5 差距，反映检索系统对表述变化的鲁棒性，比总平均更有价值。
- **结论边界声明**（写进报告）：绝对分数为合成题近似口径、偏乐观；本评测在无元数据过滤的裸查询下测量；单金标口径下 Precision@5 上限 0.2、HitRate@5 与 Recall@5 等价。

### 7.4 成本与依赖

- LLM 调用：出题 50 次（GLM）+ 生成 50 次（GLM）+ 裁判 ~50 次（DeepSeek，生成降级的题跳过）≈ 150 次小 prompt 调用；GPU 零新增（检索 encode 复用本地 BGE-M3）。
- 依赖：**无 ragas/langchain**（v1.8 弃用），全部自写；LLM 走 OpenAI SDK（llm/judge 配置段）。
- 裁判与生成异家族：生成 GLM（智谱）vs 裁判 DeepSeek（官网），破同族闭环。

## 8. 里程碑与验收

> v1.5：原 M4（首组对照）与 M5（评测定版）合并为单一 M4（系统评测）；对照实验降级为 §10 演进路线。

| 阶段 | 内容 | 产出/验收 | Gate 决策点 | 状态 |
|---|---|---|---|---|
| M0 环境（半天） | miniconda py312 环境 + 依赖 + key 连通验证 + GPU 实测 | requirements.txt、config.yaml、`src/check_env.py` 通过 | GPU 可用性记录（不阻塞） | ✅ 完成（GPU 实测可用，GTX 1650） |
| M1 下载+解析（2 天） | 接口实测 → 批量下载 → PyMuPDF 解析 → 人工抽查 | `data/raw`（60 份 + manifest.json）+ `data/parsed` + `src/parse_pdf.py` | **G1**：表格解析质量（v1.5 由 M4 质量闸补一轮抽查，见 §7.2②） | ✅ 完成（block 三分类 + section_path 标题栈 + raw_table 保留） |
| M2a 切块（1 天） | 块类型分治切块 + 元数据注入 | `data/chunks`（60 份 / **4.06 万 chunk**，v1.3 修复后：连续同 section text 聚合再切 + title 并入前缀，粒度问题已解决）+ `src/chunk_documents.py` | **G2**：jieba search/exact 定版 → **已定版 search** | ✅ 完成（768/80，实测 p50=525/max=1522 字符，chunk_size 生效） |
| M2b 入库（1~2 天） | BGE-M3 三命名向量编码 + BM25 值入库 + Qdrant 直连入库（**先 5 份小样本跑通再全量**；v1.6 起 BM25 环节 = 客户端 tf 饱和 + 服务端 Modifier.IDF，原全库两遍统计废止） | Qdrant collection `annual_reports`（**40,609 点**，3 命名向量 + payload 11 字段可过滤）+ `src/index_documents.py` + 入库质检报告 | — | ✅ 完成（历史统计留档：全库 16.76 万词/avgdl 248.5/idf 4.7MB——248.5 即 avgdl_frozen 出处；QC 三路冒烟全中。v1.3 曾在本地模式超 2 万点软告警；v1.4 已迁服务端容器重建 collection；v1.6 迁移用 `--recreate --skip-encode` 一次性重建） |
| M3 检索链路（1~2 天） | hybrid + RRF + rerank + GLM + Gradio（**主线只要求 E3 可用**） | 可演示 UI，答案带出处（`src/retrieve.py`、`src/ui.py`） | — | ✅ 完成（2026-10-17 联调收尾。E0~E4 五组 + payload 过滤实测通过；**RRF 定为客户端实现**——偏离原 §2 服务端 FusionQuery 方案，为分路归因与将来恢复对照，公式等价；重排默认仅 E3 走、失败硬抛不降级（定版）；rerank 实调 200 OK；gradio 6.30 UI 启动探活 200。2026-10-25 LLM 切智谱 glm-5.3-flash 后生成链路联调通过，403 遗留解除） |
| M4 系统评测（2 天） | 50 合成题（自写生成器）→ 质量闸人工核 10 题 → E3 单列跑批 → **自写五检索指标（Recall@5/Precision@5/HitRate@5/MRR/NDCG@5，v1.8 用户定版，弃 RAGAS）+ DeepSeek 裁判答案侧参考分** → 分层报告 + 最差 10 题归因 | `data/eval/eval_set.json` + `data/eval/run_results.json` + `data/eval/report.md` + `src/evaluate.py` | 质量闸：表格题错误集中 → 剔除数字提取型并声明 | 🔶 **正式跑批完成（2026-10-25）**：50 题（18/18/14 精确配额，弃块 2），Recall@5=0.58 / MRR=0.433 / NDCG@5=0.470，裁判 3.72；**分题型差分**：numeric 0.857 > term 0.722 ≫ **paraphrase 0.222**（改写题召回塌方——对措辞变化鲁棒性是当前最大短板，恰好是恢复 E0~E4 对照的最强理由，见 §10）；**待做**：质量闸人工抽核 10 题（含 5 表格题） |

## 9. 风险清单（v1.5 清理）

> v1.5 清理：已消解风险（M2a 切块修复、v1.4 服务端迁移、对照实验降级）移除；保留项更新为当前状态；新增 v1.5 范围调整引入的风险。

| 风险 | 等级 | 预案 |
|---|---|---|
| **年报财务表格解析质量差**：表格块占索引 **52.3%**（2.12 万/4.06 万），G1 gate 从未系统抽查（v1.5 确认）；行列错位会同时污染检索、生成、评测三端 | **高** | M4 质量闸补抽查（10 题含 5 表格题，见 §7.2②）；错误集中 → 数字提取型剔除并在报告声明；MinerU/Marker 升级路径预留（仅 table chunk 重灌，dense 路不受影响） |
| **合成评测集被解析错误污染**（表格乱 → 错误答案看起来合法） | **高** | 合成输入用 M1 gate 后的解析产物；表格型 Q&A 在 M4 质量闸单独抽核 |
| **ContextRecall/ContextPrecision 偏乐观**（合成题贴近来源 chunk，度量的是"找回出题素材的能力"而非"找回真证据的能力"） | **中** | §7.3 诚实声明 + 分层报告（改写型 vs 术语型差分才有信息量）；Faithfulness/AnswerRelevancy 不依赖金标准、相对可信，作主参考 |
| **单列 E3 无对照 → 增益不可归因**（v1.5 范围调整的代价） | **中**（已接受） | 接受：当前目标是"能用 + 初步评测"，不是"验证混合增益"；三命名向量与客户端 RRF 照存，将来恢复 E0~E4 对照零改造（§10） |
| **Rerank 失败中断检索**（v1.7 口径）：`retrieve.py` rerank 失败**直接抛 RuntimeError、不重试不做 RRF 降级**（2026-09-21 用户定版，防带病结果；UI 层捕获展示错误，同样不回退）——与生成侧"降级不断链"刻意不对称：重排是检索质量的一部分，生成只是展示层 | 已消解 | evaluate.py 跑批沿用此语义：异常即中断不写入结果，天然满足"不许带病入表" |
| **裁判同族闭环**（v1.8 已消解）：生成侧智谱 GLM + 裁判 DeepSeek 官网，异族 | 已消解 | 裁判走 config.judge 段（deepseek-flash）；出题与生成同用 GLM（题面风格与生成同源，裁判仍异族） |
| **ragas 版本冲突**（v1.8 已消解）：ragas 0.4.3 import 即挂（依赖被 langchain-community 0.4.2 移除的模块），降级需动 langchain 全家桶 | 已消解 | v1.8 定版弃 RAGAS，自写五检索指标 + 裁判参考分（evaluate.py），零第三方评测依赖；将来若恢复 RAGAS，待其官方适配 langchain 0.4 再评估 |
| **服务端 IDF 删点滞后**（v1.6 引入）：`--replace`/删点后相关词 df 仍按已删点计，segment 合并优化前 idf 偏低（qdrant/qdrant#6690、#6735） | 低 | 千级替换 / 4 万库占比小，RRF 只看名次基本无感；optimizer 合并自愈，无需人工；大规模重灌后介意则 `--recreate --skip-encode` 重建一次 |
| GPU 异常，本地 embedding 不可用 | 中 | `provider` 一键切硅基流动（免费同模型） |
| 巨潮接口限频/结构变更 | 低（降级） | PDF 已本地化（60 份），下载链路依赖已解除；仅影响将来语料扩展 |
| GLM/DeepSeek/硅基流动额度与限频（M4 评测约 150 次 LLM 调用：出题 50 + 生成 50 + 裁判 ~50） | 低（降级） | 量级小，付费档/免费档均够；run 断点续跑 + score 裁判失败跳过（不阻塞主表）兜底 |
| reranker 本地显存低估（实际 ~2G FP16 而非 1.6G） | 低 | 云 API 为主路径；本地仅在显存富余时启用 |

## 10. 演进路线（M4 之后）

> v1.5：原主线 E0~E4 对照实验降级为演进首位；三命名向量（dense/bm25/bge_sparse）与客户端 RRF 均照存，恢复对照零改造——只需多跑几遍 `retrieve.py --exp E0/E1/E2/E4` 并汇总。

1. **恢复对照实验（演进第一候选）**：E0（纯向量）vs E1（纯 BM25）vs E2（混合无重排）vs E3（完整链路）四组对照，复用 M4 的 50 合成题或扩容至 100~200；E4（bge_sparse）只换 vector name。前提：先补上 §9"单列无归因"的欠账——要讲"混合增益多少"必须有对照
2. **语料扩展**：v1.4 已是服务端模式，20 万+ chunk 无需再迁架构；新年报增量入库（去重键 `stock_code + fiscal_year + chunk_idx`，point id 由 `chunk_id` 派生 UUID5 保证幂等）——**v1.6 已具备**：`--codes` 增量入库、`--replace` 整份替换，无任何全库重算环节
3. **查询侧增强**：查询改写、HyDE、多轮对话、**query→元数据过滤自动抽取**（公司名/年份正则或 LLM 抽取，年报场景最实用——当前过滤是手动入口）
4. **切块策略优化**：切分粒度对照（512/768/1024，需第二份索引）、结构感知切分（按年报章节）、small-to-big（命中后回捞邻块补上下文）
5. **评测深化**：人工金标准标注（pooling + nDCG@10）替代 RAGAS 近似口径；RAGAS 裁判校准（align-llm-as-judge）；异家族裁判交叉验证
6. **远期候选**：GraphRAG（Neo4j 已装）、多模态（年报图表）、ColBERT late interaction（BGE-M3 第三表示，权重已在本地）

## 附录：文档体系

| 文档 | 状态 | 内容 |
|---|---|---|
| `architecture.md` | **活跃** | 实施蓝图，随实施更新 |
| `01-技术选型对比.md`（v1） | 留档 | 首轮选型过程：引擎四方对比、显存核查、C-MTEB 规模核查 |
| `01-技术选型对比-v2.md` | 留档 | 数据源变更（巨潮年报）后的方案快照 |

历史核查结论（显存数字、RRF 阈值、API 规格、数据集规模）均已吸收进本文档相应章节；完整核查来源见 v1/v2 附录。
