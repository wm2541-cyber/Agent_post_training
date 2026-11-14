# Hidden Premise Conflict Lab (`Agent_post_training`)

围绕 **多智能体双向纠错与反作弊评判** 的可验证实验基准与人机对抗环境：
复现“主从单向调度（Fork-Join）vs 两阶段试探（Two-Stage Scout）vs 双向质疑重规划（Bidirectional Escalation）”在**含隐含前提冲突**代码沙盒中的行为差异，并通过机械化硬隔离、配对归因与交互博弈验证多智能体协作边界。

## 核心机制

1. **四层独立评判框架**（[judge.py](hidden_premise_lab/judge.py) / [sandbox.py](hidden_premise_lab/sandbox.py)）
   - Layer 0：不可逾越的 Worker `writable_paths` 物理写范围锁与受保护测试目录拦截。
   - Layer 1：SHA-256 公共测试完整性校验与额外测试注入检测。
   - Layer 2：彻底隔离的 Hidden 终验套件（永不暴露给任何 Agent 提示词或 Arena 存档）。
   - Layer 3：基于真实工具观测与冲突标志物的向上质疑审计。
   - Layer 4：事件配对归因、`protocol_error` 抗早停重罚与个体级 `per_worker_rewards` 奖惩结算（[reward.py](hidden_premise_lab/reward.py)）。
2. **零兜底严格协议引擎**（[live_llm.py](hidden_premise_lab/live_llm.py)）
   - 移除未识别动作静默转 `complete` 的兜底，采用严格 JSON Schema 校验 + 有限次 `PROTOCOL_ERROR` 打回重试 + 脱敏 JSONL 原始响应日志。
3. **回合制交互博弈引擎 Interactive Arena**（[arena.py](hidden_premise_lab/arena.py)）
   - 支持命令行 `save`/`step` 持久化 JSON 状态，可由人类 / 强 Agent 扮演 Lead 或 Worker，在真实隔离沙盒中与 LLM（如 `deepseek-flash`）跨角色回合对战。

## 快速运行

在仓库根目录 `hidden_premise_lab/` 下运行：

```bash
# 1. 运行 6 种策略 x 3 个任务的模拟基准矩阵
uv run python3 -m hidden_premise_lab.cli --mode simulate

# 2. 运行真实 LLM 对照评测（需在环境变量或 ~/.env 配置 DEEPSEEK_API_KEY / OPENAI_API_KEY）
uv run python3 -m hidden_premise_lab.cli --mode live --model deepseek-flash

# 3. 交互博弈：查看任务列表 / 扮演 Lead 对战 LLM Workers
uv run python3 -m hidden_premise_lab.arena tasks
uv run python3 -m hidden_premise_lab.arena init --task invariant_cache_lock --mode bidirectional_escalation --human lead --state logs/match.json --max-steps 30 --json
uv run python3 -m hidden_premise_lab.arena step --state logs/match.json --action-json '<assignments_json>' --until-human --json

# 4. 代码质量与自动化测试（ruff + mypy + pytest）
uv run --with ruff ruff check hidden_premise_lab tests
uv run --with mypy --with pytest mypy hidden_premise_lab tests
uv run --with pytest pytest -v
```
