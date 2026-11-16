# Learned baseline 对照实验

此实验比较当前 `learned_baseline_version.py` 与 `my_minimal_continuous.py`，不修改两份教学代码。`core.py` 复用它们的网络、分布和采样函数，并忠实复现更新公式。`test_experiment.py` 直接调用原文件的 `train()` 核对一次完整更新。

后续的 `rtg_baseline` 消融实验对应新教学文件 `rtg_learned_baseline_version.py`：只将 actor 权重从整回合 `R-V(s_t)` 改成 `G_t-V(s_t)`，其余设置和 TD critic 更新保持不变。两个原始版本仍使用原有公式。

## 运行

从仓库根目录执行，使用现有 `.venv`：

```bash
.venv/bin/python -m unittest discover -s algorithms/policy_based/reinforce/experiments/learned_baseline_comparison -p 'test_*.py' -v
.venv/bin/python algorithms/policy_based/reinforce/experiments/learned_baseline_comparison/run.py run --output runs/learned_baseline_comparison/reproduction --seeds 101 202 303 404 505 --episodes 5000 --workers 3
```

输出目录必须尚不存在，以防覆盖实验结果。默认每 100 回合进行 20 回合随机策略验证，最后用 100 个独立测试种子评估；baseline 最终模型另作 64 条轨迹的梯度方差诊断。

```bash
# 只重绘图表、重新汇总已有数据
.venv/bin/python algorithms/policy_based/reinforce/experiments/learned_baseline_comparison/run.py report runs/learned_baseline_comparison/reproduction
# 重评已经冻结的模型，不训练或重新选择模型
.venv/bin/python algorithms/policy_based/reinforce/experiments/learned_baseline_comparison/run.py evaluate runs/learned_baseline_comparison/reproduction --workers 3
```

## 固定协议

- `InvertedPendulum-v5` 默认奖励与 1000 步上限，CPU 每进程单线程。
- 两组都采用 `[32,32]` Tanh actor、初始标准差 1、Adam 学习率 0.001、完整回合回报、每回合一次更新、actor loss 求和。
- Baseline 组添加 `[32,32]` critic，每回合一次学习率 0.001 的 TD 均方误差更新；目标 detach，gamma=1，真实终止不 bootstrap，截断继续 bootstrap。
- 不增加 reward-to-go、优势标准化、显式动作裁剪或提前停止。原始高斯动作直接传入环境。
- 配对 actor 初始化相同。actor、critic、训练动作、逐回合环境、评估动作随机数的确切规则保存在 `protocol.json`。评估不会改变训练随机数。
- 两组从头训练全部种子。达到连续两次验证均值 ≥950 只记录达标时刻，仍训练至 5000 回合。
- 所有训练结束后先生成 `frozen_models.json`，再评估最终模型和验证最佳模型；验证成绩相同则保留较早模型。独立测试不用于调参或模型选择。

## 结果解释

主结果是最终模型随机策略测试均值，统计单位是训练种子。报告保留每个种子与失败记录，提供配对差、种子间标准差和达标比例；均值动作策略与验证最佳模型仅作补充。

相同回合数不代表相同环境步数，因此曲线同时显示回合轴和样本数轴；步数汇总限制在所有运行的共同覆盖范围内。训练、验证、最终测试和诊断耗时分开记录，进程并行时耗时受调度影响。

梯度方差诊断在每个 baseline 最终模型上固定 actor 和 critic，对同一批独立完整轨迹计算有/无 baseline 的每轨迹梯度，使用各参数样本方差之和估计协方差迹，包含 `log_std` 参数。它只能说明该固定策略处的经验方差，不能证明整个训练过程始终减方差。

## 输出

根目录包含 `protocol.json`、源码快照、初始化检查、冻结模型哈希、`summary.json`、中文 `REPORT.md` 和三张 PNG。各算法/种子目录保存每回合 `training.csv`、验证汇总及逐回合明细、`trial.json`、`best.pt`、`last.pt`、逐回合测试数据；baseline 还保存 `gradient_probe.json`。训练或评估异常保留错误文件，不更换种子补跑。

## Reward-to-go 消融实验

```bash
.venv/bin/python algorithms/policy_based/reinforce/experiments/learned_baseline_comparison/run_rtg.py run \
  --control-run runs/learned_baseline_comparison/20261116_standard \
  --output runs/learned_baseline_comparison/rtg_reproduction --workers 3
.venv/bin/python algorithms/policy_based/reinforce/experiments/learned_baseline_comparison/run_rtg.py report runs/learned_baseline_comparison/rtg_reproduction
```

新训练 5 个 RTG + TD baseline 种子，每个仍为 5000 回合；训练种子、actor/critic 初值、随机数规则、超参数和验证种子继承已完成的对照协议。`G_t` 包含当前奖励，到已采样轨迹终点为止，gamma=1，不额外 bootstrap 或标准化。TD critic 仍在时间限制截断时 bootstrap，因此本次并没有同时改成 MC critic。

执行前校验原对照的源码、配置、依赖版本、全部回合日志和冻结模型哈希。复用其 10 次训练和既有最佳模型选择，不再次挑选、不覆盖原结果。所有三组共 30 个最佳/最终模型冻结后，统一用新的测试种子 `80000–80099` 重评。`control_provenance.json` 保存复用证据来源与哈希。

新 RTG 策略用 `90000–90063` 的轨迹做方差诊断：固定同一 actor、critic 与轨迹，比较 `G_t`、`G_t-V`、`R-V` 三种梯度权重。第一个比值检查 baseline 的局部作用，第二个比值检查 RTG 的局部作用。

报告将三组回报按回合和环境步数展示。共同步数预算的终点成绩和 AUC 是对已有验证日志作上一观测值保持的近似，不是任意等步数 checkpoint 的新独立测试；原对照没有保存所有中间模型。历史训练耗时与本轮耗时分别注明来源，不将并行调度下的墙钟差异直接归因为算法加速。
