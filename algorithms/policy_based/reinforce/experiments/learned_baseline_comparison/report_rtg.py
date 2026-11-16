"""Three-way RTG ablation report with paired seeds and a common step budget.

Controls are reused training runs, then evaluated on the same new test set as
the RTG variant. Historical wall-clock times are reported, not ranked as a
controlled speed benchmark. Learning-speed estimates use validation records.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from report import (_aggregate_curves, _curve, _finite, _fmt, _percent,
                    _plot_style, _read_json, _score, _statistics)


VARIANTS = ("vanilla", "baseline", "rtg_baseline")
LABELS = {"vanilla": "纯 REINFORCE", "baseline": "整回报 + baseline", "rtg_baseline": "RTG + baseline"}
COLORS = {"vanilla": "#2474b5", "baseline": "#e07b28", "rtg_baseline": "#238569"}
THRESHOLD = 950.0


def _load_runs(output, protocol):
    runs = []
    for seed in protocol["seeds"]:
        for variant in VARIANTS:
            folder = output / f"{variant}_seed_{seed}"
            trial = _read_json(folder / "trial.json", {})
            validation_file = folder / "validation.jsonl"
            validation = [json.loads(line) for line in validation_file.read_text().splitlines() if line.strip()] if validation_file.exists() else []
            first_pass_seconds = _finite(trial.get("first_pass_seconds"))
            if first_pass_seconds is None and trial.get("first_pass_episode") is not None:
                match = next((row for row in validation if row.get("episode") == trial["first_pass_episode"]), None)
                first_pass_seconds = _finite(match.get("train_seconds")) if match else None
            runs.append({"variant": variant, "seed": seed, "status": trial.get("status", "missing"),
                         "trial": trial, "validation": validation, "first_pass_seconds": first_pass_seconds,
                         "test": _read_json(folder / "test.json", {}),
                         "probe": _read_json(folder / "gradient_probe.json", {}) if variant == "rtg_baseline" else {},
                         "reused": bool(trial.get("reused_from")) or variant not in protocol.get("trained_variants", ["rtg_baseline"])})
    return runs


def _paired(seeds, values, control):
    rows = []
    for seed in seeds:
        left, right = _finite(values.get(("rtg_baseline", seed))), _finite(values.get((control, seed)))
        rows.append({"seed": seed, "rtg_baseline": left, "control": right,
                     "difference": left - right if left is not None and right is not None else None})
    differences = [row["difference"] for row in rows if row["difference"] is not None]
    return {"control": control, "rows": rows, **_statistics(differences),
            "rtg_wins": sum(x > 0 for x in differences), "ties": sum(x == 0 for x in differences),
            "rtg_losses": sum(x < 0 for x in differences)}


def _fixed_budget(runs, seeds):
    """Integrate left-held validation values exactly, without extrapolation.

    A checkpoint's estimate applies until the next observed checkpoint. The
    final endpoint has measure zero in the integral; it is reported separately.
    """
    available = [(run, curve) for run in runs if (curve := _curve(run, "env_steps")) is not None]
    result = {"expected_runs": len(runs), "available_runs": len(available), "start_steps": None,
              "end_steps": None, "budget_steps": None, "rows": [], "groups": {}, "paired_endpoint": {}, "paired_normalized_auc": {}}
    if not available:
        return result
    low = max(curve[0, 0] for _, curve in available)
    high = min(curve[0, -1] for _, curve in available)
    if high <= low:
        return result
    result.update(start_steps=float(low), end_steps=float(high), budget_steps=float(high - low))
    for run, curve in available:
        x, y = curve
        knots = np.unique(np.concatenate(([low], x[(x > low) & (x < high)], [high])))
        held = y[np.searchsorted(x, knots, side="right") - 1]
        area = float(np.sum(held[:-1] * np.diff(knots)))
        result["rows"].append({"variant": run["variant"], "seed": run["seed"],
                               "endpoint_return": float(held[-1]), "auc": area,
                               "normalized_auc": float(area / (high - low)),
                               "endpoint_checkpoint_steps": float(x[np.searchsorted(x, high, side="right") - 1])})
    for variant in VARIANTS:
        rows = [row for row in result["rows"] if row["variant"] == variant]
        result["groups"][variant] = {"endpoint_return": _statistics([row["endpoint_return"] for row in rows]),
                                      "normalized_auc": _statistics([row["normalized_auc"] for row in rows])}
    for key, destination in (("endpoint_return", "paired_endpoint"), ("normalized_auc", "paired_normalized_auc")):
        values = {(row["variant"], row["seed"]): row[key] for row in result["rows"]}
        result[destination] = {control: _paired(seeds, values, control) for control in ("baseline", "vanilla")}
    return result


def _summarize(runs, protocol):
    groups = {}
    for variant in VARIANTS:
        group = [r for r in runs if r["variant"] == variant]
        final = [_score(r) for r in group]
        gaps = [_score(r, "best") - _score(r) for r in group if _score(r, "best") is not None and _score(r) is not None]
        groups[variant] = {
            "expected_seeds": len(protocol["seeds"]),
            "status_by_seed": {str(r["seed"]): r["status"] for r in group},
            "reused_training_runs": sum(r["reused"] for r in group),
            "final_stochastic_return": _statistics(final),
            "best_stochastic_return": _statistics([_score(r, "best") for r in group]),
            "final_deterministic_return": _statistics([_score(r, mode="deterministic") for r in group]),
            "final_full_length_fraction": _statistics([_score(r, key="full_length_fraction") for r in group]),
            "final_passing_seeds": sum(x is not None and x >= THRESHOLD for x in final),
            "training_passed_seeds": sum(r["trial"].get("first_pass_episode") is not None for r in group),
            "pass_episodes_among_passed_only": _statistics([r["trial"].get("first_pass_episode") for r in group]),
            "pass_steps_among_passed_only": _statistics([r["trial"].get("first_pass_steps") for r in group]),
            "pass_train_seconds_among_passed_only": _statistics([r["first_pass_seconds"] for r in group]),
            "env_steps": _statistics([r["trial"].get("env_steps") for r in group]),
            "total_env_steps": sum(v for v in (_finite(r["trial"].get("env_steps")) for r in group) if v is not None),
            "train_seconds": _statistics([r["trial"].get("train_seconds") for r in group]),
            "validation_seconds": _statistics([r["trial"].get("evaluation_seconds") for r in group]),
            "test_seconds": _statistics([r["test"].get("test_seconds") for r in group]),
            "gradient_probe_seconds": _statistics([r["probe"].get("seconds") for r in group]),
            "best_minus_final_test_return": _statistics(gaps),
        }
    final_values = {(r["variant"], r["seed"]): _score(r) for r in runs}
    probes = [{"seed": r["seed"], **r["probe"]} for r in runs if r["variant"] == "rtg_baseline"]
    mechanisms = {}
    for key in ("variance_ratio", "rtg_vs_full_baseline_variance_ratio"):
        ratios = [_finite(p.get(key)) for p in probes]
        mechanisms[key] = {**_statistics(ratios), "reduced_seeds": sum(x is not None and x < 1 for x in ratios)}
    return {"environment": protocol.get("env_name", "InvertedPendulum-v5"), "episodes_per_run": protocol["episodes"],
            "seeds": protocol["seeds"], "variants": list(VARIANTS), "control_run": protocol.get("control_run"),
            "primary_comparison": "rtg_baseline minus baseline", "threshold": THRESHOLD, "groups": groups,
            "final_paired_comparisons": {control: _paired(protocol["seeds"], final_values, control) for control in ("baseline", "vanilla")},
            "fixed_step_budget": _fixed_budget(runs, protocol["seeds"]),
            "gradient_probes": probes, "gradient_variance_ratios": mechanisms,
            "limitations": ["Controls reuse historical training; test policies are reevaluated on one new common test set.",
                            "Training seed is the statistical unit, not evaluation episode.",
                            "Common-budget values are previous-validation step-function estimates, not exact budget checkpoints.",
                            "Normalized AUC is area divided by shared environment-step interval, in return units.",
                            "Historical and new train_seconds are not a controlled wall-clock speed comparison.",
                            "Final fixed-policy gradient probes do not describe all training stages."]}


def _learning_curves(output, runs, protocol, summary, plt):
    fig, axes = plt.subplots(2, 3, figsize=(18, 9.5), constrained_layout=True)
    for ax, variant in zip(axes[0], VARIANTS):
        for run in (r for r in runs if r["variant"] == variant):
            curve = _curve(run, "episode")
            if curve is not None:
                ax.plot(*curve, linewidth=1.2, alpha=.85, label=f"seed {run['seed']}")
        ax.axhline(THRESHOLD, color="#666666", linestyle="--", linewidth=1)
        ax.set(title=LABELS[variant], xlabel="训练回合 / actor 更新数", ylabel="验证平均回报", xlim=(0, protocol["episodes"]), ylim=(-20, 1030))
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8, loc="upper left")
    for ax, xkey, xlabel in ((axes[1, 0], "episode", "训练回合 / actor 更新数"),
                              (axes[1, 1], "env_steps", "累计训练环境步数")):
        curves = [curve for run in runs if (curve := _curve(run, xkey)) is not None]
        common = (max(c[0, 0] for c in curves), min(c[0, -1] for c in curves)) if curves else None
        for variant in VARIANTS:
            aggregate = _aggregate_curves([r for r in runs if r["variant"] == variant], xkey, common)
            if aggregate is None:
                continue
            x, mean, spread, count = aggregate
            ax.step(x, mean, where="post", color=COLORS[variant], label=f"{LABELS[variant]}，n={count}", linewidth=2)
            ax.fill_between(x, mean-spread, mean+spread, step="post", color=COLORS[variant], alpha=.12)
        ax.axhline(THRESHOLD, color="#666666", linestyle="--", linewidth=1)
        ax.set(title="种子均值 ± 种子间标准差", xlabel=xlabel, ylabel="验证平均回报", ylim=(-20, 1030))
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8, loc="upper left")
    budget = summary["fixed_step_budget"]
    ax = axes[1, 2]
    if budget["groups"]:
        for index, variant in enumerate(VARIANTS):
            stats = budget["groups"][variant]
            for offset, key, hatch in ((-.19, "endpoint_return", ""), (.19, "normalized_auc", "//")):
                ax.bar(index+offset, stats[key]["mean"], .35, color=COLORS[variant], hatch=hatch, alpha=.9 if not hatch else .55)
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(facecolor="#777777", label="预算终点回报"),
                           Patch(facecolor="#aaaaaa", hatch="//", label="预算内平均回报（nAUC）")], fontsize=8)
        ax.set_xticks(range(3), [LABELS[v] for v in VARIANTS], fontsize=8)
        ax.set(title=f"共同预算终点 {_fmt(budget['end_steps'], 0)} 步", ylabel="验证平均回报", ylim=(0, 1030))
    else:
        ax.text(.5, .5, "共同预算数据尚不足", transform=ax.transAxes, ha="center")
    fig.suptitle("Reward-to-go 消融：相同回合预算与共同环境步数预算分别比较", fontsize=15)
    fig.savefig(output / "learning_curves.png")
    plt.close(fig)


def _final_comparison(output, runs, protocol, plt):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.8), constrained_layout=True)
    positions = np.arange(len(protocol["seeds"]))
    for ax, checkpoint, title in zip(axes, ("last", "best"), ("最终模型（主结果）", "验证最佳模型（补充）")):
        for offset, variant in zip((-.27, 0., .27), VARIANTS):
            scores = []
            for seed in protocol["seeds"]:
                run = next(r for r in runs if r["seed"] == seed and r["variant"] == variant)
                value = _score(run, checkpoint)
                scores.append(np.nan if value is None else value)
            bars = ax.bar(positions + offset, scores, width=.25, color=COLORS[variant], label=LABELS[variant])
            for bar, value in zip(bars, scores):
                if np.isfinite(value):
                    ax.text(bar.get_x()+bar.get_width()/2, value+13, f"{value:.0f}", ha="center", va="bottom", fontsize=8)
                else:
                    ax.text(bar.get_x()+bar.get_width()/2, 30, "缺失", ha="center", fontsize=8, rotation=90)
        ax.axhline(THRESHOLD, color="#666666", linestyle="--", linewidth=1)
        ax.set(xlabel="训练种子", ylabel="共同新测试集：随机策略平均回报", ylim=(0, 1110))
        ax.set_title(title, pad=38)
        ax.set_xticks(positions, protocol["seeds"])
        ax.legend(fontsize=8, loc="lower center", bbox_to_anchor=(.5, 1.01), ncol=3, frameon=False)
    fig.suptitle("三版本在同一新测试集重评；旧对照仅复用训练日志与冻结模型", fontsize=15)
    fig.savefig(output / "final_comparison.png")
    plt.close(fig)


def _diagnostics(output, summary, plt):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    cases = (("variance_ratio", "baseline 的作用：(G_t − V) / G_t"),
             ("rtg_vs_full_baseline_variance_ratio", "RTG 的作用：(G_t − V) / (R − V)"))
    positions = np.arange(len(summary["seeds"]))
    for ax, (key, title) in zip(axes, cases):
        ratios = [_finite(probe.get(key)) for probe in summary["gradient_probes"]]
        values = [np.nan if r is None else r for r in ratios]
        ax.bar(positions, values, color=["#238569" if v < 1 else "#bd4d4d" for v in values])
        ax.axhline(1., color="#666666", linestyle="--", linewidth=1)
        finite = [r for r in ratios if r is not None]
        if finite and min(finite) > 0 and max(finite)/min(finite) > 20:
            ax.set_yscale("log")
        for index, ratio in enumerate(ratios):
            if ratio is not None:
                ax.annotate(f"{ratio:.3g}", (index, ratio), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9)
            else:
                ax.text(index, .05, "缺失", transform=ax.get_xaxis_transform(), ha="center")
        ax.set(title=title, xlabel="RTG 训练种子", ylabel="经验梯度协方差迹之比（小于 1 表示降低）")
        ax.set_xticks(positions, summary["seeds"])
        ax.margins(y=.18)
    fig.suptitle("固定最终 RTG actor 与 critic，在相同独立轨迹上比较三种权重", fontsize=15)
    fig.savefig(output / "diagnostics.png")
    plt.close(fig)


def _interpretation(summary):
    groups = summary["groups"]
    pair = summary["final_paired_comparisons"]["baseline"]
    lines = []
    if pair["n"]:
        lines.append(f"**最终表现：**RTG 相对整回报 baseline 的配对均值差为 {_fmt(pair['mean'])}，"
                     f"中位数 {_fmt(pair['median'])}，胜 / 平 / 负为 {pair['rtg_wins']} / {pair['ties']} / {pair['rtg_losses']}；"
                     f"最终测试 ≥950 的种子数为 {groups['rtg_baseline']['final_passing_seeds']} / {groups['rtg_baseline']['expected_seeds']}，"
                     f"原 baseline 为 {groups['baseline']['final_passing_seeds']} / {groups['baseline']['expected_seeds']}。")
    budget = summary["fixed_step_budget"]
    if budget["groups"]:
        endpoint, auc = (budget[key]["baseline"] for key in ("paired_endpoint", "paired_normalized_auc"))
        lines.append(f"**相同采样量：**共同区间为 {_fmt(budget['start_steps'], 0)}–{_fmt(budget['end_steps'], 0)} 步；"
                     f"RTG 相对 baseline 的预算终点验证回报差为 {_fmt(endpoint['mean'])}，归一化 AUC 差为 {_fmt(auc['mean'])}。"
                     "这是按验证记录阶梯保持的估计，不能把 5000 回合后的分数直接当作样本效率。")
    passed = " / ".join(str(groups[v]["training_passed_seeds"]) for v in VARIANTS)
    lines.append(f"**达标与耗时：**纯 REINFORCE / 整回报 baseline / RTG 的连续验证达标种子数为 {passed}；"
                 "达标步数和回合数逐种子列出，未达标保留为空。旧两组的历史训练秒数与新运行不构成严格墙钟速度对照。")
    ratios = summary["gradient_variance_ratios"]
    a, b = ratios["variance_ratio"], ratios["rtg_vs_full_baseline_variance_ratio"]
    if a["n"] or b["n"]:
        lines.append(f"**局部减方差：**RTG+baseline / RTG 的方差比中位数为 {_fmt(a['median'], 4)}，"
                     f"RTG+baseline / 整回报+baseline 为 {_fmt(b['median'], 4)}。"
                     "两者都固定最终 RTG 策略并使用同一批轨迹，不能推断整个训练过程都具有相同改善。")
    return lines


def _write_report(output, runs, protocol, summary):
    expected = len(protocol["seeds"])
    complete = all(r["status"] == "complete" and r["trial"].get("episodes") == protocol["episodes"] and _score(r) is not None for r in runs)
    lines = [f"# {summary['environment']}：Reward-to-go + learned baseline 验证实验", "",
             "本次检验把 actor 的整回合回报 `R−V(s_t)` 改成每步未来回报 `G_t−V(s_t)`，是否改善最终表现和学习速度。"
             "critic 仍使用现有 TD 更新；未把 critic 改成 Monte Carlo 回归。", ""]
    if not complete:
        lines.extend(["**实验尚不完整；以下只描述已记录的结果，缺失种子不会补成成功或达标。**", ""])
    lines.extend([*[f"- {text}" for text in _interpretation(summary)], "", "## 协议与可比范围", "",
                  f"- 三组各 {expected} 个配对种子 `{protocol['seeds']}`，每个种子 {protocol['episodes']} 回合；本次仅训练 `rtg_baseline`。",
                  f"- 原纯 REINFORCE 与整回报 baseline 的训练日志和冻结模型来自 `{protocol.get('control_run', 'protocol.control_run')}`，原目录保持只读。",
                  "- 三组最终与验证最佳模型都在同一个新测试集重新评估，不混用上次测试成绩；最佳模型只由原验证集选择。",
                  f"- 验证集 {len(protocol.get('validation_seeds', []))} 回合，测试集 {len(protocol.get('test_seeds', []))} 回合；"
                  f"新梯度探针 {len(protocol.get('probe_seeds', []))} 条轨迹。种子范围见 `protocol.json`。",
                  f"- 配置：`{json.dumps(protocol.get('config', {}), ensure_ascii=False, sort_keys=True)}`。保持 gamma=1、单回合更新、sum actor loss、原高斯动作执行方式及 TD critic 每回合一次更新。",
                  "- actor 的 G_t 截止采样轨迹结束；critic 在时间限制截断时仍 bootstrap。因此本次没有把 actor 与 critic 的回报边界完全统一。",
                  "- 训练达标定义为连续两次验证均值 ≥950；达标后继续训练。最终测试达标为随机策略测试均值 ≥950。",
                  "- 统计独立单位为训练种子。跨种子标准差不等于置信区间，多个评估回合不当作多个独立训练实验。", "",
                  "## 最终随机策略：相同回合预算", "",
                  "| 算法 | 有效 / 计划种子 | 均值 ± 种子间标准差 | 中位数 | 种子均值范围 | 测试 ≥950 | 满 1000 步比例 | 总训练步数 | 平均训练秒 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for variant in VARIANTS:
        g = summary["groups"][variant]
        s = g["final_stochastic_return"]
        lines.append(f"| {LABELS[variant]} | {s['n']} / {expected} | {_fmt(s['mean'])} ± {_fmt(s['std'])} | {_fmt(s['median'])} | "
                     f"{_fmt(s['min'])}–{_fmt(s['max'])} | {g['final_passing_seeds']} / {expected} | {_percent(g['final_full_length_fraction']['mean'])} | "
                     f"{_fmt(g['total_env_steps'], 0)} | {_fmt(g['train_seconds']['mean'])}{'（历史）' if g['reused_training_runs'] else ''} |")
    lines.extend(["", "| 种子 | 纯 REINFORCE | 整回报 baseline | RTG baseline | RTG − 整回报 baseline | RTG − 纯 REINFORCE |",
                  "|---|---:|---:|---:|---:|---:|"])
    for seed in protocol["seeds"]:
        values = {r["variant"]: _score(r) for r in runs if r["seed"] == seed}
        differences = [next(p for p in summary["final_paired_comparisons"][control]["rows"] if p["seed"] == seed)["difference"] for control in ("baseline", "vanilla")]
        lines.append(f"| {seed} | {_fmt(values['vanilla'])} | {_fmt(values['baseline'])} | {_fmt(values['rtg_baseline'])} | "
                     f"{_fmt(differences[0])} | {_fmt(differences[1])} |")
    for control in ("baseline", "vanilla"):
        p = summary["final_paired_comparisons"][control]
        lines.extend(["", f"RTG − {LABELS[control]}：配对均值差 **{_fmt(p['mean'])} ± {_fmt(p['std'])}**，中位数 {_fmt(p['median'])}，"
                      f"胜 / 平 / 负 {p['rtg_wins']} / {p['ties']} / {p['rtg_losses']}（有效配对 {p['n']}）。"])
    lines.extend(["", "![最终与最佳模型测试](final_comparison.png)", "", "## 固定共同采样预算：学习速度", ""])
    budget = summary["fixed_step_budget"]
    if budget["groups"]:
        lines.extend([f"共有 {budget['available_runs']} / {budget['expected_runs']} 个运行具备验证曲线；共同覆盖区间为 "
                      f"**{_fmt(budget['start_steps'], 0)}–{_fmt(budget['end_steps'], 0)} 个训练环境步**。"
                      "终点取所有运行最后一次验证对应步数的最小值，避免向训练尚未覆盖的预算外推。", "",
                      "每次验证结果保持到下一次验证；归一化 AUC = 该阶梯曲线面积 / 共同步数区间长度，单位仍为回报。"
                      "这是验证集上的粗间隔估计，不是在精确共同步数保存并重新测试的模型成绩；算法的验证节点也不完全相同。", "",
                      "| 算法 | 预算终点验证回报：均值 ± 标准差 | 归一化 AUC：均值 ± 标准差 |",
                      "|---|---:|---:|"])
        for variant in VARIANTS:
            g = budget["groups"][variant]
            endpoint, auc = g["endpoint_return"], g["normalized_auc"]
            lines.append(f"| {LABELS[variant]} | {_fmt(endpoint['mean'])} ± {_fmt(endpoint['std'])} | {_fmt(auc['mean'])} ± {_fmt(auc['std'])} |")
        lines.extend(["", "| 种子 | 算法 | 终点前最后验证的训练步数 | 预算终点回报 | 归一化 AUC |", "|---:|---|---:|---:|---:|"])
        for row in budget["rows"]:
            lines.append(f"| {row['seed']} | {LABELS[row['variant']]} | {_fmt(row['endpoint_checkpoint_steps'], 0)} | "
                         f"{_fmt(row['endpoint_return'])} | {_fmt(row['normalized_auc'])} |")
        lines.extend(["", "| 配对比较 | 终点回报均值差 | 终点胜 / 平 / 负 | nAUC 均值差 | nAUC 胜 / 平 / 负 |", "|---|---:|---:|---:|---:|"])
        for control in ("baseline", "vanilla"):
            end, auc = (budget[key][control] for key in ("paired_endpoint", "paired_normalized_auc"))
            lines.append(f"| RTG − {LABELS[control]} | {_fmt(end['mean'])} | {end['rtg_wins']} / {end['ties']} / {end['rtg_losses']} | "
                         f"{_fmt(auc['mean'])} | {auc['rtg_wins']} / {auc['ties']} / {auc['rtg_losses']} |")
    else:
        lines.append("共同覆盖区间不足，无法计算固定预算比较。")
    lines.extend(["", "![学习曲线与共同预算比较](learning_curves.png)", "", "## 达标速度、稳定性与时间记录", "",
                  "| 算法 | 种子 | 状态 | 首次连续达标回合 | 达标环境步数 | 达标训练秒 | 最佳验证 / 回合 | 最终验证 | 训练秒 | 验证秒 | 新测试秒 |",
                  "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for run in runs:
        t = run["trial"]
        lines.append(f"| {LABELS[run['variant']]}{'（历史）' if run['reused'] else ''} | {run['seed']} | {run['status']} | "
                     f"{_fmt(t.get('first_pass_episode'), 0)} | {_fmt(t.get('first_pass_steps'), 0)} | {_fmt(run['first_pass_seconds'])} | "
                     f"{_fmt(t.get('best_validation_mean'))} / {_fmt(t.get('best_episode'), 0)} | {_fmt(t.get('last_validation_mean'))} | "
                     f"{_fmt(t.get('train_seconds'))} | {_fmt(t.get('evaluation_seconds'))} | {_fmt(run['test'].get('test_seconds'))} |")
    lines.extend(["", "“—”表示未达标或缺失，不能将未达标种子填为训练上限。达标秒数取对应验证节点记录的累计训练时间，"
                  "不含验证开销。达标速度需结合成功种子数与完整曲线判断，不能只挑成功种子计算平均。", "",
                  "历史对照和新增 RTG 训练并非同时运行，机器负载与并行调度不同；这里保留训练秒数作成本记录，"
                  "不据此声称严格的墙钟加速。相同回合数也不代表相同环境步数。", "",
                  "## 最佳模型与均值动作评估：补充", "",
                  "| 算法 | 种子 | 最终随机 | 最佳随机 | 最终均值动作 | 最佳均值动作 | 重载复现 |",
                  "|---|---:|---:|---:|---:|---:|---|"])
    for run in runs:
        reload = run["test"].get("reload_reproduced")
        lines.append(f"| {LABELS[run['variant']]} | {run['seed']} | {_fmt(_score(run))} | {_fmt(_score(run, 'best'))} | "
                     f"{_fmt(_score(run, mode='deterministic'))} | {_fmt(_score(run, 'best', 'deterministic'))} | "
                     f"{'是' if reload is True else '否' if reload is False else '—'} |")
    lines.extend(["", "均值动作不包含高斯探索，不能代替随机策略主结果。最佳模型优于最终模型时，应结合训练曲线检查后期退化，"
                  "不将最佳成绩当成一直保持的性能。", "", "## 固定最终 RTG 策略的梯度方差诊断", "",
                  "固定同一最终 RTG actor 与 critic，在同一批独立完整轨迹上分别计算 `G_t`、`G_t−V(s_t)`、"
                  "`R−V(s_t)` 加权的逐轨迹策略梯度（含 log_std）。方差使用各参数样本方差之和，即经验协方差迹。", "",
                  "| RTG 种子 | 轨迹数 | 探针均值回报 | RTG 无 baseline 方差迹 | RTG + baseline 方差迹 | 整回报 + baseline 方差迹 | (G_t−V) / G_t | (G_t−V) / (R−V) |",
                  "|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for p in summary["gradient_probes"]:
        variances = [_finite(p.get(key)) for key in ("without_baseline_variance_trace", "with_baseline_variance_trace", "full_return_baseline_variance_trace")]
        formatted = ["—" if value is None else f"{value:.6g}" for value in variances]
        lines.append(f"| {p['seed']} | {_fmt(p.get('trajectory_count'), 0)} | {_fmt(p.get('mean_return'))} | {' | '.join(formatted)} | "
                     f"{_fmt(p.get('variance_ratio'), 4)} | {_fmt(p.get('rtg_vs_full_baseline_variance_ratio'), 4)} |")
    for key, description in (("variance_ratio", "RTG + baseline / RTG"), ("rtg_vs_full_baseline_variance_ratio", "RTG + baseline / 整回报 + baseline")):
        s = summary["gradient_variance_ratios"][key]
        lines.extend(["", f"{description}：比值中位数 {_fmt(s['median'], 4)}，范围 {_fmt(s['min'], 4)}–{_fmt(s['max'], 4)}；"
                      f"{s['reduced_seeds']} / {s['n']} 个有效种子小于 1。"])
    lines.extend(["", "这是固定最终策略上的局部诊断，不是三个不同训练策略之间的梯度方差对比；不能推断全程训练方差，"
                  "也不能仅从低 TD loss 或回报标准差推断梯度方差。", "", "![两个局部方差比](diagnostics.png)", "",
                  "## 产物与复现", "",
                  "`protocol.json` 保存协议与随机种子，`frozen_models.json` 保存测试前模型哈希；各运行目录保存训练日志、"
                  "验证记录、最终 / 最佳 checkpoint 与新测试结果。`rtg_baseline_seed_*` 还保存新梯度探针。"
                  "`summary.json` 保存种子层面统计、配对差和共同预算计算结果；`snapshots/` 保存实验源码。"])
    execution = _read_json(output / "execution.json", {})
    if _finite(execution.get("elapsed_seconds")) is not None:
        lines.extend(["", f"本次执行墙钟时间 {_fmt(execution['elapsed_seconds'])} 秒，包含新增 RTG 训练及三组重新评估，"
                      "不包含历史控制组的训练时间。"])
    errors = [r for r in runs if r["trial"].get("error") or r["test"].get("error")]
    if errors:
        lines.extend(["", "失败 / 不完整记录："])
        for run in errors:
            lines.append(f"- {run['variant']} seed {run['seed']}：`{str(run['trial'].get('error') or run['test'].get('error')).replace('`', '')}`")
    lines.extend(["", "从仓库根目录重新执行本次对照；输出目录必须尚不存在。种子、回合数和验证间隔继承控制组协议：", "", "```bash",
                  ".venv/bin/python algorithms/policy_based/reinforce/experiments/learned_baseline_comparison/run_rtg.py run \\",
                  f"  --control-run '{protocol.get('control_run', 'runs/learned_baseline_comparison/20261116_standard')}' \\",
                  f"  --output runs/learned_baseline_comparison/rtg_reproduction --workers {protocol.get('workers', 3)}",
                  "```", "", "重新生成本报告，或重新评估已冻结模型：", "", "```bash",
                  f".venv/bin/python algorithms/policy_based/reinforce/experiments/learned_baseline_comparison/run_rtg.py report '{output}'",
                  f".venv/bin/python algorithms/policy_based/reinforce/experiments/learned_baseline_comparison/run_rtg.py evaluate '{output}' --workers {protocol.get('workers', 3)}",
                  "```", "", "依赖版本：", "", "```json", json.dumps(protocol.get("versions", {}), ensure_ascii=False, indent=2), "```", "",
                  "源码哈希：", "", "```json", json.dumps(protocol.get("source_hashes", {}), ensure_ascii=False, indent=2), "```", "",
                  f"结论只覆盖本次 {expected} 个训练种子、一个环境和当前参数，不构成对 RTG 或 baseline 的普遍性能保证。", ""])
    (output / "REPORT.md").write_text("\n".join(lines))


def generate_report(output: Path) -> dict:
    output = Path(output).resolve()
    protocol = _read_json(output / "protocol.json")
    if protocol is None:
        raise FileNotFoundError(output / "protocol.json")
    runs = _load_runs(output, protocol)
    summary = _summarize(runs, protocol)
    plt = _plot_style()
    _learning_curves(output, runs, protocol, summary, plt)
    _final_comparison(output, runs, protocol, plt)
    _diagnostics(output, summary, plt)
    _write_report(output, runs, protocol, summary)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = generate_report(args.output)
    print(json.dumps({"report": str(args.output / "REPORT.md"), "rtg_minus_baseline": result["final_paired_comparisons"]["baseline"]}, ensure_ascii=False))
