"""Generate descriptive, seed-paired reports for the REINFORCE comparison.

The independent unit is a training seed, not an evaluation episode. Reports
retain missing/failed seeds and never turn unfinished training into a success.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np


VARIANTS = ("vanilla", "baseline")
LABELS = {"vanilla": "纯 REINFORCE", "baseline": "Learned baseline"}
COLORS = {"vanilla": "#2474b5", "baseline": "#e07b28"}
THRESHOLD = 950.0


def _read_json(path: Path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def _finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _statistics(values):
    data = np.asarray([v for v in (_finite(x) for x in values) if v is not None])
    if not data.size:
        return {"n": 0, "mean": None, "std": None, "median": None, "min": None, "max": None}
    return {"n": int(data.size), "mean": float(data.mean()),
            "std": float(data.std(ddof=1)) if data.size > 1 else 0.0,
            "median": float(np.median(data)), "min": float(data.min()), "max": float(data.max())}


def _score(run, checkpoint="last", mode="stochastic", key="mean"):
    return _finite(run["test"].get(checkpoint, {}).get(mode, {}).get("summary", {}).get(key))


def _fmt(value, digits=2):
    number = _finite(value)
    return "—" if number is None else f"{number:,.{digits}f}"


def _percent(value):
    number = _finite(value)
    return "—" if number is None else f"{100 * number:.1f}%"


def _load_runs(output, protocol):
    runs = []
    for seed in protocol["seeds"]:
        for variant in VARIANTS:
            directory = output / f"{variant}_seed_{seed}"
            trial = _read_json(directory / "trial.json", {})
            validation_path = directory / "validation.jsonl"
            validation = []
            if validation_path.exists():
                for line in validation_path.read_text().splitlines():
                    if line.strip():
                        validation.append(json.loads(line))
            training_path = directory / "training.csv"
            training = []
            if training_path.exists():
                with training_path.open(newline="") as handle:
                    training = list(csv.DictReader(handle))
            runs.append({"variant": variant, "seed": seed, "directory": directory.name,
                         "trial": trial, "status": trial.get("status", "missing"),
                         "validation": validation, "training": training,
                         "test": _read_json(directory / "test.json", {}),
                         "probe": _read_json(directory / "gradient_probe.json", {})})
    return runs


def _summarize(runs, protocol):
    groups = {}
    for variant in VARIANTS:
        group = [r for r in runs if r["variant"] == variant]
        final = [_score(r) for r in group]
        groups[variant] = {
            "expected_seeds": len(protocol["seeds"]),
            "status_by_seed": {str(r["seed"]): r["status"] for r in group},
            "final_stochastic_return": _statistics(final),
            "best_stochastic_return": _statistics([_score(r, "best") for r in group]),
            "final_deterministic_return": _statistics([_score(r, mode="deterministic") for r in group]),
            "final_full_length_fraction": _statistics([_score(r, key="full_length_fraction") for r in group]),
            "final_passing_seeds": sum(x is not None and x >= THRESHOLD for x in final),
            "training_passed_seeds": sum(r["trial"].get("first_pass_episode") is not None for r in group),
            "pass_episodes_among_passed_only": _statistics([r["trial"].get("first_pass_episode") for r in group]),
            "pass_steps_among_passed_only": _statistics([r["trial"].get("first_pass_steps") for r in group]),
            "train_seconds": _statistics([r["trial"].get("train_seconds") for r in group]),
            "validation_seconds": _statistics([r["trial"].get("evaluation_seconds") for r in group]),
            "test_seconds": _statistics([r["test"].get("test_seconds") for r in group]),
            "gradient_probe_seconds": _statistics([r["probe"].get("seconds") for r in group]),
            "env_steps": _statistics([r["trial"].get("env_steps") for r in group]),
            "validation_best_minus_last": _statistics([
                _finite(r["trial"].get("best_validation_mean")) - _finite(r["trial"].get("last_validation_mean"))
                for r in group if _finite(r["trial"].get("best_validation_mean")) is not None
                and _finite(r["trial"].get("last_validation_mean")) is not None]),
        }
    pairs = []
    for seed in protocol["seeds"]:
        pair = {r["variant"]: r for r in runs if r["seed"] == seed}
        vanilla, baseline = _score(pair["vanilla"]), _score(pair["baseline"])
        pairs.append({"seed": seed, "vanilla": vanilla, "baseline": baseline,
                      "difference": None if vanilla is None or baseline is None else baseline - vanilla})
    differences = [p["difference"] for p in pairs if p["difference"] is not None]
    positive_pairs = sorted((p for p in pairs if p["difference"] is not None and p["difference"] > 0),
                            key=lambda p: p["difference"], reverse=True)
    largest_positive_pairs = positive_pairs[:2]
    positive_total = sum(p["difference"] for p in positive_pairs)
    step_totals = {variant: sum(value for value in
                              (_finite(r["trial"].get("env_steps")) for r in runs if r["variant"] == variant)
                              if value is not None) for variant in VARIANTS}
    step_ratio = step_totals["baseline"] / step_totals["vanilla"] if step_totals["vanilla"] > 0 else None
    paired_best_final_gaps = {}
    for variant in VARIANTS:
        gaps = [_score(r, "best") - _score(r) for r in runs if r["variant"] == variant
                and _score(r, "best") is not None and _score(r) is not None]
        paired_best_final_gaps[variant] = _statistics(gaps)
    probes = [{"seed": r["seed"], **r["probe"]} for r in runs if r["variant"] == "baseline"]
    ratios = [_finite(p.get("variance_ratio")) for p in probes]
    return {
        "environment": protocol.get("env_name", "InvertedPendulum-v5"),
        "episodes_per_run": protocol["episodes"], "seeds": protocol["seeds"],
        "primary_endpoint": "Final checkpoint stochastic evaluation mean per training seed",
        "threshold": THRESHOLD, "groups": groups, "paired_results": pairs,
        "paired_difference": {**_statistics(differences),
                              "baseline_wins": sum(x > 0 for x in differences),
                              "ties": sum(x == 0 for x in differences),
                              "baseline_losses": sum(x < 0 for x in differences)},
        "largest_positive_pairs": largest_positive_pairs,
        "top_two_positive_difference_share": (sum(p["difference"] for p in largest_positive_pairs) / positive_total
                                                if positive_total > 0 else None),
        "training_step_cost": {"total_steps": step_totals, "baseline_to_vanilla_ratio": step_ratio,
                               "baseline_relative_change": step_ratio - 1 if step_ratio is not None else None},
        "best_minus_final_test_return": paired_best_final_gaps,
        "gradient_probes": probes, "gradient_variance_ratio": _statistics(ratios),
        "gradient_variance_reduced_seeds": sum(x is not None and x < 1 for x in ratios),
        "limitations": [f"{len(protocol['seeds'])} training seeds are descriptive evidence, not a universal guarantee.",
                        "Equal episodes do not imply equal environment steps.",
                        "Gradient probes compare weights on the same frozen baseline-trained policy.",
                        "Validation checkpoint selection uses no final-test data."],
    }


def _plot_style():
    # Set before importing matplotlib, including on read-only home directories.
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "reinforce-comparison-mpl"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    available = {f.name for f in font_manager.fontManager.ttflist}
    if "Noto Sans CJK JP" in available:
        plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.18,
                         "axes.unicode_minus": False, "savefig.dpi": 160,
                         "figure.facecolor": "white", "axes.facecolor": "white"})
    return plt


def _curve(run, xkey):
    pairs = [(float(row[xkey]), float(row["mean"])) for row in run["validation"]
             if _finite(row.get(xkey)) is not None and _finite(row.get("mean")) is not None]
    # Last observation wins if a final validation repeats an x coordinate.
    pairs = sorted(dict(pairs).items())
    if not pairs:
        return None
    return np.asarray(pairs, dtype=float).T


def _aggregate_curves(runs, xkey, common_interval=None):
    curves = [_curve(r, xkey) for r in runs]
    curves = [c for c in curves if c is not None]
    if not curves:
        return None
    low = max(c[0, 0] for c in curves)
    high = min(c[0, -1] for c in curves)
    if common_interval is not None:
        low, high = max(low, common_interval[0]), min(high, common_interval[1])
    if high < low:
        return None
    grid = np.unique(np.concatenate([c[0, (c[0] >= low) & (c[0] <= high)] for c in curves] + [np.array([low, high])]))
    # Previous-observation interpolation; common coverage prevents extrapolation.
    values = np.vstack([c[1, np.searchsorted(c[0], grid, side="right") - 1] for c in curves])
    spread = values.std(axis=0, ddof=1) if len(curves) > 1 else np.zeros(len(grid))
    return grid, values.mean(axis=0), spread, len(curves)


def _learning_curves(output, runs, protocol, plt):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for ax, variant in zip(axes[0], VARIANTS):
        group = [r for r in runs if r["variant"] == variant]
        for run in group:
            curve = _curve(run, "episode")
            if curve is not None:
                ax.plot(*curve, alpha=0.8, linewidth=1.2, label=f"seed {run['seed']} ({run['status']})")
        ax.set(title=f"{LABELS[variant]}：逐种子随机策略验证", xlabel="训练回合 / actor 更新数", ylabel="验证平均回报")
        ax.set_xlim(0, protocol["episodes"])
        ax.set_ylim(-20, 1030)
        ax.axhline(THRESHOLD, color="#555555", linestyle="--", linewidth=1)
        if ax.lines:
            ax.legend(fontsize=8, loc="upper left")
    for ax, xkey, xlabel in [(axes[1, 0], "episode", "训练回合 / actor 更新数"),
                              (axes[1, 1], "env_steps", "累计训练环境步数（不含评估）")]:
        all_curves = [c for c in (_curve(r, xkey) for r in runs) if c is not None]
        common = None if not all_curves else (max(c[0, 0] for c in all_curves), min(c[0, -1] for c in all_curves))
        for variant in VARIANTS:
            result = _aggregate_curves([r for r in runs if r["variant"] == variant], xkey, common)
            if result is None:
                continue
            x, y, spread, n = result
            ax.step(x, y, where="post", color=COLORS[variant], linewidth=2,
                    label=f"{LABELS[variant]}，n={n}")
            ax.fill_between(x, y - spread, y + spread, step="post", color=COLORS[variant], alpha=0.14)
        ax.axhline(THRESHOLD, color="#555555", linestyle="--", linewidth=1)
        ax.set(title="种子均值 ± 1 个种子间标准差", xlabel=xlabel, ylabel="验证平均回报")
        ax.set_ylim(-20, 1030)
        ax.legend(fontsize=9, loc="upper left")
    fig.suptitle("训练过程：验证集与训练随机数隔离；步数汇总仅画共同覆盖区间", fontsize=14)
    fig.savefig(output / "learning_curves.png")
    plt.close(fig)


def _final_comparison(output, runs, protocol, plt):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    positions = np.arange(len(protocol["seeds"]))
    for ax, checkpoint, title in zip(axes, ("last", "best"), ("最终模型（主结果）", "验证最佳模型（补充）")):
        for offset, variant in zip((-0.18, 0.18), VARIANTS):
            scores = []
            for seed in protocol["seeds"]:
                run = next(r for r in runs if r["seed"] == seed and r["variant"] == variant)
                value = _score(run, checkpoint)
                scores.append(np.nan if value is None else value)
            bars = ax.bar(positions + offset, scores, width=0.34, color=COLORS[variant], label=LABELS[variant])
            for bar, score in zip(bars, scores):
                if np.isfinite(score):
                    ax.text(bar.get_x() + bar.get_width() / 2, score + 13, f"{score:.0f}",
                            ha="center", va="bottom", fontsize=8)
                else:
                    ax.text(bar.get_x() + bar.get_width() / 2, 30, "缺失", ha="center", fontsize=8, rotation=90)
        ax.axhline(THRESHOLD, color="#555555", linestyle="--", linewidth=1)
        ax.set(xlabel="训练种子", ylabel="独立测试随机策略平均回报", ylim=(0, 1110))
        ax.set_title(title, pad=38)
        ax.set_xticks(positions, protocol["seeds"])
        ax.legend(fontsize=9, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False)
    fig.suptitle("相同测试种子的配对比较；最佳模型仅由验证集选择", fontsize=14)
    fig.savefig(output / "final_comparison.png")
    plt.close(fig)


def _diagnostics(output, runs, protocol, plt):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    positions = np.arange(len(protocol["seeds"]))
    ratios = []
    for seed in protocol["seeds"]:
        run = next(r for r in runs if r["seed"] == seed and r["variant"] == "baseline")
        ratio = _finite(run["probe"].get("variance_ratio"))
        ratios.append(np.nan if ratio is None else ratio)
    finite_ratios = [r for r in ratios if np.isfinite(r)]
    axes[0].bar(positions, ratios, color=["#39836e" if r < 1 else "#bd4d4d" for r in ratios])
    axes[0].axhline(1, color="#555555", linestyle="--", linewidth=1)
    if finite_ratios and all(r > 0 for r in finite_ratios) and max(finite_ratios) / min(finite_ratios) > 20:
        axes[0].set_yscale("log")
    for index, ratio in enumerate(ratios):
        if np.isfinite(ratio):
            axes[0].annotate(f"{ratio:.3g}", (index, ratio), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9)
        else:
            axes[0].text(index, 0.05, "缺失", ha="center", fontsize=8, transform=axes[0].get_xaxis_transform())
    axes[0].set(title="固定策略：有 / 无 baseline 梯度方差比", xlabel="baseline 训练种子", ylabel="经验协方差迹之比（小于 1 表示降低）")
    axes[0].set_xticks(positions, protocol["seeds"])
    axes[0].margins(y=0.18)
    for offset, variant in zip((-0.18, 0.18), VARIANTS):
        gaps = []
        for seed in protocol["seeds"]:
            run = next(r for r in runs if r["seed"] == seed and r["variant"] == variant)
            best, last = (_finite(run["trial"].get(key)) for key in ("best_validation_mean", "last_validation_mean"))
            gaps.append(np.nan if best is None or last is None else best - last)
        axes[1].bar(positions + offset, gaps, width=0.34, color=COLORS[variant], label=LABELS[variant])
    axes[1].set(title="训练后期退化：最佳验证均值 − 最终验证均值", xlabel="训练种子", ylabel="回报下降量（越小越好）")
    axes[1].set_xticks(positions, protocol["seeds"])
    axes[1].legend(fontsize=9)
    fig.suptitle("梯度方差诊断与训练稳定性回答不同的问题", fontsize=14)
    fig.savefig(output / "diagnostics.png")
    plt.close(fig)


def _write_report(output, runs, protocol, summary):
    groups = summary["groups"]
    paired = summary["paired_difference"]
    expected = len(protocol["seeds"])
    complete = all(r["status"] == "complete" and r["trial"].get("episodes") == protocol["episodes"] for r in runs)
    all_tested = paired["n"] == expected
    difference = paired["mean"]
    if difference is None:
        conclusion = "尚无完整配对测试结果，不能判断改进效果。"
    else:
        change = f"{'提高' if difference > 0 else '降低'}了 {_fmt(abs(difference))}" if difference else "保持相同（配对均值差 0.00）"
        conclusion = (f"在 {paired['n']} 个有完整测试结果的配对种子上，learned baseline 的最终随机策略平均回报"
                      f"{change}；胜 / 平 / 负为 "
                      f"{paired['baseline_wins']} / {paired['ties']} / {paired['baseline_losses']}。")
    if not complete or not all_tested:
        conclusion = "**实验尚不完整；以下只描述已有结果。** " + conclusion
    interpretation = []
    if paired["n"]:
        text = (f"**最终表现：**纯 REINFORCE / baseline 的种子间标准差为 "
                f"{_fmt(groups['vanilla']['final_stochastic_return']['std'])} / {_fmt(groups['baseline']['final_stochastic_return']['std'])}，"
                f"配对差中位数为 {_fmt(paired['median'])}。")
        largest = summary["largest_positive_pairs"]
        if largest:
            gains = "、".join(f"{p['seed']}（+{_fmt(p['difference'])}）" for p in largest)
            concentration = summary["top_two_positive_difference_share"]
            text += f"{'正向增益主要来自' if concentration >= .75 else '正向差最大的种子为'} {gains}；"
        text += (f"最终均分 ≥950 的种子数分别为 {groups['vanilla']['final_passing_seeds']}/{expected}、"
                 f"{groups['baseline']['final_passing_seeds']}/{expected}。")
        interpretation.append(text)
    gaps = summary["best_minus_final_test_return"]
    if all(gaps[v]["n"] for v in VARIANTS):
        gap_text = (f"**训练退化：**纯 REINFORCE / baseline 的验证最佳模型测试均值分别为 "
                    f"{_fmt(groups['vanilla']['best_stochastic_return']['mean'])} / {_fmt(groups['baseline']['best_stochastic_return']['mean'])}，"
                    f"比各自最终模型平均高 {_fmt(gaps['vanilla']['mean'])} / {_fmt(gaps['baseline']['mean'])}。")
        if all(gaps[v]["mean"] > 0 for v in VARIANTS):
            gap_text += "两组都存在从最佳模型到最终模型的性能损失，baseline 没有消除训练后期退化。"
        interpretation.append(gap_text)
    cost = summary["training_step_cost"]
    if cost["baseline_relative_change"] is not None:
        change = cost["baseline_relative_change"]
        direction = "多" if change > 0 else "少" if change < 0 else "相同"
        comparison = f"{direction} {100 * abs(change):.2f}%" if change else direction
        interpretation.append(f"**样本成本：**baseline 共用 {_fmt(cost['total_steps']['baseline'], 0)} 步，"
                              f"纯 REINFORCE 为 {_fmt(cost['total_steps']['vanilla'], 0)} 步，{comparison}。"
                              "训练回合数相同不代表采样量相同，不能仅凭回合轴上的成绩断言样本效率提升。")
    ratio = summary["gradient_variance_ratio"]
    if ratio["n"]:
        sizes = sorted({_finite(p.get("trajectory_count")) for p in summary["gradient_probes"]
                        if _finite(p.get("trajectory_count")) is not None})
        sample_text = "/".join(_fmt(n, 0) for n in sizes)
        interpretation.append(f"**减方差机制：**最终固定策略的梯度方差比中位数为 {_fmt(ratio['median'], 4)}，"
                              f"{summary['gradient_variance_reduced_seeds']}/{ratio['n']} 个种子的比值小于 1。"
                              f"这是每个最终 baseline 策略上 {sample_text} 条轨迹的局部诊断，不代表全程训练效果。")
    lines = [f"# {summary['environment']}：Learned baseline 与纯 REINFORCE 对照实验", "", conclusion, "",
             "主要比较训练结束时的模型。每个训练种子的测试均值作为一个统计样本；测试回合不当作独立训练重复。", "",
             *[f"- {item}" for item in interpretation], "",
             "## 实验协议", "",
             f"- 计划规模：两组各 {expected} 个种子，每个种子 {protocol['episodes']} 回合；训练种子 `{protocol['seeds']}`。",
             "- 环境使用默认奖励、初始化分布和 1000 步上限；保持原代码的原始高斯动作执行方式，不额外添加动作裁剪。",
             f"- 配置：`{json.dumps(protocol.get('config', {}), ensure_ascii=False, sort_keys=True)}`；完整回合 return、gamma=1、每回合一次 actor 更新。",
             "- 唯一算法差别为当前版本的 TD learned baseline：平方误差、固定 TD target、真实终止不 bootstrap，时间限制截断继续 bootstrap。",
             "- 配对 actor 和 log_std 初值相同；critic 初始化、训练采样、评估随机数隔离。每组均重新训练，不直接引用历史调参成绩。",
             f"- 验证集 {len(protocol.get('validation_seeds', []))} 回合；独立测试集 {len(protocol.get('test_seeds', []))} 回合。最佳模型仅按验证集选择，测试前冻结模型。",
             "- 训练达标定义为连续两次验证均值 ≥950；达标后继续训练。最终测试达标为随机策略测试均值 ≥950。", "",
             "## 最终随机策略：主要结果", "",
             "| 算法 | 有效种子 / 计划 | 种子均值 ± 种子间标准差 | 中位数 | 种子均值范围 | 测试达标种子 | 平均满 1000 步比例 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for variant in VARIANTS:
        group = groups[variant]
        stats = group["final_stochastic_return"]
        lines.append(f"| {LABELS[variant]} | {stats['n']} / {expected} | {_fmt(stats['mean'])} ± {_fmt(stats['std'])} | "
                     f"{_fmt(stats['median'])} | {_fmt(stats['min'])}–{_fmt(stats['max'])} | {group['final_passing_seeds']} / {expected} | "
                     f"{_percent(group['final_full_length_fraction']['mean'])} |")
    lines.extend(["", "| 配对种子 | 纯 REINFORCE 最终测试均值 | Baseline 最终测试均值 | Baseline − REINFORCE |",
                  "|---|---:|---:|---:|"])
    for pair in summary["paired_results"]:
        lines.append(f"| {pair['seed']} | {_fmt(pair['vanilla'])} | {_fmt(pair['baseline'])} | {_fmt(pair['difference'])} |")
    lines.extend(["", f"配对差的均值 ± 标准差：**{_fmt(paired['mean'])} ± {_fmt(paired['std'])}**；"
                  f"中位数 {_fmt(paired['median'])}，范围 {_fmt(paired['min'])}–{_fmt(paired['max'])}。", "",
                  "![最终与最佳模型比较](final_comparison.png)", "", "## 学习速度、稳定性与运行成本", "",
                  "| 算法 | 种子 | 状态 | 完成回合 | 训练环境步数 | 首次连续达标回合 | 首次连续达标步数 | 最佳验证 / 回合 | 最终验证 | 训练秒 | 验证秒 | 测试秒 | 梯度诊断秒 |",
                  "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for run in runs:
        t = run["trial"]
        lines.append(f"| {LABELS[run['variant']]} | {run['seed']} | {run['status']} | {_fmt(t.get('episodes'), 0)} | "
                     f"{_fmt(t.get('env_steps'), 0)} | {_fmt(t.get('first_pass_episode'), 0)} | {_fmt(t.get('first_pass_steps'), 0)} | "
                     f"{_fmt(t.get('best_validation_mean'))} / {_fmt(t.get('best_episode'), 0)} | {_fmt(t.get('last_validation_mean'))} | "
                     f"{_fmt(t.get('train_seconds'))} | {_fmt(t.get('evaluation_seconds'))} | "
                     f"{_fmt(run['test'].get('test_seconds'))} | {_fmt(run['probe'].get('seconds'))} |")
    lines.extend(["", f"“—”表示未达到标准、不适用或数据缺失，不将未达标种子填成第 {protocol['episodes']} 回合。相同回合数代表相同 actor 更新次数，"
                  "不代表相同训练样本量。达到 950 的时间只在达标种子中有定义，不能忽略失败种子后直接比较速度。", "",
                  "按环境步数汇总的曲线使用上一观测值阶梯保持，仅展示所有有记录运行的共同覆盖区间，不向未观测步数外推。"
                  "阴影是种子间标准差，不是置信区间。耗时分别计量：训练为采样与更新，验证为训练中的周期性验证，"
                  "测试为最终 / 最佳模型的随机与均值动作评估及重载检查，梯度诊断为冻结策略的方差探针。"
                  "并行运行时各项耗时受系统调度影响，其总和不等于整场实验的墙钟时间。", "",
                  "![训练与验证曲线](learning_curves.png)", "", "## 最佳模型与均值动作策略：补充结果", "",
                  "| 算法 | 种子 | 最终随机策略 | 最佳随机策略 | 最终均值动作 | 最佳均值动作 | 重载复现 |",
                  "|---|---:|---:|---:|---:|---:|---|"])
    for run in runs:
        reloaded = run["test"].get("reload_reproduced")
        lines.append(f"| {LABELS[run['variant']]} | {run['seed']} | {_fmt(_score(run))} | {_fmt(_score(run, 'best'))} | "
                     f"{_fmt(_score(run, mode='deterministic'))} | {_fmt(_score(run, 'best', 'deterministic'))} | "
                     f"{'是' if reloaded is True else '否' if reloaded is False else '—'} |")
    lines.extend(["", "均值动作评估使用高斯分布的均值，不包含训练策略的随机探索；它不能替代随机策略的主结果。"
                  "最佳模型可能明显优于最终模型，这反映训练过程的退化风险。", "",
                  "## 固定策略的梯度方差验证", "",
                  "每个 baseline 运行固定最终 actor 与 critic，另采样独立完整轨迹。对同一条轨迹分别计算 "
                  "`−Σ logπ × R` 与 `−Σ logπ × (R−V)` 的参数梯度（包括 log_std），以各参数样本方差之和估计协方差迹。"
                  "这些轨迹不参与训练或模型选择。", "",
                  "| Baseline 种子 | 轨迹数 | 探针平均回报 | 无 baseline 方差迹 | 有 baseline 方差迹 | 有 / 无 比值 |",
                  "|---|---:|---:|---:|---:|---:|"])
    for probe in summary["gradient_probes"]:
        without = _finite(probe.get("without_baseline_variance_trace"))
        with_ = _finite(probe.get("with_baseline_variance_trace"))
        lines.append(f"| {probe['seed']} | {_fmt(probe.get('trajectory_count'), 0)} | {_fmt(probe.get('mean_return'))} | "
                     f"{'—' if without is None else f'{without:.6g}'} | {'—' if with_ is None else f'{with_:.6g}'} | {_fmt(probe.get('variance_ratio'), 4)} |")
    ratio = summary["gradient_variance_ratio"]
    lines.extend(["", f"有有效诊断的 {ratio['n']} 个种子中，{summary['gradient_variance_reduced_seeds']} 个比值小于 1；"
                  f"比值中位数 {_fmt(ratio['median'], 4)}，范围 {_fmt(ratio['min'], 4)}–{_fmt(ratio['max'], 4)}。", "",
                  "这是最终 baseline 策略上的局部诊断，不是两个不同训练策略的梯度方差比较，也不能证明整个训练过程始终减方差。"
                  "低 TD loss、较低回报标准差都不等同于梯度方差降低。当前 critic 对时间限制继续 bootstrap，"
                  "actor 则使用截至回合结束的回报，因此 critic 值超过单回合回报上限本身不构成实现错误。", "",
                  "![梯度方差与退化诊断](diagnostics.png)", "", "## 解释范围与复现", "",
                  f"这是 {expected} 个训练种子、一个环境和当前超参数下的对照，结论仅限本次设置。"
                  "不进行显著性或普遍优越性的断言；全部种子的结果均已保留。"])
    errors = [(r["variant"], r["seed"], r["trial"].get("error")) for r in runs if r["trial"].get("error")]
    if errors:
        lines.extend(["", "失败记录："])
        for variant, seed, error in errors:
            lines.append(f"- {LABELS[variant]} seed {seed}：`{str(error).replace('`', '')}`")
    execution = _read_json(output / "execution.json", {})
    if _finite(execution.get("elapsed_seconds")) is not None:
        lines.extend(["", f"整场实验墙钟时间：{_fmt(execution['elapsed_seconds'])} 秒（来自 `execution.json`）。"])
    lines.extend(["", "主要文件：`protocol.json` 保存固定协议；`frozen_models.json` 保存测试前模型哈希；"
                  "各运行目录保存 `training.csv`、`validation.jsonl`、`trial.json`、`best.pt`、`last.pt`、"
                  "`test.json`，baseline 目录额外保存 `gradient_probe.json`；`summary.json` 保存机器可读汇总。", "",
                  "`snapshots/` 保存两份原算法和实验程序的源码副本，用于核对本次执行版本；复现命令从仓库根目录运行，"
                  "应先核对源码哈希及依赖版本。原算法文件未为本实验修改。报告脚本如在训练后完善，以最终保存的报告脚本快照和生成哈希为准。", "",
                  "在一个尚不存在的新输出目录重新执行本次协议：", "", "```bash",
                  ".venv/bin/python algorithms/policy_based/reinforce/experiments/learned_baseline_comparison/run.py run \\",
                  "  --output runs/learned_baseline_comparison/reproduction \\",
                  f"  --seeds {' '.join(str(s) for s in protocol['seeds'])} --episodes {protocol['episodes']} --workers {protocol.get('workers', 3)} \\",
                  f"  --validation-interval {protocol.get('validation_interval', 100)} --validation-episodes {len(protocol.get('validation_seeds', []))} \\",
                  f"  --test-episodes {len(protocol.get('test_seeds', []))} --probe-episodes {len(protocol.get('probe_seeds', []))}",
                  "```", "", "仅重新生成本报告，或重新评估已经冻结的模型：", "", "```bash",
                  f".venv/bin/python algorithms/policy_based/reinforce/experiments/learned_baseline_comparison/run.py report '{output}'",
                  f".venv/bin/python algorithms/policy_based/reinforce/experiments/learned_baseline_comparison/run.py evaluate '{output}' --workers {protocol.get('workers', 3)}",
                  "```", "", "依赖版本：", "", "```json", json.dumps(protocol.get("versions", {}), ensure_ascii=False, indent=2),
                  "```", "", "算法源码 SHA-256（对应本次实验保存的源码快照）：", "", "```json",
                  json.dumps(protocol.get("source_hashes", {}), ensure_ascii=False, indent=2), "```", ""])
    (output / "REPORT.md").write_text("\n".join(lines))


def generate_report(output: Path) -> dict:
    """Write plots, a Chinese report, and seed-level summaries to ``output``."""
    output = Path(output).resolve()
    protocol = _read_json(output / "protocol.json")
    if protocol is None:
        raise FileNotFoundError(output / "protocol.json")
    runs = _load_runs(output, protocol)
    summary = _summarize(runs, protocol)
    plt = _plot_style()
    _learning_curves(output, runs, protocol, plt)
    _final_comparison(output, runs, protocol, plt)
    _diagnostics(output, runs, protocol, plt)
    _write_report(output, runs, protocol, summary)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = generate_report(args.output)
    print(json.dumps({"report": str(args.output / "REPORT.md"), "paired_difference": result["paired_difference"]}, ensure_ascii=False))
