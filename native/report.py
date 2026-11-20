"""Summarize the matched autonomous runs and plot their heat and retention."""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def array(rows, name):
    return np.asarray([row[name] for row in rows], dtype=float)


def measure(rows, begin, end):
    rows = [row for row in rows if begin <= row["s"] <= end]
    time = array(rows, "s")
    heat = array(rows, "heat")
    power = (heat[-1] - heat[0]) / (time[-1] - time[0])
    b = array(rows, "B")
    radius = array(rows, "R")
    result = dict(
        window=[float(time[0]), float(time[-1])], samples=len(rows),
        core_resolved_fraction=float(np.isfinite(b).mean()),
        final_B=rows[-1]["B"], final_R=rows[-1]["R"],
        mean_B=float(np.nanmean(b)) if np.isfinite(b).any() else None,
        mean_R=float(np.nanmean(radius)) if np.isfinite(radius).any() else None,
        mean_power=float(power[:4].sum()),
        mean_far_power=float(power[3]), mean_moving_edge_power=float(power[4]),
    )
    for name in ("edge", "far"):
        inward, outward = array(rows, f"u_{name}").T * np.array([[-1], [1]])
        both = (inward > 1e-5) & (outward > 1e-5)
        result[f"{name}_duty_both_gt_1e-5"] = float(both.mean())
        result[f"{name}_mean_sum_speed_when_both"] = (
            float(np.mean(inward[both] + outward[both])) if both.any() else None
        )
    return result


def summarize(runs):
    summary = {}
    for name, run in runs.items():
        rows = run["rows"]
        start, end = rows[0]["s"], rows[-1]["s"]
        heat = array(rows, "heat")
        total = float(heat[-1, :4].sum())
        assert np.min(np.diff(heat, axis=0)) > -1e-8
        assert np.min(array(rows, "power")) >= 0
        result = dict(
            cells=run["cells"], kappa=run["kappa"], completed=end == run["stop"],
            window=[start, end], deposited_heat=total,
            percent_of_initial_energy=100 * total / run["initial_energy"],
            heat_by_fixed_region=heat[-1, :4].tolist(),
            heat_within_moving_edge_band=float(heat[-1, 4]),
            fraction_in_inner=heat[-1, 0] / total if total else None,
            fraction_in_far=heat[-1, 3] / total if total else None,
            max_energy_error=max(row["energy_error"] for row in rows),
            max_content_error=max(row["content_error"] for row in rows),
            max_rhs_power_error=max(check["power_balance_error"] for check in run["checks"]),
            archive_comparison=run["archive_comparison"],
            first20=measure(rows, start, start+20),
            final20=measure(rows, end-20, end),
        )
        if end >= 200:
            result["s180_to_200"] = measure(rows, 180, 200)
        summary[name] = result
    return summary


def plot(runs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10.9, 7.1), layout="constrained")
    fig.suptitle("Autonomous RGT geometry with counterflow heat transfer", fontsize=16)
    colors = {0: "#616b76", 1: "#007b83", 10: "#5e49a0", 100: "#ba561a"}
    for run in runs.values():
        rows = run["rows"]
        time = array(rows, "s")
        heat = array(rows, "heat")
        kappa = run["kappa"]
        cells = run["cells"]
        style = dict(color=colors[kappa], lw=1.25, linestyle="-" if cells == 256 else "--")
        label = f"κ={kappa:g}, {cells} cells"
        axes[0, 0].plot(time, array(rows, "B"), label=label, **style)
        # Resample before smoothing to use the same 10-unit window on both legs.
        step = np.arange(time[0], time[-1]+.05, .1)
        radius = np.interp(step, time, array(rows, "R"))
        radius_average = np.convolve(radius, np.ones(101)/101, mode="valid")
        axes[0, 1].plot(step[50:-50], radius_average, **style)
        if kappa:
            accumulated = np.interp(step, time, heat[:, :4].sum(axis=1))
            window_power = (accumulated[100:] - accumulated[:-100]) / 10
            axes[1, 0].plot(step[50:-50], window_power, **style)
        if kappa == 100:
            axes[1, 1].plot(time, heat[:, 0], color="#ba561a", ls=style["linestyle"],
                            label=f"Inner, {cells}")
            axes[1, 1].plot(time, heat[:, 4], color="#007b83", ls=style["linestyle"],
                            label=f"Moving edge band, {cells}")
            axes[1, 1].plot(time, heat[:, 3], color="#5e49a0", ls=style["linestyle"],
                            label=f"Far, {cells}")
    titles = ("Signed core retention", "Core radius · 10-unit mean",
              "Heat deposition power · 10-unit mean", "Deposited heat by location · κ=100")
    units = ("Retention B", "Areal radius R", "Energy per model time", "Energy")
    for ax, title, unit in zip(axes.flat, titles, units):
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
        ax.set_xlabel("Model time s")
        ax.set_ylabel(unit)
        ax.grid(axis="y", alpha=.23)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(ncols=2, fontsize=7, frameon=False, loc="upper left")
    axes[1, 0].set_yscale("log")
    axes[1, 1].legend(ncols=2, fontsize=7, frameon=False, loc="center right")
    fig.savefig(HERE / "autonomous_heat.png", dpi=190)


if __name__ == "__main__":
    runs = {path.stem: json.loads(path.read_text()) for path in sorted((HERE / "runs").glob("n*_k*.json"))}
    summary = summarize(runs)
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    plot(runs)
