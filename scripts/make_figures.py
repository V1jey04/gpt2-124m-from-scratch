"""Draw the README figures from the real training logs and benchmark results."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG, SURF, INK, MUTED, GRID, ACC, GREY = "#0f0f0f", "#161616", "#ffffff", "#8f8f8f", "#262626", "#dafa9c", "#5a5a5a"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": SURF, "savefig.facecolor": BG,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "font.size": 12, "axes.titlesize": 15, "axes.titleweight": "bold",
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
})

PRE = [(0,10.9559),(250,6.2592),(500,5.5077),(750,4.8391),(1000,4.3325),(1250,4.1349),(1500,3.9938),(1750,3.8909),(2000,3.8198),(2250,3.77),(2500,3.7194),(2750,3.6793),(3000,3.6434),(3250,3.618),(3500,3.5844),(3750,3.5564),(4000,3.5345),(4250,3.5183),(5250,3.4494),(5500,3.4288),(5750,3.4133),(6000,3.4041),(6250,3.3872),(6500,3.3745),(6750,3.3632),(7000,3.3577),(7250,3.3431),(7500,3.3322),(7750,3.3257),(8000,3.3205),(8250,3.3098),(8500,3.3038),(8750,3.3014),(9000,3.297)]
SFT = [(0,2.7371),(200,1.8048),(400,1.7024),(600,1.6471),(800,1.6119),(1000,1.5827),(1200,1.5609),(1400,1.5433),(1600,1.5279),(1800,1.5124),(2000,1.5008),(2200,1.492),(2400,1.4834),(2600,1.4755),(2800,1.469),(3000,1.4628),(3200,1.4585),(3400,1.4545),(3600,1.4515),(3800,1.4487),(4000,1.4466),(4217,1.4452)]
TPS = 262144

fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1.35, 1]})
x = [s * TPS / 1e9 for s, _ in PRE]; y = [v for _, v in PRE]
a.plot(x, y, color=ACC, lw=2.4); a.fill_between(x, y, 2, color=ACC, alpha=0.07)
r = 5066 * TPS / 1e9
a.axvline(r, color="#555", ls="--", lw=1.2); a.text(r + 0.03, 10.2, "session 2 resumes\nfrom checkpoint", color=MUTED, fontsize=10, va="top")
a.set_title("Pretraining: FineWeb-Edu, 2.42B tokens", loc="left"); a.set_xlabel("tokens seen (billions)"); a.set_ylabel("validation loss")
a.set_ylim(2.5, 11.5); a.set_xlim(0, 2.42)
a.annotate("3.297", (x[-1], y[-1]), xytext=(-46, 22), textcoords="offset points", color=ACC, fontsize=12, fontweight="bold")
x2 = [s for s, _ in SFT]; y2 = [v for _, v in SFT]
b.plot(x2, y2, color=ACC, lw=2.4); b.fill_between(x2, y2, 1.2, color=ACC, alpha=0.07)
b.set_title("Chat fine-tuning: 276M tokens", loc="left"); b.set_xlabel("step"); b.set_ylabel("validation loss (reply tokens)")
b.set_ylim(1.2, 2.9); b.set_xlim(0, 4217)
b.annotate("1.445", (x2[-1], y2[-1]), xytext=(-46, 16), textcoords="offset points", color=ACC, fontsize=12, fontweight="bold")
fig.tight_layout(); fig.savefig("docs/training.png", dpi=160); plt.close(fig)

res = json.load(open("benchmark_results.json"))
ours = res["ours (124M, 2.4B tok FineWeb-Edu + chat)"]; g2 = res["OpenAI GPT-2 124M"]
tasks = [("ARC-Easy", "arc_easy", 0.25), ("HellaSwag", "hellaswag", 0.25), ("LAMBADA", "lambada", None)]
fig, ax = plt.subplots(figsize=(10, 4.2))
ax.grid(axis="y", visible=False); ax.set_axisbelow(True)
for i, (name, k, chance) in enumerate(tasks):
    yy = len(tasks) - 1 - i
    ax.barh(yy + 0.18, ours[k] * 100, 0.32, color=ACC)
    ax.barh(yy - 0.18, g2[k] * 100, 0.32, color=GREY)
    ax.text(ours[k] * 100 + 0.6, yy + 0.18, f"{ours[k]*100:.1f}%", va="center", color=ACC, fontsize=11, fontweight="bold")
    ax.text(g2[k] * 100 + 0.6, yy - 0.18, f"{g2[k]*100:.1f}%", va="center", color=MUTED, fontsize=11)
    if chance:
        ax.plot([chance * 100] * 2, [yy - 0.42, yy + 0.42], color="#777", ls="--", lw=1)
ax.set_yticks(range(len(tasks))); ax.set_yticklabels([t[0] for t in tasks][::-1], color=INK, fontsize=13)
ax.set_xlim(0, 50); ax.set_xlabel("accuracy, zero-shot (%)   ·   dashed line: random chance")
ax.set_title("This model (lime) vs OpenAI GPT-2 124M (grey), same scoring script", loc="left")
fig.tight_layout(); fig.savefig("docs/benchmarks.png", dpi=160); plt.close(fig)
print("ok")
