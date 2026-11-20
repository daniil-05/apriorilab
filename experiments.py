"""experiments.py - эксперименты для отчёта: время и число наборов по порогам.
Выход: results/summary.csv, results/itemsets_<порог>.csv, results/fig_time.png, results/fig_lengths.png"""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from apriori import load_transactions, run_experiments, sort_itemsets

THRESHOLDS = [0.01, 0.03, 0.05, 0.10, 0.15]
os.makedirs("results", exist_ok=True)
tx = load_transactions("data/baskets.csv")
exp = run_experiments(tx, THRESHOLDS, repeats=21)

with open("results/summary.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["порог_%", "время_мс", "всего", "k=1", "k=2", "k=3"])
    for e in exp:
        bl = e["by_length"]
        w.writerow([round(e["threshold"]*100), round(e["seconds"]*1000, 2), e["total"], bl.get(1, 0), bl.get(2, 0), bl.get(3, 0)])
for e in exp:
    with open(f"results/itemsets_{round(e['threshold']*100)}pct.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["набор", "длина", "поддержка_%"])
        for s, sup in sort_itemsets(e["result"], "support"):
            w.writerow([", ".join(s), len(s), round(sup*100, 2)])

labels = [f"{round(e['threshold']*100)}%" for e in exp]
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(labels, [e["seconds"]*1000 for e in exp], "o-", color="#1f77b4", lw=2)
for x, e in zip(labels, exp): ax.annotate(f"{e['seconds']*1000:.1f}", (x, e["seconds"]*1000), textcoords="offset points", xytext=(0, 7), ha="center")
ax.set_xlabel("Порог поддержки"); ax.set_ylabel("Время работы, мс"); ax.set_title("Время работы Apriori (baskets.csv, медиана 21 запуска)")
ax.set_ylim(0, max(e["seconds"] for e in exp)*1000*1.2); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig("results/fig_time.png", dpi=200); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4))
maxk = max(max(e["by_length"]) for e in exp); width = 0.8/maxk
for k in range(1, maxk+1):
    vals = [e["by_length"].get(k, 0) for e in exp]
    bars = ax.bar([i + (k-1)*width for i in range(len(exp))], vals, width, label=f"длина {k}")
    for b, v in zip(bars, vals):
        if v: ax.annotate(str(v), (b.get_x()+b.get_width()/2, v), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
ax.set_xticks([i + width*(maxk-1)/2 for i in range(len(exp))]); ax.set_xticklabels(labels)
ax.set_xlabel("Порог поддержки"); ax.set_ylabel("Число частых наборов"); ax.set_title("Число частых наборов разной длины")
ax.legend(); ax.grid(axis="y", alpha=.3); fig.tight_layout(); fig.savefig("results/fig_lengths.png", dpi=200); plt.close(fig)
with open("results/summary.csv", encoding="utf-8") as f:
    print(f.read())
