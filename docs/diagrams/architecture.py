"""Draws docs/diagrams/architecture_as_built.png (as-built architecture, Final Project)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

TEAL, ORANGE, INK, MUTED, FILL, BG = "#00927F", "#D9691A", "#1B2430", "#5B6B7A", "#E3F2EF", "#F3F6F7"
cols = [
    ("1  Data pipeline", "Module 3 · Prefect in Docker", ["Olist public data\n9 CSV files", "Clean · integrate\npurchase-time features",
                                                          "Pseudonymise\nsalted SHA-256", "Quality gate (35 checks)\n+ bias check", "Model-ready table"]),
    ("2  Model", "Module 4 · MLflow", ["Train and tune\nLR · RF · XGBoost", "SHAP · DiCE · Fairlearn\nmodel card",
                                        "Registry\nchampion v3", "Monthly retraining\nwith promotion gates"]),
    ("3  Serve", "Modules 4–5 + Final", ["FastAPI /predict\nscore · reasons · actions", "Explanation store\naudit trail",
                                          "Streamlit dashboard\nXAI · what-if · fairness"]),
    ("4  Monitor", "Final Project", ["Prometheus + Grafana\nlive API signals", "Weekly monitor (reads model-ready\ntable) · drift · fairness · Evidently",
                                      "Monitoring site\nGitHub Pages", "Alerts → incident plan\n→ retraining"]),
]
W, H, GAP, X0 = 3.0, 0.82, 0.32, 0.2
fig, ax = plt.subplots(figsize=(13.6, 6.6))
ax.set_xlim(0, 4 * (W + 0.45) + 0.1)
ax.set_ylim(-1.35, 6.25)
ax.axis("off")
centers = {}
for i, (title, sub, boxes) in enumerate(cols):
    x = X0 + i * (W + 0.45)
    ax.add_patch(FancyBboxPatch((x - 0.12, -0.25), W + 0.24, 6.3, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=BG, ec="#D5DDE2", lw=1))
    ax.text(x + W / 2, 5.78, title, ha="center", va="center", fontsize=14, fontweight="bold", color=INK)
    ax.text(x + W / 2, 5.43, sub, ha="center", va="center", fontsize=10, color=MUTED)
    for j, label in enumerate(boxes):
        y = 4.75 - j * (H + GAP)
        hl = (i, j) in {(1, 2), (2, 0), (3, 1)}
        ax.add_patch(FancyBboxPatch((x, y - H / 2), W, H, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc="white" if not hl else FILL, ec=TEAL, lw=1.6 if hl else 1.1))
        ax.text(x + W / 2, y, label, ha="center", va="center", fontsize=10.5, color=INK, linespacing=1.35)
        centers[(i, j)] = (x, y)
        if j and i in (0, 1) and not (i == 1 and j == 3):
            ax.add_patch(FancyArrowPatch((x + W / 2, y + H / 2 + GAP - 0.02), (x + W / 2, y + H / 2 + 0.02),
                                         arrowstyle="-|>", mutation_scale=12, color=MUTED, lw=1.2))


def link(a, b, color=MUTED, style="-", rad=0.0, label=None):
    (xa, ya), (xb, yb) = centers[a], centers[b]
    p = FancyArrowPatch((xa + W, ya), (xb, yb), arrowstyle="-|>", mutation_scale=13, color=color, lw=1.4,
                        linestyle=style, connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(p)
    if label:
        ax.text((xa + W + xb) / 2, (ya + yb) / 2 + 0.5, label, ha="center", fontsize=9, color=color)


link((0, 4), (1, 0), rad=-0.25)
link((1, 2), (2, 0), rad=0.15)
link((1, 2), (2, 2), rad=-0.1)
link((2, 0), (3, 0), rad=0.0, label="/metrics")
ax.add_patch(FancyArrowPatch((centers[(2, 0)][0] + W / 2, centers[(2, 0)][1] - H / 2),
                             (centers[(2, 1)][0] + W / 2, centers[(2, 1)][1] + H / 2),
                             arrowstyle="-|>", mutation_scale=12, color=MUTED, lw=1.2))
ax.add_patch(FancyArrowPatch((centers[(3, 1)][0] + W / 2, centers[(3, 1)][1] - H / 2),
                             (centers[(3, 2)][0] + W / 2, centers[(3, 2)][1] + H / 2),
                             arrowstyle="-|>", mutation_scale=12, color=MUTED, lw=1.2))
ax.add_patch(FancyArrowPatch((centers[(3, 2)][0] + W / 2, centers[(3, 2)][1] - H / 2),
                             (centers[(3, 3)][0] + W / 2, centers[(3, 3)][1] + H / 2),
                             arrowstyle="-|>", mutation_scale=12, color=MUTED, lw=1.2))
# alerts trigger retraining
xa, ya = centers[(3, 3)]
xb, yb = centers[(1, 3)]
ax.add_patch(FancyArrowPatch((xa, ya - 0.15), (xb + W, yb - 0.15), arrowstyle="-|>", mutation_scale=13, color=ORANGE,
                             lw=1.6, linestyle="--", connectionstyle="arc3,rad=0.12"))
ax.text((xa + xb + W) / 2, ya - 0.72, "drift or performance alert triggers retraining; a person approves promotion",
        ha="center", fontsize=9.5, color=ORANGE)
ax.add_patch(FancyBboxPatch((X0, -1.25), 4 * (W + 0.45) - 0.45, 0.62, boxstyle="round,pad=0.02,rounding_size=0.1",
                            fc=INK, ec=INK))
ax.text(X0 + (4 * (W + 0.45) - 0.45) / 2, -0.94,
        "Users: delivery operations · seller support · customer service · Head of Logistics · Responsible AI Review Board",
        ha="center", va="center", fontsize=11, color="white")
fig.tight_layout(pad=0.2)
fig.savefig("docs/diagrams/architecture_as_built.png", dpi=200, facecolor="white")
