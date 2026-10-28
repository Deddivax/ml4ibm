from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "outputs"
    / "presentations"
    / "media"
    / "rich_sdf_transformer_architecture.png"
)


def add_box(axis, x, y, width, height, title, body, color, text_size=12):
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.010,rounding_size=0.012",
        linewidth=1.1,
        edgecolor="#d9dce2",
        facecolor=color,
    )
    axis.add_patch(box)
    axis.text(x + width / 2, y + height - 0.032, title, ha="center", va="top", color="#ffffff", fontsize=14, fontweight="bold")
    axis.text(x + width / 2, y + height / 2 - 0.025, body, ha="center", va="center", color="#f7f8fa", fontsize=text_size, linespacing=1.6)


def arrow(axis, x0, y0, x1, y1):
    axis.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops={"arrowstyle": "->", "color": "#9ba5b1", "lw": 1.6})


def main():
    figure, axis = plt.subplots(figsize=(16, 10), facecolor="#f4f5f7")
    axis.set_facecolor("#f4f5f7")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    axis.text(0.5, 0.955, "Rich-SDF Transformer Stiffness Predictor", ha="center", va="center", color="#18212b", fontsize=24, fontweight="bold")
    axis.text(0.5, 0.917, "256-dimensional tokens  |  8 pre-norm encoder layers  |  8 heads  |  1024-dimensional FFN", ha="center", va="center", color="#536170", fontsize=12)

    add_box(axis, 0.04, 0.60, 0.205, 0.20, "1. Rich SDF", "$N=27$ or $64$ lattice values\n\n$\\tilde d_i=(d_i-\\mu_i)/\\sigma_i$\n\n$c_i=(x_i,y_i,z_i)\\in[-1,1]^3$", "#48535e")
    add_box(axis, 0.29, 0.60, 0.205, 0.20, "2. Input Tokens", "$t_i=[\\tilde d_i,c_i]\\in\\mathbb{R}^{4}$\n\n$x_i^{(0)}=W_{node}t_i+b_{node}+p_i$\n\n$W_{node}:\\mathbb{R}^{4}\\rightarrow\\mathbb{R}^{256}$", "#1865a2")
    add_box(axis, 0.54, 0.53, 0.265, 0.34, "3. Transformer Encoder x 8", "Pre-norm residual block\n\n$u=x+\\mathrm{MHA}(\\mathrm{LN}(x))$\n$x_{next}=u+\\mathrm{FFN}(\\mathrm{LN}(u))$\n\n$\\mathrm{head}_j=\\mathrm{softmax}(Q_jK_j^\\top/\\sqrt{32})V_j$\n\n$\\mathrm{FFN}(h)=W_2\\,\\mathrm{ReLU}(W_1h+b_1)+b_2$\n$256\\rightarrow1024\\rightarrow256$", "#493b91", 11)
    add_box(axis, 0.85, 0.60, 0.115, 0.20, "4. Readout", "$m=\\frac{1}{N}\\sum_i x_i^{(8)}$\n\n$z=\\mathrm{SiLU}(W_r\\,\\mathrm{LN}(m)+b_r)$\n\nplus ordered-SDF\nSiLU residual", "#9b5717", 9.5)

    arrow(axis, 0.245, 0.70, 0.29, 0.70)
    arrow(axis, 0.495, 0.70, 0.54, 0.70)
    arrow(axis, 0.805, 0.70, 0.85, 0.70)

    add_box(axis, 0.10, 0.20, 0.26, 0.20, "5a. PSD Matrix Head", "$r=W_Mz+b_M\\in\\mathbb{R}^{171}$\n\n$L_{ii}=\\mathrm{softplus}(r_i)+10^{-6}$\n$L_{ij}=r_{ij},\\ i>j$\n\n$K_{18}=LL^\\top$", "#8b3051", 10.5)
    add_box(axis, 0.41, 0.20, 0.20, 0.20, "5b. Scale Head", "$\\log s=W_sz+b_s$\n\n$\\text{Predicted scale}=\\exp(\\log s)$", "#167164", 11)
    add_box(axis, 0.66, 0.16, 0.27, 0.28, "6. Constrained Output", "$K_{24}=TK_{18}T^\\top$\n$T\\in\\mathbb{R}^{24\\times18}$\n\n$K_{unit}=K_{24}/\\max(\\lVert K_{24}\\rVert_F,10^{-6})$\n\n$K_{24}R=0$ for six rigid-body modes\n\noutput: $(K_{unit},\\log s)$", "#48535e", 10.5)

    arrow(axis, 0.905, 0.60, 0.23, 0.40)
    arrow(axis, 0.905, 0.60, 0.51, 0.40)
    arrow(axis, 0.36, 0.30, 0.66, 0.30)
    arrow(axis, 0.61, 0.30, 0.66, 0.30)

    axis.text(0.5, 0.065, "Learned positional embeddings $p_i$ are added directly after the node projection. Dropout is 0. The only activations are softmax, ReLU, SiLU, and softplus as shown.", ha="center", va="center", color="#536170", fontsize=11)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=200, bbox_inches="tight", facecolor=figure.get_facecolor())
    print(OUTPUT)


if __name__ == "__main__":
    main()
