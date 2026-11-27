"""Chương 1 - Nền tảng: "bức tường hàm mũ" và nguyên lý biến phân.

Hình 1: số con số cần lưu để ghi hàm sóng N electron trên lưới.
Hình 2: nguyên lý biến phân với nguyên tử He (2 electron) - ví dụ khởi động
        giải được bằng tay trước khi sang Li (3 electron).

Chạy:  python 01_nen_tang.py
"""
import numpy as np
import matplotlib.pyplot as plt

from common import (C, TEXT, TEXT2, NEUTRAL, E_EXACT_HE, setup_style, save_fig,
                    save_json)

setup_style()

# ---------------------------------------------------------------------------
# 1) Bức tường hàm mũ
# Hàm sóng N electron là hàm của 3N tọa độ. Nếu mỗi tọa độ chỉ lấy 10 điểm
# lưới, ta cần 10^(3N) con số. Mỗi số thực "double" chiếm 8 byte.
# ---------------------------------------------------------------------------
print("1) Bức tường hàm mũ")
N = np.arange(1, 13)
log10_numbers = 3 * N  # log10(10^(3N))
for n in (1, 2, 3, 6, 10):
    gb = 10.0 ** (3 * n) * 8 / 1e9
    print(f"  N = {n:2d} electron: 10^{3*n} số  ~ {gb:.3g} GB")

fig, ax = plt.subplots(figsize=(6.4, 3.8))
ax.plot(N, log10_numbers, "-o", color=C["blue"], markersize=6, zorder=3,
        markeredgecolor="white", markeredgewidth=1.5)
refs = [  # (giá trị log10, nhãn, đặt nhãn bên phải hay trái)
    (9 + np.log10(16 / 8), "RAM 16 GB của một laptop", "right"),
    (np.log10(1.8e23 / 8), "toàn bộ dữ liệu số của loài người (~10$^{23}$ byte)", "left"),
    (80, "số nguyên tử trong vũ trụ quan sát được (~10$^{80}$)", "right"),
]
for y, label, side in refs:
    ax.axhline(y, color=NEUTRAL, lw=1, ls="--", zorder=1)
    x = 12.3 if side == "right" else 0.7
    ax.text(x, y + 0.8, label, ha=side, va="bottom", fontsize=8.5, color=TEXT2)
labels = {1: "H", 2: "He", 3: "Li", 6: "C", 10: "Ne"}
for n, el in labels.items():
    ax.annotate(el, (n, 3 * n), textcoords="offset points", xytext=(-12, 6),
                fontsize=9, color=TEXT, fontweight="bold")
ax.set_xlabel("Số electron N")
ax.set_ylabel("log$_{10}$(số con số phải lưu)")
ax.set_title("Lưu hàm sóng trên lưới 10 điểm/chiều cần 10$^{3N}$ con số", loc="left")
ax.set_xticks(N)
ax.set_ylim(0, 92)
save_fig(fig, "01_buc_tuong_ham_mu")

# ---------------------------------------------------------------------------
# 2) Nguyên lý biến phân với He
# Hàm thử: psi = exp(-zeta*(r1 + r2)) (hai electron 1s, "điện tích hiệu dụng" zeta).
# Tính giải tích (xem báo cáo):  E(zeta) = zeta^2 - 2*Z*zeta + (5/8)*zeta
# Cực tiểu: zeta* = Z - 5/16 = 27/16  ->  E* = -(27/16)^2 = -2.8477 Ha.
# ---------------------------------------------------------------------------
print("2) Nguyên lý biến phân cho He")
Z = 2.0
zeta = np.linspace(1.2, 2.4, 400)
E = zeta**2 - 2 * Z * zeta + 5 / 8 * zeta
zeta_opt = Z - 5 / 16
E_opt = -(zeta_opt**2)
E_no_screen = Z**2 - 2 * Z * Z + 5 / 8 * Z  # zeta = Z = 2 (không có che chắn)
E_no_repulsion = -Z**2  # bỏ hẳn lực đẩy giữa 2 electron: 2 * (-Z^2/2)
print(f"  zeta tối ưu = {zeta_opt:.4f},  E = {E_opt:.6f} Ha")
print(f"  zeta = Z = 2 (không che chắn): E = {E_no_screen:.6f} Ha")
print(f"  bỏ lực đẩy e-e: E = {E_no_repulsion:.6f} Ha")
print(f"  chính xác: E = {E_EXACT_HE:.6f} Ha;  sai số = {1000*(E_opt-E_EXACT_HE):.1f} mHa")

fig, ax = plt.subplots(figsize=(6.4, 3.8))
ax.plot(zeta, E, color=C["blue"], label="E(ζ) của hàm thử")
ax.axhline(E_EXACT_HE, color=C["orange"], lw=2, ls="--",
           label=f"năng lượng chính xác  {E_EXACT_HE:.4f} Ha")
ax.plot([zeta_opt], [E_opt], "o", color=C["blue"], markersize=8,
        markeredgecolor="white", markeredgewidth=1.5, zorder=4)
ax.annotate(f"cực tiểu: ζ = 27/16 = {zeta_opt:.4f}\nE = {E_opt:.4f} Ha",
            (zeta_opt, E_opt), xytext=(1.40, -2.53), textcoords="data",
            fontsize=9, color=TEXT, arrowprops=dict(arrowstyle="-", color=TEXT2, lw=0.8))
ax.annotate(f"ζ = Z = 2 (không che chắn)\nE = {E_no_screen:.4f} Ha", (2.0, E_no_screen),
            textcoords="offset points", xytext=(14, -40), fontsize=9, color=TEXT2,
            arrowprops=dict(arrowstyle="-", color=TEXT2, lw=0.8))
ax.plot([2.0], [E_no_screen], "o", color=NEUTRAL, markersize=6, zorder=4)
ax.fill_between(zeta, E_EXACT_HE, E_EXACT_HE - 0.25, color="#efefef", zorder=0)
ax.text(1.22, E_EXACT_HE - 0.13, "vùng cấm: không hàm thử nào xuống được đây",
        fontsize=8.5, color=TEXT2)
ax.set_xlabel("Điện tích hiệu dụng ζ (tham số biến phân)")
ax.set_ylabel("Năng lượng (Ha)")
ax.set_title("Nguyên lý biến phân với nguyên tử He: E(ζ) luôn nằm trên E chính xác",
             loc="left")
ax.set_ylim(E_EXACT_HE - 0.25, -2.45)
ax.legend(loc="lower right", fontsize=8.5)
save_fig(fig, "01_bien_phan_he")

# ---------------------------------------------------------------------------
# 3) "Bậc thang" năng lượng của Li
# Hàm thử: định thức |1s↑ 1s↓ 2s↑| với orbital kiểu hydro, CÙNG điện tích hiệu dụng zeta.
# E(zeta) = 9/8 zeta^2 - 27/4 zeta + (5/8 + 34/81 - 16/729) zeta      (Z = 3)
#   9/8 zeta^2   : động năng (2 electron 1s: zeta^2/2 mỗi cái; 1 electron 2s: zeta^2/8)
#   -27/4 zeta   : hút hạt nhân (-Z zeta cho mỗi 1s, -Z zeta/4 cho 2s)
#   5/8 zeta     : đẩy 1s-1s (J_1s1s);  34/81 zeta = 2 J_1s2s;  -16/729 zeta = -K_1s2s (trao đổi)
# ---------------------------------------------------------------------------
print("3) Bậc thang năng lượng của Li")
from common import E_EXACT_LI, E_HF_LIMIT_LI  # noqa: E402
k_ee = 5 / 8 + 34 / 81 - 16 / 729
z_li = (27 / 4 - k_ee) * 4 / 9
E_li_var = 9 / 8 * z_li**2 - 27 / 4 * z_li + k_ee * z_li
E_li_first = 9 / 8 * 9 - 27 / 4 * 3 + k_ee * 3          # zeta = Z = 3: nhiễu loạn bậc 1
E_li_zero = -81 / 8                                       # bỏ hẳn đẩy e-e: 3 bài toán hydro
z_boson = 19 / 8                                          # cả 3 electron vào 1s (vi phạm Pauli)
E_boson = 3 / 2 * z_boson**2 - 9 * z_boson + 15 / 8 * z_boson
assert E_li_var > E_HF_LIMIT_LI > E_EXACT_LI              # đúng thứ tự của nguyên lý biến phân
levels = [
    (E_li_zero, "bỏ hẳn lực đẩy e–e\n(3 bài toán hydro)", NEUTRAL, "--"),
    (E_boson, "\"bẫy boson\": cả 3 electron vào 1s\n(cấm với electron - vi phạm Pauli)", C["red"], ":"),
    (E_li_first, "nhiễu loạn bậc 1 (ζ = Z = 3)", C["violet"], "-"),
    (E_li_var, f"biến phân 1 tham số (ζ* = {z_li:.4f})", C["blue"], "-"),
    (E_HF_LIMIT_LI, "giới hạn Hartree–Fock", C["orange"], "-"),
    (E_EXACT_LI, "chính xác (Hylleraas)", TEXT, "-"),
]
for E_, lab, _, _ in levels:
    print(f"  {E_:10.5f} Ha  {lab.replace(chr(10), ' ')}")
fig, (ax, az) = plt.subplots(1, 2, figsize=(9.0, 4.3), gridspec_kw={"width_ratios": [1, 1.25]})
# (a) toàn cảnh
for E_, lab, col, ls in levels:
    ax.hlines(E_, 0, 1, color=col, lw=2.2, ls=ls)
for E_, lab, col, ls in levels[:2]:
    ax.text(1.05, E_, f"{E_:.3f}\n" + lab, va="center", fontsize=7.8, color=TEXT2)
ax.axhspan(-7.52, -7.0, color="#efefef", zorder=0)
ax.text(1.05, -7.26, "phóng to\nở hình (b) →", va="center", fontsize=8, color=TEXT2)
ax.set_xlim(0, 2.6)
ax.set_ylim(-10.4, -6.8)
ax.set_xticks([])
ax.set_ylabel("Năng lượng (Ha)")
ax.set_title("(a) Toàn cảnh", loc="left")
ax.grid(axis="x", visible=False)
ax.spines["bottom"].set_visible(False)
# (b) phóng to vùng -7.0 ... -7.5
for E_, lab, col, ls in levels[2:]:
    az.hlines(E_, 0, 1, color=col, lw=2.2, ls=ls)
    dy = {E_HF_LIMIT_LI: 0.012, E_EXACT_LI: -0.012}.get(E_, 0.0)
    az.text(1.05, E_ + dy, f"{E_:.4f}  " + lab, va="center", fontsize=8.3, color=TEXT)
az.annotate("", xy=(0.5, E_EXACT_LI), xytext=(0.5, E_HF_LIMIT_LI),
            arrowprops=dict(arrowstyle="<->", color=C["green"], lw=1.2, shrinkA=0, shrinkB=0))
az.text(0.42, 0.5 * (E_EXACT_LI + E_HF_LIMIT_LI),
        f"tương quan: {1000*(E_EXACT_LI-E_HF_LIMIT_LI):.1f} mHa", fontsize=8,
        color=C["green"], va="center", ha="right")
az.set_xlim(-0.75, 3.0)
az.set_ylim(-7.52, -7.0)
az.set_xticks([])
az.set_title("(b) Phóng to: các mức gần đáp án", loc="left")
az.grid(axis="x", visible=False)
az.spines["bottom"].set_visible(False)
fig.tight_layout()
save_fig(fig, "01_bac_thang_li")

save_json(
    {
        "li_ladder": {"E_no_ee": E_li_zero, "E_first_order": E_li_first,
                      "zeta_opt": z_li, "E_variational": E_li_var, "E_boson_trap": E_boson,
                      "zeta_boson": z_boson, "E_HF_limit": E_HF_LIMIT_LI, "E_exact": E_EXACT_LI},
        "he_variational": {
            "zeta_opt": zeta_opt,
            "E_opt": E_opt,
            "E_zeta_equals_Z": E_no_screen,
            "E_no_repulsion": E_no_repulsion,
            "E_exact": E_EXACT_HE,
            "error_mHa": 1000 * (E_opt - E_EXACT_HE),
        },
        "grid_numbers_log10": {int(n): int(3 * n) for n in N},
    },
    "01_nen_tang",
)
