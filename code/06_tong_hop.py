"""Chương 8 - Tổng hợp: so sánh mọi phương pháp trên cùng một biểu đồ.

Đọc kết quả của các script 01-05 (thư mục results/), rồi:
  * vẽ hình tổng kết figures/06_tong_ket.pdf
  * sinh các bảng LaTeX cho báo cáo (report/chuong/bang_*.tex), để mọi con số
    trong PDF đều lấy trực tiếp từ kết quả tính toán.

Chạy (sau khi đã chạy 01-05):  python 06_tong_hop.py
"""
import numpy as np
import matplotlib.pyplot as plt

from common import (C, TEXT, TEXT2, NEUTRAL, ROOT, E_EXACT_LI, E_HF_LIMIT_LI, CHEM_ACC_HA,
                    HARTREE_TO_EV, IE_EXP_LI_EV, setup_style, save_fig, load_json)

setup_style()
TAB_DIR = ROOT / "report" / "chuong"
E_CORR = E_EXACT_LI - E_HF_LIMIT_LI

r1 = load_json("01_nen_tang")
r2 = load_json("02_hartree_fock")
r3 = load_json("03_sau_hartree_fock")
r4 = load_json("04_dft")
r5 = load_json("05_qmc")


def mha(e):
    return 1000 * (e - E_EXACT_LI)


def pct(e):
    return 100 * (e - E_HF_LIMIT_LI) / E_CORR


def write(name, text):
    (TAB_DIR / name).write_text(text, encoding="utf-8")
    print(f"  -> đã ghi report/chuong/{name}")


# ===========================================================================
# Bảng 1: các phương pháp sau HF
# ===========================================================================
runs = r3["runs"]
cbs = r3["cbs"]
methods = ["UHF", "MP2", "CISD", "CCSD", "CCSD(T)", "FCI"]
lines = [r"\begin{table}[h]", r"\centering\small",
         r"\caption{Năng lượng của Li (Ha) tính bằng code đi kèm. Trong ngoặc: sai số so với năng "
         r"lượng chính xác (mHa). Dòng cuối: phần trăm năng lượng tương quan thu được ở cc-pCVQZ. "
         r"Cột CBS: HF giới hạn cộng năng lượng tương quan ngoại suy từ T và Q.}",
         r"\label{tab:sauhf}",
         r"\begin{tabular}{@{}lllll@{}}", r"\toprule",
         r"Phương pháp & cc-pCVDZ & cc-pCVTZ & cc-pCVQZ & CBS (T,Q) \\", r"\midrule"]
for m in methods:
    cells = []
    for X in ["2", "3", "4"]:
        e = runs[X][m][0]
        cells.append(f"${e:.5f}$ ({mha(e):.2f})")
    if m in cbs:
        e = cbs[m]["E_tot_CBS"]
        cells.append(f"${e:.5f}$ ({mha(e):.2f})")
    else:
        cells.append("---")
    lines.append(f"{m} & " + " & ".join(cells) + r" \\")
lines += [r"\midrule",
          r"Số hàm cơ sở & " + " & ".join(str(runs[X]["nao"]) for X in ["2", "3", "4"]) + r" & $\infty$ \\",
          r"Số định thức FCI & " + " & ".join(f"{runs[X]['n_det_FCI']:,}".replace(",", r"\,")
                                              for X in ["2", "3", "4"]) + r" & --- \\",
          r"\% tương quan (QZ) & \multicolumn{4}{l}{" + ", ".join(
              f"{m} {pct(runs['4'][m][0]):.1f}\\%" for m in methods[1:]) + r"} \\",
          r"\bottomrule", r"\end{tabular}", r"\end{table}"]
write("bang_sau_hf.tex", "\n".join(lines) + "\n")

core = r3["core_functions"]
lines = [r"\begin{table}[h]", r"\centering\small",
         r"\caption{Full CI cho Li với hai tập cơ sở cỡ QZ: không có và có hàm lõi.}",
         r"\label{tab:loi}", r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
         r"Tập cơ sở & Số hàm & $\Ecorr$ (FCI, mHa) & \% của giá trị đúng \\", r"\midrule"]
for k, nao in [("cc-pVQZ", 55), ("cc-pCVQZ", 84)]:
    v = core[k]
    lines.append(f"{k} & {nao} & ${1000*v['E_corr']:.2f}$ & {100*v['E_corr']/E_CORR:.1f}\\% \\\\")
lines += [r"\midrule", f"Giá trị đúng & $\\infty$ & ${1000*E_CORR:.2f}$ & 100\\% \\\\",
          r"\bottomrule", r"\end{tabular}", r"\end{table}"]
write("bang_loi.tex", "\n".join(lines) + "\n")

# ===========================================================================
# Bảng 2: DFT
# ===========================================================================
ie_exact = (-7.279913412669 - E_EXACT_LI) * HARTREE_TO_EV
lines = [r"\begin{table}[h]", r"\centering\small",
         r"\caption{Li với các phiếm hàm (tập cơ sở aug-cc-pCVQZ, tính bằng code đi kèm). "
         r"Cột cuối: nguyên tử H, nơi đáp án đúng là $-0.5\Ha$; mọi sai số ở đó là lỗi tự tương tác.}",
         r"\label{tab:dft}", r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
         r"Phương pháp & Bậc & $E-E_{\text{chính xác}}$ & IE ($\Delta$SCF) & $-\varepsilon_{\rm HOMO}$ & H: $E+0.5$ \\",
         r" & & (mHa) & (eV) & (eV) & (mHa) \\", r"\midrule"]
hmap = {h["name"]: h for h in r4["h_atom"]}
for row in r4["li"]:
    h = hmap.get(row["name"])
    rung = row["rung"].split(" ", 1)[1] if " " in row["rung"] else row["rung"]
    lines.append(f"{row['name']} & {rung} & ${row['err_mHa']:+.1f}$ & {row['IE_delta_eV']:.3f} & "
                 f"{row['IE_homo_eV']:.3f} & ${h['err_mHa']:+.1f}$ \\\\")
lines += [r"\midrule",
          f"Chính xác / thực nghiệm & & 0 & {ie_exact:.3f} / {IE_EXP_LI_EV:.3f} & {ie_exact:.3f} & 0 \\\\",
          r"\bottomrule", r"\end{tabular}", r"\end{table}"]
write("bang_dft.tex", "\n".join(lines) + "\n")

# ===========================================================================
# Bảng 3: QMC
# ===========================================================================
v = r5["vmc"]
dm = r5["dmc"]
ex = r5["dmc_extrapolated"]
lines = [r"\begin{table}[h]", r"\centering\small",
         r"\caption{Kết quả Monte Carlo lượng tử tự viết cho Li (script \code{05\_monte\_carlo\_luong\_tu.py}). "
         r"Sai số thống kê ($1\sigma$) ghi sau dấu $\pm$.}",
         r"\label{tab:qmc}", r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
         r"Phương pháp & $E$ (Ha) & $E-E_{\text{chính xác}}$ (mHa) & \% tương quan \\", r"\midrule",
         f"VMC, chỉ định thức Slater & ${v['E_no_jastrow']:.4f}\\pm{v['err_no_jastrow']:.4f}$ & "
         f"${mha(v['E_no_jastrow']):+.1f}$ & {pct(v['E_no_jastrow']):.0f}\\% \\\\",
         f"VMC, Slater--Jastrow tối ưu & ${v['E']:.5f}\\pm{v['err']:.5f}$ & ${mha(v['E']):+.2f}$ & "
         f"{pct(v['E']):.1f}\\% \\\\"]
for tau in sorted(dm, key=float, reverse=True):
    d = dm[tau]
    lines.append(f"DMC, $\\tau={float(tau):g}$ & ${d['E']:.5f}\\pm{d['err']:.5f}$ & "
                 f"${mha(d['E']):+.2f}$ & {pct(d['E']):.1f}\\% \\\\")
lines += [f"DMC, ngoại suy $\\tau\\to0$ & ${ex['E']:.5f}\\pm{ex['err']:.5f}$ & ${mha(ex['E']):+.2f}$ & "
          f"{pct(ex['E']):.1f}\\% \\\\",
          r"\midrule", f"Chính xác & ${E_EXACT_LI:.5f}$ & 0 & 100\\% \\\\",
          r"\bottomrule", r"\end{tabular}", r"\end{table}"]
write("bang_qmc.tex", "\n".join(lines) + "\n")

# ===========================================================================
# Hình tổng kết
# ===========================================================================
entries = []  # (nhãn, E, họ, ghi chú chi phí)
ladder = r1["li_ladder"]
entries.append(("Biến phân 1 tham số (ch. 2)", ladder["E_variational"], "khác"))
entries.append(("Hartree–Fock (giới hạn)", E_HF_LIMIT_LI, "hàm sóng"))
entries.append(("MP2 / CBS", cbs["MP2"]["E_tot_CBS"], "hàm sóng"))
entries.append(("CCSD(T) / cc-pCVQZ", runs["4"]["CCSD(T)"][0], "hàm sóng"))
entries.append(("CCSD(T) / CBS", cbs["CCSD(T)"]["E_tot_CBS"], "hàm sóng"))
entries.append(("Full CI / cc-pCVQZ", runs["4"]["FCI"][0], "hàm sóng"))
entries.append(("Full CI / CBS", cbs["FCI"]["E_tot_CBS"], "hàm sóng"))
for row in r4["li"]:
    if row["name"] != "Hartree-Fock":
        entries.append((f"DFT: {row['name']}", row["E"], "DFT"))
entries.append(("VMC (tự viết)", v["E"], "QMC"))
entries.append(("DMC (tự viết, τ→0)", ex["E"], "QMC"))
# Kết quả từ tài liệu (không tự tính): ghi rõ nguồn trong báo cáo
LITERATURE = []
try:
    lit = load_json("06_tai_lieu")
    for item in lit:
        LITERATURE.append((item["label"], item["E"], item["family"]))
except FileNotFoundError:
    pass
entries += LITERATURE

fam_color = {"hàm sóng": C["blue"], "DFT": C["orange"], "QMC": C["aqua"], "khác": NEUTRAL,
             "tài liệu": C["violet"]}
entries.sort(key=lambda t: abs(mha(t[1])))
fig, ax = plt.subplots(figsize=(7.4, 0.33 * len(entries) + 1.4))
y = np.arange(len(entries))[::-1]
for yi, (lab, e, fam) in zip(y, entries):
    err = mha(e)
    a = max(abs(err), 1e-3)
    ax.hlines(yi, 1e-3, a, color="#dddddd", lw=1, zorder=1)
    marker = "o" if err >= 0 else "v"
    ax.plot([a], [yi], marker, color=fam_color[fam], markersize=7, markeredgecolor="white",
            markeredgewidth=1, zorder=3)
    ax.text(a * 1.3, yi, f"{err:+.2f}", va="center", fontsize=7.5, color=TEXT2)
ax.axvline(1000 * CHEM_ACC_HA, color=C["green"], lw=1.2, ls="--")
ax.text(1000 * CHEM_ACC_HA * 1.07, len(entries) - 0.3, "độ chính xác hóa học (1.6 mHa)",
        fontsize=7.5, color=C["green"], va="bottom")
ax.set_xscale("log")
ax.set_xlim(1e-2, 3e3)
ax.set_ylim(-0.8, len(entries) + 0.4)
ax.set_yticks(y, [t[0] for t in entries], fontsize=8)
ax.grid(axis="y", visible=False)
ax.set_xlabel("|E − E$_{chính\\ xác}$| (mHa, thang log)    ● cao hơn đáp án   ▼ thấp hơn đáp án")
ax.set_title("Nguyên tử Li: sai số năng lượng của mọi phương pháp trong tài liệu", loc="left")
from matplotlib.lines import Line2D  # noqa: E402
handles = [Line2D([], [], marker="o", ls="", color=c, label=f, markersize=7)
           for f, c in fam_color.items() if any(t[2] == f for t in entries)]
ax.legend(handles=handles, fontsize=8, loc="center right", frameon=True, framealpha=0.95)
save_fig(fig, "06_tong_ket")
print("  Bảng xếp hạng (|sai số| tăng dần):")
for lab, e, fam in entries:
    print(f"    {lab:32s} {e:.6f}  {mha(e):+9.3f} mHa  ({fam})")

# ===========================================================================
# Macro LaTeX cho các con số được nhắc trong câu chữ của báo cáo
# ===========================================================================
li = {row["name"]: row for row in r4["li"]}
hh = {row["name"]: row for row in r4["h_atom"]}
fci_w = r3["fci_wavefunction"]
macros = {
    # chương 4
    "czeroSq": f"{fci_w['c0_sq_QZ']:.4f}",
    "PctFciQZ": f"{pct(runs['4']['FCI'][0]):.1f}",
    "ErrFciQZ": f"{mha(runs['4']['FCI'][0]):+.2f}",
    "PctFciCBS": f"{pct(cbs['FCI']['E_tot_CBS']):.1f}",
    "ErrFciCBS": f"{mha(cbs['FCI']['E_tot_CBS']):+.2f}",
    "ErrMPtwoCBS": f"{mha(cbs['MP2']['E_tot_CBS']):+.1f}",
    "ErrCCSDTQZ": f"{mha(runs['4']['CCSD(T)'][0]):+.2f}",
    "ErrCCSDTCBS": f"{mha(cbs['CCSD(T)']['E_tot_CBS']):+.2f}",
    "PctMPtwoQZ": f"{pct(runs['4']['MP2'][0]):.0f}",
    "PctCorePVQZ": f"{100*core['cc-pVQZ']['E_corr']/E_CORR:.0f}",
    "PctCoreCVQZ": f"{100*core['cc-pCVQZ']['E_corr']/E_CORR:.0f}",
    # chương 5
    "ErrLDA": f"{li['LDA (SVWN5)']['err_mHa']:+.0f}",
    "ErrPBE": f"{li['PBE']['err_mHa']:+.0f}",
    "ErrBLYP": f"{li['BLYP']['err_mHa']:+.0f}",
    "ErrRtwoSCAN": f"{li['r²SCAN']['err_mHa']:+.1f}",
    "ErrBthreeLYP": f"{li['B3LYP']['err_mHa']:+.0f}",
    "ErrPBEzero": f"{li['PBE0']['err_mHa']:+.0f}",
    "HomoLDA": f"{li['LDA (SVWN5)']['IE_homo_eV']:.2f}",
    "HomoPBE": f"{li['PBE']['IE_homo_eV']:.2f}",
    "HomoBthreeLYP": f"{li['B3LYP']['IE_homo_eV']:.2f}",
    "HomoHF": f"{li['Hartree-Fock']['IE_homo_eV']:.2f}",
    "IEdeltaLDA": f"{li['LDA (SVWN5)']['IE_delta_eV']:.2f}",
    "IEdeltaPBE": f"{li['PBE']['IE_delta_eV']:.2f}",
    "IEdeltaBthreeLYP": f"{li['B3LYP']['IE_delta_eV']:.2f}",
    "HerrLDA": f"{hh['LDA (SVWN5)']['err_mHa']:+.1f}",
    "HerrPBE": f"{hh['PBE']['err_mHa']:+.2f}",
    "HerrBthreeLYP": f"{hh['B3LYP']['err_mHa']:+.1f}",
    "HerrHF": f"{hh['Hartree-Fock']['err_mHa']:+.3f}",
    "HhomoLDA": f"{hh['LDA (SVWN5)']['IE_homo_eV']:.1f}",
    "HhomoPBE": f"{hh['PBE']['IE_homo_eV']:.1f}",
    "HhomoHF": f"{hh['Hartree-Fock']['IE_homo_eV']:.2f}",
    # chương 6
    "EvmcSlater": f"{v['E_no_jastrow']:.4f}",
    "PctVmcSlater": f"{pct(v['E_no_jastrow']):.0f}",
    "Evmc": f"{v['E']:.4f}",
    "EvmcErr": f"{1000*v['err']:.1f}",
    "PctVmc": f"{pct(v['E']):.0f}",
    "ErrVmc": f"{mha(v['E']):+.1f}",
    "Edmc": f"{ex['E']:.5f}",
    "EdmcErr": f"{1000*ex['err']:.2f}",
    "ErrDmc": f"{mha(ex['E']):+.2f}",
    "PctDmc": f"{pct(ex['E']):.1f}",
    "ZoneOpt": f"{v['params_opt'][0]:.2f}",
    "ZtwoOpt": f"{v['params_opt'][1]:.2f}",
}
text = "% Tự sinh bởi code/06_tong_hop.py - không sửa tay\n" + "\n".join(
    f"\\newcommand{{\\{k}}}{{{val}}}" for k, val in macros.items()) + "\n"
write("so_lieu.tex", text)
