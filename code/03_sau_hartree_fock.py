"""Chương 3 - Sau Hartree-Fock: MP2, CISD, CCSD, CCSD(T) và Full CI cho Li.

Phần A: chạy mọi phương pháp trên 3 tập cơ sở cc-pCVXZ (X = D, T, Q), đo thời gian.
Phần B: ngoại suy về tập cơ sở vô hạn (CBS) và so với năng lượng chính xác.
Phần C: "mổ xẻ" hàm sóng Full CI: định thức Hartree-Fock chiếm bao nhiêu phần?
Phần D: vì sao cần hàm "lõi" (core): so cc-pVQZ với cc-pCVQZ.
Phần E: số định thức bùng nổ thế nào khi số electron và orbital tăng.
Phần F: chi phí tính toán lý thuyết của từng phương pháp.

Chạy:  python 03_sau_hartree_fock.py     (vài phút)
"""
from math import comb

import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, scf, mp, ci, cc, fci

import json

from common import (C, TEXT, TEXT2, NEUTRAL, E_EXACT_LI, E_HF_LIMIT_LI, CHEM_ACC_HA,
                    RES_DIR, setup_style, save_fig, save_json, Timer)

setup_style()
results = {}
E_CORR_EXACT = E_EXACT_LI - E_HF_LIMIT_LI   # năng lượng tương quan "chính xác" của Li


def li_atom(basis):
    return gto.M(atom="Li 0 0 0", basis=basis, spin=1, verbose=0)


CACHE = RES_DIR / "cache_03"
CACHE.mkdir(exist_ok=True)


def run_all(basis):
    """Chạy HF, MP2, CISD, CCSD, CCSD(T), FCI; trả về năng lượng và thời gian.
    Kết quả được lưu tạm vào results/cache_03/ để chạy lại không phải tính lại từ đầu."""
    f_json, f_ci = CACHE / f"{basis}.json", CACHE / f"{basis}_civec.npy"
    if f_json.exists() and f_ci.exists():
        print(f"  (dùng kết quả đã lưu cho {basis})")
        out = json.loads(f_json.read_text())
        rohf = scf.ROHF(li_atom(basis)).run(conv_tol=1e-11)
        return out, rohf, None, np.load(f_ci)
    out, rohf, solver, civec = _run_all(basis)
    f_json.write_text(json.dumps(out))
    np.save(f_ci, civec)
    return out, rohf, solver, civec


def _run_all(basis):
    mol = li_atom(basis)
    out = {"basis": basis, "nao": mol.nao_nr()}
    with Timer() as t:
        rohf = scf.ROHF(mol).run(conv_tol=1e-11)
    out["ROHF"] = (rohf.e_tot, t.seconds)
    with Timer() as t:
        uhf = scf.UHF(mol).run(conv_tol=1e-11)
    out["UHF"] = (uhf.e_tot, t.seconds)
    with Timer() as t:
        e_mp2 = mp.UMP2(uhf).run().e_tot
    out["MP2"] = (e_mp2, t.seconds)
    with Timer() as t:
        e_cisd = ci.UCISD(uhf).run().e_tot
    out["CISD"] = (e_cisd, t.seconds)
    with Timer() as t:
        mycc = cc.UCCSD(uhf).run()
    out["CCSD"] = (mycc.e_tot, t.seconds)
    with Timer() as t:
        e_t = mycc.ccsd_t()
    out["CCSD(T)"] = (mycc.e_tot + e_t, out["CCSD"][1] + t.seconds)
    with Timer() as t:
        solver = fci.FCI(rohf)
        e_fci, civec = solver.kernel()
    out["FCI"] = (e_fci, t.seconds)
    norb = rohf.mo_coeff.shape[1]
    out["n_det_FCI"] = comb(norb, 2) * comb(norb, 1)
    out["S2_FCI"] = float(solver.spin_square(civec, norb, mol.nelec)[0])
    return out, rohf, solver, civec


METHODS = ["UHF", "MP2", "CISD", "CCSD", "CCSD(T)", "FCI"]

# ===========================================================================
# PHẦN A
# ===========================================================================
print("A) Các phương pháp sau HF trên cc-pCVXZ")
runs = {}
for X, basis in zip([2, 3, 4], ["cc-pcvdz", "cc-pcvtz", "cc-pcvqz"]):
    out, rohf, solver, civec = run_all(basis)
    runs[X] = out
    print(f"  {basis}  ({out['nao']} orbital, FCI có {out['n_det_FCI']:,} định thức, "
          f"<S^2>_FCI = {out['S2_FCI']:.4f})")
    for m in ["ROHF"] + METHODS:
        e, sec = out[m]
        print(f"    {m:8s} E = {e:.8f} Ha   sai số so với chính xác = "
              f"{1000*(e - E_EXACT_LI):8.3f} mHa   ({sec:6.2f} s)")
    if X == 4:
        qz_rohf, qz_solver, qz_civec = rohf, solver, civec
results["runs"] = runs

# Phần trăm năng lượng tương quan thu được (so với FCI cùng tập cơ sở)
print("  Phần trăm năng lượng tương quan thu được (so với FCI cùng tập cơ sở, tham chiếu ROHF):")
pct = {}
for X, out in runs.items():
    e_ref = out["ROHF"][0]
    e_corr_fci = out["FCI"][0] - e_ref
    pct[X] = {m: 100 * (out[m][0] - e_ref) / e_corr_fci for m in METHODS}
    print("    X=%d: " % X + "  ".join(f"{m} {pct[X][m]:6.2f}%" for m in METHODS))
results["percent_corr"] = pct

# ===========================================================================
# PHẦN B. Ngoại suy CBS (Helgaker và cộng sự 1997): E_corr(X) = E_CBS + A / X^3
# ===========================================================================
print("B) Ngoại suy về tập cơ sở vô hạn từ T và Q")


def cbs(e_t, e_q):
    return (4**3 * e_q - 3**3 * e_t) / (4**3 - 3**3)


cbs_res = {}
for m in ["MP2", "CCSD", "CCSD(T)", "FCI"]:
    ec_t = runs[3][m][0] - runs[3]["UHF"][0]
    ec_q = runs[4][m][0] - runs[4]["UHF"][0]
    if m == "FCI":  # FCI dựng trên orbital ROHF -> lấy tương quan so với ROHF
        ec_t = runs[3][m][0] - runs[3]["ROHF"][0]
        ec_q = runs[4][m][0] - runs[4]["ROHF"][0]
    ec_cbs = cbs(ec_t, ec_q)
    e_tot = E_HF_LIMIT_LI + ec_cbs
    cbs_res[m] = {"E_corr_T": ec_t, "E_corr_Q": ec_q, "E_corr_CBS": ec_cbs, "E_tot_CBS": e_tot,
                  "err_mHa": 1000 * (e_tot - E_EXACT_LI)}
    print(f"  {m:8s} E_corr(CBS) = {1000*ec_cbs:8.3f} mHa ->  E = {e_tot:.6f} Ha, "
          f"sai số {1000*(e_tot-E_EXACT_LI):7.3f} mHa")
print(f"  Năng lượng tương quan chính xác = {1000*E_CORR_EXACT:.3f} mHa")
results["cbs"] = cbs_res
results["E_corr_exact"] = E_CORR_EXACT

# ===========================================================================
# PHẦN C. Mổ xẻ hàm sóng Full CI (cc-pCVQZ)
# ===========================================================================
print("C) Thành phần hàm sóng Full CI (cc-pCVQZ, orbital ROHF)")
mol = li_atom("cc-pcvqz")
norb = qz_rohf.mo_coeff.shape[1]
# Đặc trưng mômen động lượng l của từng orbital: s, p, d, f, g
ao_l = np.array([lab.split()[2][1] for lab in mol.ao_labels()])
S = mol.intor("int1e_ovlp")
Cm = qz_rohf.mo_coeff
lchar = []
for k in range(norb):
    w = {}
    for l in "spdfg":
        idx = ao_l == l
        w[l] = float(Cm[idx, k] @ (S[idx][:, :] @ Cm[:, k]))  # trọng số Mulliken theo l
    lchar.append(max(w, key=w.get))
names = []
count = {}
for k in range(norb):
    l = lchar[k]
    count[l] = count.get(l, 0) + 1
    names.append({0: "1s", 1: "2s"}.get(k, f"{l}{count[l]}*"))

c0 = qz_civec[0, 0]
print(f"  hệ số định thức HF: c0 = {c0:.5f}  ->  trọng số c0^2 = {c0**2:.5f}")
top = fci.addons.large_ci(qz_civec, norb, (2, 1), tol=0.01, return_strs=False)
top = sorted(top, key=lambda t: -abs(t[0]))[:8]
top_rows = []
for coef, occ_a, occ_b in top:
    desc = "α: " + " ".join(names[i] for i in occ_a) + " | β: " + " ".join(names[i] for i in occ_b)
    n_exc = len(set(occ_a) - {0, 1}) + len(set(occ_b) - {0})
    print(f"    c = {coef:+.5f}  (kích thích bậc {n_exc})  {desc}")
    top_rows.append({"coef": float(coef), "excitation": n_exc, "alpha": [names[i] for i in occ_a],
                     "beta": [names[i] for i in occ_b]})
# Trọng số theo bậc kích thích
civ = qz_civec
from pyscf.fci import cistring
strs_a = cistring.make_strings(range(norb), 2)
strs_b = cistring.make_strings(range(norb), 1)
exc_a = np.array([2 - bin(s & 0b11).count("1") for s in strs_a])
exc_b = np.array([1 - bin(s & 0b1).count("1") for s in strs_b])
exc = exc_a[:, None] + exc_b[None, :]
weights = {int(k): float(np.sum(civ[exc == k] ** 2)) for k in range(4)}
print("  trọng số theo bậc kích thích:", {k: f"{v:.6f}" for k, v in weights.items()})
results["fci_wavefunction"] = {"c0": float(c0), "c0_sq": float(c0**2), "top": top_rows,
                               "weights_by_excitation": weights, "n_det": int(civ.size)}

# ===========================================================================
# PHẦN D. Tầm quan trọng của hàm lõi
# ===========================================================================
print("D) cc-pVQZ (không có hàm lõi) và cc-pCVQZ (có hàm lõi), Full CI")
mol_v = li_atom("cc-pvqz")
rohf_v = scf.ROHF(mol_v).run(conv_tol=1e-11)
f_v = CACHE / "cc-pvqz_fci.json"
if f_v.exists():
    e_fci_v = json.loads(f_v.read_text())["E_FCI"]
else:
    e_fci_v = fci.FCI(rohf_v).kernel()[0]
    f_v.write_text(json.dumps({"E_FCI": e_fci_v}))
core = {
    "cc-pVQZ": {"E_ROHF": rohf_v.e_tot, "E_FCI": e_fci_v, "E_corr": e_fci_v - rohf_v.e_tot},
    "cc-pCVQZ": {"E_ROHF": runs[4]["ROHF"][0], "E_FCI": runs[4]["FCI"][0],
                 "E_corr": runs[4]["FCI"][0] - runs[4]["ROHF"][0]},
}
for k, v in core.items():
    print(f"  {k:9s} E_corr(FCI) = {1000*v['E_corr']:8.3f} mHa  "
          f"({100*v['E_corr']/E_CORR_EXACT:5.1f}% của giá trị chính xác)")
results["core_functions"] = core

# Biểu đồ: năng lượng tương quan theo tập cơ sở
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), gridspec_kw={"width_ratios": [1.1, 1]})
ax = axes[0]
Xs = [2, 3, 4]
cols = {"MP2": C["blue"], "CISD": C["orange"], "CCSD": C["aqua"], "CCSD(T)": C["yellow"],
        "FCI": C["magenta"]}
styles = {"MP2": "-o", "CISD": "-s", "CCSD": "-^", "CCSD(T)": "-D", "FCI": "-v"}
for m in ["MP2", "CISD", "CCSD", "CCSD(T)", "FCI"]:
    ref = "ROHF" if m == "FCI" else "UHF"
    y = [1000 * (runs[X][m][0] - runs[X][ref][0]) for X in Xs]
    ax.plot(Xs, y, styles[m], color=cols[m], label=m, markersize=6,
            markeredgecolor="white", markeredgewidth=1)
ax.axhline(1000 * E_CORR_EXACT, color=TEXT, lw=1.2, ls="--")
ax.text(2.02, 1000 * E_CORR_EXACT - 1.3, f"chính xác: {1000*E_CORR_EXACT:.2f} mHa",
        fontsize=8.5, color=TEXT)
ax.set_xticks(Xs, ["cc-pCVDZ", "cc-pCVTZ", "cc-pCVQZ"])
ax.set_ylabel("Năng lượng tương quan (mHa)")
ax.set_title("(a) Tương quan theo tập cơ sở", loc="left")
ax.legend(fontsize=8, ncol=2, loc="upper right")

ax = axes[1]
labels, vals, colors = [], [], []
for m in ["UHF", "MP2", "CISD", "CCSD", "CCSD(T)", "FCI"]:
    labels.append(f"{m} / cc-pCVQZ")
    vals.append(1000 * (runs[4][m][0] - E_EXACT_LI))
    colors.append(C["blue"])
for m in ["MP2", "CCSD(T)", "FCI"]:
    labels.append(f"{m} / CBS")
    vals.append(1000 * cbs_res[m]["E_tot_CBS"] - 1000 * E_EXACT_LI)
    colors.append(C["orange"])
yy = np.arange(len(labels))[::-1]
absvals = np.abs(vals)
ax.hlines(yy, 1e-2, absvals, color="#d0d0d0", lw=1)
for yi, v, c in zip(yy, absvals, colors):
    ax.plot([v], [yi], "o", color=c, markersize=7, markeredgecolor="white", markeredgewidth=1)
for yi, v, raw in zip(yy, absvals, vals):
    ax.text(v * 1.25, yi, f"{raw:+.2f}", va="center", fontsize=7.5, color=TEXT2)
ax.axvline(1000 * CHEM_ACC_HA, color=C["green"], lw=1.2, ls="--")
ax.text(1000 * CHEM_ACC_HA * 1.08, len(labels) - 0.6, "độ chính xác\nhóa học", fontsize=7.5,
        color=C["green"], va="top")
ax.set_xscale("log")
ax.set_xlim(1e-2, 300)
ax.set_yticks(yy, labels, fontsize=8)
ax.set_xlabel("|E − E$_{chính\\ xác}$| (mHa, log)")
ax.set_title("(b) Sai số so với năng lượng chính xác", loc="left")
ax.grid(axis="y", visible=False)
fig.tight_layout()
save_fig(fig, "03_tuong_quan_theo_tap_co_so")

# ===========================================================================
# PHẦN E. Bùng nổ số định thức
# ===========================================================================
print("E) Số định thức Full CI")
M = np.arange(2, 201)
systems = [("Li (3 e)", 2, 1, C["blue"]), ("C (6 e)", 4, 2, C["orange"]),
           ("Ne (10 e)", 5, 5, C["aqua"]), ("Fe (26 e)", 15, 11, C["violet"])]
fig, ax = plt.subplots(figsize=(6.6, 3.9))
det_table = {}
for name, na, nb, col in systems:
    valid = M >= na
    nd = np.array([comb(int(m), na) * comb(int(m), nb) if m >= na else np.nan for m in M],
                  dtype=float)
    ax.semilogy(M[valid], nd[valid], color=col, label=name)
    det_table[name] = {int(m): comb(int(m), na) * comb(int(m), nb) for m in (10, 50, 100, 200)}
for X in Xs:
    ax.plot([runs[X]["nao"]], [runs[X]["n_det_FCI"]], "o", color=C["blue"], markersize=7,
            markeredgecolor="white", markeredgewidth=1.2, zorder=4)
ax.annotate("các phép Full CI cho Li\nđã chạy trong code này", (runs[4]["nao"], runs[4]["n_det_FCI"]),
            textcoords="offset points", xytext=(25, -38), fontsize=8.5, color=TEXT,
            arrowprops=dict(arrowstyle="-", color=TEXT2, lw=0.8))
ax.axhline(1e12, color=NEUTRAL, lw=1, ls="--")
ax.text(200, 1e12 * 2.2, "~10$^{12}$ định thức: cỡ giới hạn của siêu máy tính hiện nay",
        fontsize=8, color=TEXT2, ha="right")
ax.set_xlabel("Số orbital không gian M (kích thước tập cơ sở)")
ax.set_ylabel("Số định thức Slater (log)")
ax.set_title("Full CI: số định thức = C(M, N$_α$) × C(M, N$_β$)", loc="left")
ax.legend(fontsize=8.5, loc="lower right")
ax.set_ylim(1, 1e40)
save_fig(fig, "03_bung_no_dinh_thuc")
results["det_counts"] = {k: {str(m): str(v) for m, v in d.items()} for k, d in det_table.items()}
for k, d in det_table.items():
    print(f"  {k}: M=100 -> {d[100]:.3e} định thức")

# ===========================================================================
# PHẦN F. Chi phí lý thuyết
# ===========================================================================
n = np.linspace(1, 60, 300)
fig, ax = plt.subplots(figsize=(6.6, 3.9))
curves = [("DFT ~ n³", 3, C["aqua"]), ("Hartree-Fock ~ n⁴", 4, C["blue"]),
          ("MP2 ~ n⁵", 5, C["orange"]), ("CCSD ~ n⁶", 6, C["yellow"]),
          ("CCSD(T) ~ n⁷", 7, C["violet"])]
for name, p, col in curves:
    ax.semilogy(n, (n / 10) ** p, color=col, label=name)
ax.semilogy(n, np.exp(n - 10), color=C["red"], label="Full CI ~ eⁿ")
ax.set_xlabel("Kích thước hệ n (tỉ lệ với số electron / hàm cơ sở)")
ax.set_ylabel("Chi phí tương đối (log; = 1 khi n = 10)")
ax.set_title("Thứ bậc chi phí: mỗi bậc chính xác hơn đều đắt hơn", loc="left")
ax.set_ylim(1e-3, 1e15)
ax.legend(fontsize=8.5, loc="upper left")
save_fig(fig, "03_chi_phi")

save_json(results, "03_sau_hartree_fock")
