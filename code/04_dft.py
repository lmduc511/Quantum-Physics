"""Chương 4 - Lý thuyết phiếm hàm mật độ (DFT) cho Li.

Phần A: năng lượng toàn phần của Li với nhiều phiếm hàm (LDA, GGA, meta-GGA, hybrid).
Phần B: năng lượng ion hóa: tính bằng hiệu năng lượng (Delta-SCF) và bằng -eps_HOMO.
Phần C: lỗi tự tương tác: nguyên tử H (1 electron) - nơi đáp án đúng là -0.5 Ha.
Phần D: mật độ electron của DFT/HF so với mật độ Full CI (gần như chính xác).

Chạy:  python 04_dft.py     (vài phút)
"""
import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, scf, dft, fci

from common import (C, TEXT, TEXT2, NEUTRAL, HARTREE_TO_EV, E_EXACT_LI, E_EXACT_LI_PLUS,
                    IE_EXP_LI_EV, setup_style, save_fig, save_json)

setup_style()
results = {}
BASIS = "aug-cc-pcvqz"
FUNCTIONALS = [  # (tên hiển thị, chuỗi PySCF, bậc thang Jacob)
    ("LDA (SVWN5)", "lda,vwn", "1 LDA"),
    ("PBE", "pbe,pbe", "2 GGA"),
    ("BLYP", "b88,lyp", "2 GGA"),
    ("r²SCAN", "r2scan,r2scan", "3 meta-GGA"),   # SCAN gốc không hội tụ với Li
    ("B3LYP", "b3lyp", "4 hybrid"),
    ("PBE0", "pbe0", "4 hybrid"),
]


def atom(sym, charge, spin, basis=BASIS):
    return gto.M(atom=f"{sym} 0 0 0", basis=basis, charge=charge, spin=spin, verbose=0)


def run_ks(mol, xc, dm0=None):
    """Kohn-Sham không hạn chế (UKS) cho hệ vỏ mở, hạn chế (RKS) cho vỏ đóng.
    Lưu ý PySCF: 'lda' chỉ có phần trao đổi -> dùng 'lda,vwn' để có cả tương quan;
    'b3lyp' trong PySCF dùng VWN-RPA (giống Gaussian), 'b3lyp5' dùng VWN5."""
    mf = dft.UKS(mol) if mol.spin else dft.RKS(mol)
    mf.xc = xc
    mf.grids.level = 5          # lưới tích phân đủ dày (sai lệch cỡ micro-hartree)
    mf.conv_tol = 1e-9
    mf.max_cycle = 150
    return mf.run(dm0) if dm0 is not None else mf.run()


def homo(mf):
    """Năng lượng orbital bị chiếm cao nhất (HOMO)."""
    if isinstance(mf.mo_energy, (list, tuple)) or np.ndim(mf.mo_energy) == 2:
        ea, eb = mf.mo_energy
        oa, ob = mf.mo_occ
        return max(ea[oa > 0].max(), eb[ob > 0].max() if (ob > 0).any() else -np.inf)
    return mf.mo_energy[mf.mo_occ > 0].max()


# ===========================================================================
# PHẦN A + B
# ===========================================================================
print("A+B) Li và Li+ với các phiếm hàm (aug-cc-pCVQZ)")
li, li_p = atom("Li", 0, 1), atom("Li", 1, 0)
rows = []
hf_li = scf.UHF(li).run()
hf_lip = scf.RHF(li_p).run()
rows.append({"name": "Hartree-Fock", "rung": "0 HF", "E": hf_li.e_tot, "E_plus": hf_lip.e_tot,
             "homo": float(homo(hf_li))})
# Mật độ PBE dùng làm điểm khởi đầu cho các phiếm hàm khác (giúp meta-GGA hội tụ)
pbe_li, pbe_lip = run_ks(li, "pbe,pbe"), run_ks(li_p, "pbe,pbe")
dm_li, dm_lip = pbe_li.make_rdm1(), pbe_lip.make_rdm1()
for name, xc, rung in FUNCTIONALS:
    a = run_ks(li, xc, dm_li)
    b = run_ks(li_p, xc, dm_lip)
    if not (a.converged and b.converged):
        print(f"  CẢNH BÁO: {name} chưa hội tụ (Li: {a.converged}, Li+: {b.converged})")
    rows.append({"name": name, "rung": rung, "E": a.e_tot, "E_plus": b.e_tot,
                 "homo": float(homo(a)), "converged": bool(a.converged and b.converged),
                 "S2": float(a.spin_square()[0])})
    print(f"  ... xong {name}: E = {a.e_tot:.6f} Ha", flush=True)
for r in rows:
    r["err_mHa"] = 1000 * (r["E"] - E_EXACT_LI)
    r["IE_delta_eV"] = (r["E_plus"] - r["E"]) * HARTREE_TO_EV
    r["IE_homo_eV"] = -r["homo"] * HARTREE_TO_EV
    print(f"  {r['name']:12s} E = {r['E']:.6f}  sai số {r['err_mHa']:+8.2f} mHa   "
          f"IE(ΔSCF) = {r['IE_delta_eV']:.3f} eV   -ε_HOMO = {r['IE_homo_eV']:.3f} eV")
ie_exact = (E_EXACT_LI_PLUS - E_EXACT_LI) * HARTREE_TO_EV
print(f"  chính xác: E = {E_EXACT_LI:.6f};  IE (phi tương đối tính) = {ie_exact:.3f} eV; "
      f"thực nghiệm {IE_EXP_LI_EV:.3f} eV")
results["li"] = rows

fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6), gridspec_kw={"width_ratios": [1, 1.15]})
ax = axes[0]
names = [r["name"] for r in rows]
y = np.arange(len(rows))[::-1]
errs = [r["err_mHa"] for r in rows]
bar_cols = [NEUTRAL if r["name"] == "Hartree-Fock" else C["blue"] for r in rows]
ax.barh(y, errs, color=bar_cols, height=0.6)
for yi, e in zip(y, errs):
    ax.text(e + (4 if e >= 0 else -4), yi, f"{e:+.1f}", va="center",
            ha="left" if e >= 0 else "right", fontsize=8, color=TEXT)
ax.axvline(0, color=TEXT, lw=1)
ax.set_yticks(y, names, fontsize=8.5)
ax.set_xlabel("E − E$_{chính\\ xác}$ (mHa)")
ax.set_title("(a) Sai số năng lượng toàn phần của Li", loc="left")
ax.grid(axis="y", visible=False)
lim = max(abs(min(errs)), abs(max(errs))) * 1.35
ax.set_xlim(-lim, lim)
ax.text(-lim * 0.97, -0.9, "← thấp hơn đáp án đúng", fontsize=7.5, color=TEXT2)
ax.text(lim * 0.97, -0.9, "cao hơn đáp án đúng →", fontsize=7.5, color=TEXT2, ha="right")
ax.set_ylim(-1.3, len(rows) - 0.4)

ax = axes[1]
ax.plot([r["IE_delta_eV"] for r in rows], y, "o", color=C["blue"], markersize=7,
        markeredgecolor="white", markeredgewidth=1, label="ΔSCF: E(Li⁺) − E(Li)")
ax.plot([r["IE_homo_eV"] for r in rows], y, "s", color=C["orange"], markersize=7,
        markeredgecolor="white", markeredgewidth=1, label="−ε$_{HOMO}$")
for yi, r in zip(y, rows):
    ax.plot([r["IE_homo_eV"], r["IE_delta_eV"]], [yi, yi], color="#d0d0d0", lw=1, zorder=0)
ax.axvline(IE_EXP_LI_EV, color=C["green"], lw=1.2, ls="--")
ax.text(IE_EXP_LI_EV + 0.05, len(rows) - 0.55, f"thực nghiệm\n{IE_EXP_LI_EV:.2f} eV",
        fontsize=7.5, color=C["green"], va="top")
ax.set_yticks(y, names, fontsize=8.5)
ax.set_xlabel("Năng lượng ion hóa (eV)")
ax.set_title("(b) Hai cách tính năng lượng ion hóa", loc="left")
ax.legend(fontsize=8, loc="lower left")
ax.grid(axis="y", visible=False)
ax.set_xlim(2.0, 6.6)
fig.tight_layout()
save_fig(fig, "04_dft_li")

# ===========================================================================
# PHẦN C. Lỗi tự tương tác với nguyên tử H
# ===========================================================================
print("C) Nguyên tử H (1 electron): đáp án đúng E = -0.5 Ha, -ε = 13.606 eV")
h = gto.M(atom="H 0 0 0", basis="aug-cc-pv5z", spin=1, verbose=0)
hrows = []
mf = scf.UHF(h).run()
hrows.append({"name": "Hartree-Fock", "E": mf.e_tot, "homo": float(homo(mf))})
for name, xc, rung in FUNCTIONALS:
    mf = run_ks(h, xc)
    hrows.append({"name": name, "E": mf.e_tot, "homo": float(homo(mf))})
    print(f"  ... xong H với {name}", flush=True)
for r in hrows:
    r["err_mHa"] = 1000 * (r["E"] + 0.5)
    r["IE_homo_eV"] = -r["homo"] * HARTREE_TO_EV
    print(f"  {r['name']:12s} E = {r['E']:.6f} (sai số {r['err_mHa']:+7.2f} mHa)   "
          f"-ε_HOMO = {r['IE_homo_eV']:.3f} eV")
results["h_atom"] = hrows

# ===========================================================================
# PHẦN D. Mật độ so với Full CI (cùng tập cơ sở cc-pCVTZ)
# ===========================================================================
print("D) Mật độ xuyên tâm: so với Full CI (cc-pCVTZ)")
mol = atom("Li", 0, 1, basis="cc-pcvtz")
r = np.linspace(1e-4, 12.0, 3000)
pts = np.column_stack([np.zeros_like(r), np.zeros_like(r), r])
ao = mol.eval_gto("GTOval_sph", pts)


def radial_from_dm(dm):
    dm_tot = dm[0] + dm[1] if np.ndim(dm) == 3 else dm
    rho = np.einsum("pi,ij,pj->p", ao, dm_tot, ao)
    return 4 * np.pi * r**2 * rho


rohf = scf.ROHF(mol).run(conv_tol=1e-11)
solver = fci.FCI(rohf)
e_fci, civec = solver.kernel()
dm1a, dm1b = solver.make_rdm1s(civec, rohf.mo_coeff.shape[1], mol.nelec)
Cmo = rohf.mo_coeff
P_fci = radial_from_dm(np.array([Cmo @ dm1a @ Cmo.T, Cmo @ dm1b @ Cmo.T]))
dens = {"Hartree-Fock": radial_from_dm(scf.UHF(mol).run().make_rdm1())}
for name, xc in [("LDA (SVWN5)", "lda,vwn"), ("PBE", "pbe,pbe"), ("B3LYP", "b3lyp")]:
    dens[name] = radial_from_dm(run_ks(mol, xc).make_rdm1())
fig, ax = plt.subplots(figsize=(6.6, 3.8))
for (name, P), col in zip(dens.items(), [NEUTRAL, C["blue"], C["orange"], C["aqua"]]):
    ax.plot(r, P - P_fci, color=col, lw=1.8, label=name)
ax.axhline(0, color=TEXT, lw=0.8)
ax.set_xlim(0, 6)
ax.set_xlabel("r (bohr)")
ax.set_ylabel("P$_{phương\\ pháp}$(r) − P$_{FCI}$(r)  (electron/bohr)")
ax.set_title("Sai lệch mật độ xuyên tâm so với Full CI (cc-pCVTZ)", loc="left")
ax.legend(fontsize=8.5)
save_fig(fig, "04_sai_lech_mat_do")
dr = r[1] - r[0]
dev = {name: float(np.sum(np.abs(P - P_fci)) * dr) for name, P in dens.items()}
for k, v in dev.items():
    print(f"  ∫|ΔP| dr  {k:12s} = {v:.4f} electron")
results["density_deviation"] = {"basis": "cc-pCVTZ", "E_FCI": e_fci, "int_abs_dP": dev}

save_json(results, "04_dft")
