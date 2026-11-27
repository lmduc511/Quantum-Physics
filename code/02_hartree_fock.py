"""Chương 2 - Hartree-Fock cho nguyên tử Li.

Phần A: TỰ VIẾT vòng lặp trường tự hợp (SCF) Hartree-Fock không hạn chế (UHF)
        chỉ bằng numpy - để thấy thuật toán thật sự làm gì. PySCF chỉ được dùng
        để tính các tích phân (các con số "đầu vào" S, T, V, (pq|rs)).
Phần B: So sánh với PySCF, xem năng lượng HF hội tụ thế nào khi tập cơ sở lớn dần.
Phần C: ROHF và UHF, nhiễm bẩn spin <S^2>.
Phần D: Định lý Koopmans: năng lượng ion hóa từ năng lượng orbital.
Phần E: Mật độ electron theo bán kính - thấy rõ "lớp vỏ" 1s và 2s.

Chạy:  python 02_hartree_fock.py      (khoảng 1 phút)
"""
import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, scf

from common import (C, TEXT, TEXT2, NEUTRAL, HARTREE_TO_EV, E_HF_LIMIT_LI,
                    IE_EXP_LI_EV, E_EXACT_LI, E_EXACT_LI_PLUS, setup_style,
                    save_fig, save_json, Timer)

setup_style()
results = {}


def li_atom(basis: str) -> gto.Mole:
    # spin = N_alpha - N_beta = 2 - 1 = 1 (một electron không ghép cặp)
    return gto.M(atom="Li 0 0 0", basis=basis, spin=1, charge=0, verbose=0)


# ===========================================================================
# PHẦN A. Vòng lặp SCF tự viết
# ===========================================================================
def my_uhf(mol: gto.Mole, max_iter: int = 100, tol_e: float = 1e-10,
           tol_d: float = 1e-8, damping: float = 0.0):
    """Hartree-Fock không hạn chế (UHF), viết tay cho dễ đọc.

    Ký hiệu (giống trong báo cáo):
      S  : ma trận chồng phủ (overlap) giữa các hàm cơ sở
      H  : Hamiltonian lõi = động năng T + hút hạt nhân V
      eri: tích phân đẩy electron (pq|rs)
      D  : ma trận mật độ,  D_pq = sum_i C_pi C_qi  (tổng trên orbital bị chiếm)
      F  : ma trận Fock,  F = H + J - K
    """
    S = mol.intor("int1e_ovlp")
    H = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    eri = mol.intor("int2e")                   # mảng 4 chiều (pq|rs)
    n_a, n_b = mol.nelec                       # (2, 1) cho Li

    # Trực giao hóa: X = S^(-1/2) biến bài toán FC = SCe thành bài toán trị riêng thường
    s_val, s_vec = np.linalg.eigh(S)
    X = s_vec @ np.diag(s_val ** -0.5) @ s_vec.T

    def solve(F):
        e, Cp = np.linalg.eigh(X.T @ F @ X)
        return e, X @ Cp

    # Đoán ban đầu: bỏ hết lực đẩy electron-electron (chỉ dùng H)
    eps, Cmo = solve(H)
    Ca = Cb = Cmo
    Da = Ca[:, :n_a] @ Ca[:, :n_a].T
    Db = Cb[:, :n_b] @ Cb[:, :n_b].T

    history = []
    E_old = 0.0
    for it in range(1, max_iter + 1):
        J = np.einsum("pqrs,rs->pq", eri, Da + Db, optimize=True)   # Coulomb
        Ka = np.einsum("prqs,rs->pq", eri, Da, optimize=True)       # trao đổi (alpha)
        Kb = np.einsum("prqs,rs->pq", eri, Db, optimize=True)       # trao đổi (beta)
        Fa = H + J - Ka
        Fb = H + J - Kb
        E = 0.5 * np.sum((Da + Db) * H + Da * Fa + Db * Fb)
        ea, Ca = solve(Fa)
        eb, Cb = solve(Fb)
        Da_new = Ca[:, :n_a] @ Ca[:, :n_a].T
        Db_new = Cb[:, :n_b] @ Cb[:, :n_b].T
        dD = max(np.abs(Da_new - Da).max(), np.abs(Db_new - Db).max())
        Da = (1 - damping) * Da_new + damping * Da
        Db = (1 - damping) * Db_new + damping * Db
        history.append(E)
        if abs(E - E_old) < tol_e and dD < tol_d:
            break
        E_old = E
    return {"E": E, "history": history, "eps_a": ea, "eps_b": eb, "Ca": Ca, "Cb": Cb,
            "converged": it < max_iter}


print("A) Vòng lặp SCF tự viết (UHF, tập cơ sở cc-pCVTZ)")
mol = li_atom("cc-pcvtz")
with Timer() as t:
    mine = my_uhf(mol)
ref = scf.UHF(mol).run()
print(f"  tự viết : E = {mine['E']:.10f} Ha sau {len(mine['history'])} vòng ({t.seconds:.1f} s)")
print(f"  PySCF   : E = {ref.e_tot:.10f} Ha")
print(f"  chênh lệch = {abs(mine['E'] - ref.e_tot):.2e} Ha")
results["my_uhf"] = {
    "basis": "cc-pCVTZ",
    "E_mine": mine["E"],
    "E_pyscf": ref.e_tot,
    "n_iter": len(mine["history"]),
    "history": mine["history"],
}

hist = np.array(mine["history"])
err = np.abs(hist - mine["E"])
fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))
ax = axes[0]
ax.plot(np.arange(1, len(hist) + 1), hist, "-o", color=C["blue"], markersize=5,
        markeredgecolor="white", markeredgewidth=1)
ax.axhline(mine["E"], color=NEUTRAL, lw=1, ls="--")
ax.set_xlabel("Vòng lặp")
ax.set_ylabel("Năng lượng (Ha)")
ax.set_title("Năng lượng qua từng vòng SCF", loc="left")
ax.annotate(f"hội tụ: {mine['E']:.6f} Ha", (len(hist), mine["E"]), textcoords="offset points",
            xytext=(-110, 30), fontsize=9, color=TEXT,
            arrowprops=dict(arrowstyle="-", color=TEXT2, lw=0.8))
ax = axes[1]
mask = err > 1e-13   # dưới mức này chỉ còn là sai số làm tròn của máy tính
ax.semilogy(np.arange(1, len(hist) + 1)[mask], err[mask], "-o", color=C["blue"],
            markersize=5, markeredgecolor="white", markeredgewidth=1)
ax.set_xlabel("Vòng lặp")
ax.set_ylabel("|E - E$_{hội tụ}$| (Ha)")
ax.set_title("Sai số giảm theo cấp số nhân", loc="left")
fig.tight_layout()
save_fig(fig, "02_scf_hoi_tu")

# ===========================================================================
# PHẦN B. Hội tụ theo tập cơ sở
# ===========================================================================
print("B) Năng lượng ROHF theo tập cơ sở (giới hạn HF = %.6f Ha)" % E_HF_LIMIT_LI)
families = {
    "Pople (STO-3G, 6-31G)": ["sto-3g", "6-31g"],
    "cc-pVXZ (X = D, T, Q, 5)": ["cc-pvdz", "cc-pvtz", "cc-pvqz", "cc-pv5z"],
    "cc-pCVXZ (X = D, T, Q) + aug": ["cc-pcvdz", "cc-pcvtz", "cc-pcvqz", "aug-cc-pcvqz"],
}
pretty = {"sto-3g": "STO-3G", "6-31g": "6-31G", "cc-pvdz": "cc-pVDZ", "cc-pvtz": "cc-pVTZ",
          "cc-pvqz": "cc-pVQZ", "cc-pv5z": "cc-pV5Z", "cc-pcvdz": "cc-pCVDZ",
          "cc-pcvtz": "cc-pCVTZ", "cc-pcvqz": "cc-pCVQZ", "aug-cc-pcvqz": "aug-cc-pCVQZ"}
basis_rows = []
for fam, blist in families.items():
    for b in blist:
        m = li_atom(b)
        e = scf.ROHF(m).run().e_tot
        row = {"family": fam, "basis": pretty[b], "nao": m.nao_nr(), "E_ROHF": e,
               "err_mHa": 1000 * (e - E_HF_LIMIT_LI)}
        basis_rows.append(row)
        print(f"  {pretty[b]:13s} {m.nao_nr():4d} hàm   E = {e:.8f}   sai số = {row['err_mHa']:9.4f} mHa")
results["basis_convergence"] = basis_rows

# Hình chỉ vẽ 2 họ cho dễ nhìn; họ cc-pCVXZ cho năng lượng HF gần như trùng cc-pVXZ
# (hàm "lõi" chỉ quan trọng khi tính tương quan - xem bảng trong báo cáo).
fig, ax = plt.subplots(figsize=(6.6, 3.9))
for (fam, _), col in zip(list(families.items())[:2], [C["blue"], C["orange"]]):
    rows = [r for r in basis_rows if r["family"] == fam]
    x = [r["nao"] for r in rows]
    y = [max(r["err_mHa"], 1e-4) for r in rows]
    ax.loglog(x, y, "-o", color=col, label=fam, markersize=6,
              markeredgecolor="white", markeredgewidth=1.2)
    below = fam.startswith("cc-pCVXZ")       # nhãn họ cc-pCVXZ đặt dưới điểm cho khỏi đè
    for r, xi, yi in zip(rows, x, y):
        off = (4, -12) if below else (5, 4)
        if r["basis"] == "aug-cc-pCVQZ":
            off = (-20, -22)
        ax.annotate(r["basis"], (xi, yi), textcoords="offset points", xytext=off,
                    fontsize=7.5, color=TEXT2)
ax.set_xlabel("Số hàm cơ sở (log)")
ax.set_ylabel("E$_{ROHF}$ − E$_{giới hạn HF}$ (mHa, log)")
ax.set_title("Tập cơ sở càng lớn, năng lượng HF càng gần giới hạn HF", loc="left")
ax.legend(fontsize=8.5, loc="lower left")
save_fig(fig, "02_hoi_tu_tap_co_so")

# ===========================================================================
# PHẦN C. ROHF vs UHF, nhiễm bẩn spin
# ===========================================================================
print("C) ROHF và UHF trong aug-cc-pCVQZ")
mol = li_atom("aug-cc-pcvqz")
rohf = scf.ROHF(mol).run()
uhf = scf.UHF(mol).run()
s2, mult = uhf.spin_square()
print(f"  ROHF: E = {rohf.e_tot:.8f} Ha")
print(f"  UHF : E = {uhf.e_tot:.8f} Ha  <S^2> = {s2:.6f}  (giá trị đúng cho doublet: 0.75)")
results["rohf_uhf"] = {"basis": "aug-cc-pCVQZ", "E_ROHF": rohf.e_tot, "E_UHF": uhf.e_tot,
                       "S2_UHF": s2, "E_HF_limit": E_HF_LIMIT_LI}

# ===========================================================================
# PHẦN D. Định lý Koopmans và năng lượng ion hóa
# ===========================================================================
print("D) Năng lượng ion hóa của Li")
eps_2s = uhf.mo_energy[0][1]          # orbital alpha thứ 2 = 2s
ie_koopmans = -eps_2s * HARTREE_TO_EV
li_plus = scf.RHF(gto.M(atom="Li 0 0 0", basis="aug-cc-pcvqz", charge=1, spin=0,
                        verbose=0)).run()
ie_delta_scf = (li_plus.e_tot - uhf.e_tot) * HARTREE_TO_EV
ie_exact_nonrel = (E_EXACT_LI_PLUS - E_EXACT_LI) * HARTREE_TO_EV
print(f"  Koopmans (-eps_2s)          = {ie_koopmans:.4f} eV")
print(f"  Delta-SCF E(Li+) - E(Li)     = {ie_delta_scf:.4f} eV")
print(f"  Từ năng lượng chính xác     = {ie_exact_nonrel:.4f} eV")
print(f"  Thực nghiệm (NIST)          = {IE_EXP_LI_EV:.4f} eV")
results["ionization"] = {
    "eps_1s_alpha_Ha": float(uhf.mo_energy[0][0]),
    "eps_2s_alpha_Ha": float(eps_2s),
    "eps_1s_beta_Ha": float(uhf.mo_energy[1][0]),
    "IE_koopmans_eV": ie_koopmans,
    "IE_deltaSCF_eV": ie_delta_scf,
    "E_HF_Li_plus": li_plus.e_tot,
    "IE_exact_nonrel_eV": ie_exact_nonrel,
    "IE_exp_eV": IE_EXP_LI_EV,
}

# ===========================================================================
# PHẦN E. Mật độ electron theo bán kính
# ===========================================================================
print("E) Mật độ xuyên tâm")
r = np.linspace(1e-4, 30.0, 6000)                     # bohr (đủ xa để bắt hết đuôi 2s)
pts = np.column_stack([np.zeros_like(r), np.zeros_like(r), r])
ao = mol.eval_gto("GTOval_sph", pts)                  # giá trị hàm cơ sở tại các điểm
phi_a = ao @ uhf.mo_coeff[0]                          # orbital alpha
phi_b = ao @ uhf.mo_coeff[1]                          # orbital beta
rho_1s = phi_a[:, 0] ** 2 + phi_b[:, 0] ** 2          # hai electron 1s
rho_2s = phi_a[:, 1] ** 2                             # một electron 2s
P_1s = 4 * np.pi * r**2 * rho_1s                      # mật độ xuyên tâm P(r)
P_2s = 4 * np.pi * r**2 * rho_2s
P_tot = P_1s + P_2s
dr = r[1] - r[0]
print(f"  tích phân P(r) dr = {P_tot.sum() * dr:.4f}  (phải bằng 3 electron)")
r_peak_1s = r[np.argmax(P_1s)]
r_peak_2s = r[np.argmax(np.where(r > 1.0, P_2s, 0))]
print(f"  đỉnh lớp 1s tại r = {r_peak_1s:.3f} bohr, đỉnh lớp 2s tại r = {r_peak_2s:.3f} bohr")
np.save("../results/02_radial_density.npy", np.vstack([r, P_1s, P_2s]))

fig, ax = plt.subplots(figsize=(6.6, 3.8))
ax.plot(r, P_tot, color=C["blue"], label="tổng (3 electron)")
ax.plot(r, P_1s, color=C["orange"], lw=1.6, ls="--", label="2 electron 1s (lớp K)")
ax.plot(r, P_2s, color=C["aqua"], lw=1.6, ls="-.", label="1 electron 2s (lớp L)")
ax.annotate(f"lớp K: đỉnh ở r ≈ {r_peak_1s:.2f} bohr", (r_peak_1s, P_tot.max()),
            textcoords="offset points", xytext=(18, -8), fontsize=9, color=TEXT)
ax.annotate(f"lớp L: đỉnh ở r ≈ {r_peak_2s:.2f} bohr", (r_peak_2s, P_2s[np.argmin(abs(r - r_peak_2s))]),
            textcoords="offset points", xytext=(10, 18), fontsize=9, color=TEXT,
            arrowprops=dict(arrowstyle="-", color=TEXT2, lw=0.8))
i_node = np.argmin(np.where((r > 0.3) & (r < 1.5), P_2s, np.inf))
ax.annotate(f"nút của orbital 2s (r ≈ {r[i_node]:.2f})", (r[i_node], 0.0), textcoords="offset points",
            xytext=(12, 40), fontsize=8.5, color=TEXT2,
            arrowprops=dict(arrowstyle="-", color=TEXT2, lw=0.8))
results["radial_node_2s_bohr"] = float(r[i_node])
ax.set_xlabel("Khoảng cách tới hạt nhân r (bohr;  1 bohr ≈ 0.529 Å)")
ax.set_ylabel("P(r) = 4πr²ρ(r)  (electron/bohr)")
ax.set_title("Mật độ electron xuyên tâm của Li (UHF): hai lớp vỏ tách bạch", loc="left")
ax.legend(fontsize=8.5)
ax.set_xlim(0, 10)
ax.set_ylim(0, None)
save_fig(fig, "02_mat_do_xuyen_tam")
results["radial"] = {"r_peak_1s_bohr": float(r_peak_1s), "r_peak_2s_bohr": float(r_peak_2s),
                     "integral": float(P_tot.sum() * dr)}

save_json(results, "02_hartree_fock")
