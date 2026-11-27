"""Chương 5 - Monte Carlo lượng tử (QMC) tự viết cho nguyên tử Li, chỉ dùng numpy.

Hàm thử Slater-Jastrow:
    psi(R) = D_up(r0, r1) * D_dn(r2) * exp(J_ee + J_en)
    D_up   = phi1(r0) phi2(r1) - phi1(r1) phi2(r0)     (định thức 2x2, hai electron spin lên)
    D_dn   = phi1(r2)                                 (một electron spin xuống)
    phi1(r) = exp(-z1 r)                              (kiểu orbital 1s)
    phi2(r) = (1 - c r) exp(-z2 r),  c = z1 - z2      (kiểu orbital 2s, có nút)
    J_ee   = sum_{i<j} a_ij r_ij / (1 + b r_ij)       (Jastrow electron-electron)
             a_ij = 1/4 nếu cùng spin, 1/2 nếu ngược spin  -> thỏa "cusp" e-e của Kato
    J_en   = -sum_i (Z - z1) r_i / (1 + d r_i)        (Jastrow electron-hạt nhân)
             -> cộng với orbital, đạo hàm log của psi tại hạt nhân đúng bằng -Z ("cusp" e-n)
    Nhờ thỏa cả hai điều kiện cusp, năng lượng cục bộ E_L không bị vô hạn khi hai hạt
    chạm nhau - điều bắt buộc để DMC ổn định.

Phần A: Monte Carlo biến phân (VMC): lấy mẫu |psi|^2 bằng thuật toán Metropolis,
        tối ưu 4 tham số (z1, z2, c, b) bằng "lấy mẫu tương quan".
Phần B: Monte Carlo khuếch tán (DMC) với "nút cố định": chiếu hàm thử về trạng thái cơ bản.
Phần C: ảnh hưởng của bước thời gian tau.

Chạy:  python 05_monte_carlo_luong_tu.py      (khoảng 10-20 phút với 4 lõi CPU)
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from common import (C, TEXT, TEXT2, NEUTRAL, E_EXACT_LI, E_HF_LIMIT_LI, setup_style,
                    save_fig, save_json, Timer)

setup_style()
rng = np.random.default_rng(20261127)
Z = 3.0
PAIRS = [(0, 1, 0.25), (0, 2, 0.5), (1, 2, 0.5)]  # (i, j, a_ij): 0,1 cùng spin lên


# ===========================================================================
# Hàm sóng và năng lượng cục bộ (tính đạo hàm GIẢI TÍCH, không xấp xỉ sai phân)
# ===========================================================================
def radial_parts(r, z1, z2, c):
    """Giá trị, đạo hàm bậc 1 và laplacian của phi1, phi2 theo bán kính r."""
    e1 = np.exp(-z1 * r)
    e2 = np.exp(-z2 * r)
    g = 1.0 - c * r
    phi1, d1, lap1 = e1, -z1 * e1, (z1 * z1 - 2 * z1 / r) * e1
    phi2 = g * e2
    d2 = (-c - z2 * g) * e2
    dd2 = (2 * c * z2 + z2 * z2 * g) * e2
    lap2 = dd2 + 2 * d2 / r
    return phi1, d1, lap1, phi2, d2, lap2


def evaluate(R, p, need_derivs=True, jastrow=True):
    """R: mảng (W, 3, 3) = (walker, electron, xyz). p = (z1, z2, d, b).

    Trả về: psi, drift F = grad ln|psi| (W,3,3), năng lượng cục bộ E_L (W,)."""
    z1, z2, d, b = p
    c = z1 - z2                                            # để hai orbital cùng "cusp" tại gốc
    r = np.linalg.norm(R, axis=2)                          # (W, 3) khoảng cách tới hạt nhân
    rhat = R / r[:, :, None]
    phi1, d1, lap1, phi2, d2, lap2 = radial_parts(r, z1, z2, c)

    # Định thức spin lên (electron 0, 1) và spin xuống (electron 2)
    Dup = phi1[:, 0] * phi2[:, 1] - phi1[:, 1] * phi2[:, 0]
    Ddn = phi1[:, 2]

    # Jastrow
    J = np.zeros(len(R))
    gJ = np.zeros_like(R)
    lapJ = np.zeros((len(R), 3))
    if jastrow:
        # electron - hạt nhân:  u(r) = -A r / (1 + d r),  A = Z - z1
        A = Z - z1
        den = 1.0 + d * r
        J += np.sum(-A * r / den, axis=1)
        if need_derivs:
            u1 = -A / den**2
            u2 = 2 * A * d / den**3
            gJ += u1[:, :, None] * rhat
            lapJ += u2 + 2 * u1 / r
    for i, j, a in (PAIRS if jastrow else []):
        rij_vec = R[:, i] - R[:, j]
        rij = np.linalg.norm(rij_vec, axis=1)
        den = 1.0 + b * rij
        J += a * rij / den
        if need_derivs:
            u1 = a / den**2                                  # u'(r)
            u2 = -2 * a * b / den**3                         # u''(r)
            gvec = (u1 / rij)[:, None] * rij_vec
            gJ[:, i] += gvec
            gJ[:, j] -= gvec
            lap_u = u2 + 2 * u1 / rij
            lapJ[:, i] += lap_u
            lapJ[:, j] += lap_u
    psi = Dup * Ddn * np.exp(J)
    if not need_derivs:
        return psi, None, None

    # Gradient và laplacian của các định thức, chia cho chính định thức
    gD = np.zeros_like(R)
    lapD = np.zeros((len(R), 3))
    gphi1 = d1[:, :, None] * rhat
    gphi2 = d2[:, :, None] * rhat
    gD[:, 0] = (gphi1[:, 0] * phi2[:, 1, None] - phi1[:, 1, None] * gphi2[:, 0]) / Dup[:, None]
    gD[:, 1] = (phi1[:, 0, None] * gphi2[:, 1] - gphi1[:, 1] * phi2[:, 0, None]) / Dup[:, None]
    lapD[:, 0] = (lap1[:, 0] * phi2[:, 1] - phi1[:, 1] * lap2[:, 0]) / Dup
    lapD[:, 1] = (phi1[:, 0] * lap2[:, 1] - lap1[:, 1] * phi2[:, 0]) / Dup
    gD[:, 2] = gphi1[:, 2] / Ddn[:, None]
    lapD[:, 2] = lap1[:, 2] / Ddn

    # psi = D * e^J  =>  lap(psi)/psi = lapD/D + 2 (gradD/D).gradJ + lapJ + |gradJ|^2
    lap_over_psi = lapD + 2 * np.sum(gD * gJ, axis=2) + lapJ + np.sum(gJ * gJ, axis=2)
    kinetic = -0.5 * lap_over_psi.sum(axis=1)
    V = -Z * np.sum(1.0 / r, axis=1)
    for i, j, _ in PAIRS:
        V += 1.0 / np.linalg.norm(R[:, i] - R[:, j], axis=1)
    return psi, gD + gJ, kinetic + V


def init_walkers(W):
    """Đặt 2 electron gần hạt nhân (1s) và 1 electron ở xa hơn (2s)."""
    R = rng.normal(size=(W, 3, 3))
    R[:, 0] *= 0.4
    R[:, 2] *= 0.4
    R[:, 1] *= 2.0
    return R


def blocking_error(x, n_blocks=50):
    """Sai số chuẩn bằng cách chia chuỗi thành các khối (vì các mẫu liên tiếp tương quan)."""
    x = np.asarray(x)
    n_blocks = min(n_blocks, len(x))
    nb = len(x) // n_blocks
    blocks = x[: nb * n_blocks].reshape(n_blocks, nb).mean(axis=1)
    return blocks.mean(), blocks.std(ddof=1) / np.sqrt(n_blocks)


# ===========================================================================
# PHẦN A. VMC
# ===========================================================================
def vmc(p, W=4000, n_steps=3000, n_equil=500, step=(0.35, 1.2, 0.35), keep_every=0,
        jastrow=True):
    """Metropolis: dịch từng electron một; chấp nhận với xác suất min(1, |psi'|^2/|psi|^2)."""
    R = init_walkers(W)
    psi, _, _ = evaluate(R, p, need_derivs=False, jastrow=jastrow)
    energies, samples, acc = [], [], np.zeros(3)
    for it in range(n_equil + n_steps):
        for e in range(3):
            Rn = R.copy()
            Rn[:, e] += step[e] * rng.normal(size=(W, 3))
            psin, _, _ = evaluate(Rn, p, need_derivs=False, jastrow=jastrow)
            accept = rng.random(W) < (psin / psi) ** 2
            R[accept] = Rn[accept]
            psi[accept] = psin[accept]
            if it >= n_equil:
                acc[e] += accept.mean()
        if it >= n_equil:
            _, _, EL = evaluate(R, p, jastrow=jastrow)
            energies.append(EL.mean())
            if keep_every and (it - n_equil) % keep_every == 0:
                samples.append(R.copy())
    E, err = blocking_error(energies)
    return {"E": E, "err": err, "acc": (acc / n_steps).tolist(), "trace": energies,
            "samples": np.concatenate(samples) if samples else None}


def optimize_params(p0, rounds=4):
    """Tối ưu tham số: lấy mẫu từ psi(p0), rồi cực tiểu năng lượng có trọng số
       E(p) = sum w E_L / sum w,  w = |psi_p / psi_p0|^2   ("lấy mẫu tương quan")."""
    p = np.array(p0, float)
    history = []
    for k in range(rounds):
        run = vmc(p, W=4000, n_steps=800, n_equil=300, keep_every=20)
        Rs = run["samples"]
        psi0, _, _ = evaluate(Rs, p, need_derivs=False)

        def objective(q):
            psi_q, _, EL = evaluate(Rs, q)
            w = (psi_q / psi0) ** 2
            if not np.all(np.isfinite(w)) or np.any(~np.isfinite(EL)):
                return 1e3
            w /= w.mean()
            e = np.sum(w * EL) / np.sum(w)
            var = np.sum(w * (EL - e) ** 2) / np.sum(w)
            neff = w.sum() ** 2 / np.sum(w**2) / len(w)   # tỉ lệ mẫu hiệu dụng
            penalty = 0.0 if neff > 0.5 else 10 * (0.5 - neff)
            return e + 0.01 * var + penalty                # thêm chút phương sai cho ổn định

        res = minimize(objective, p, method="Nelder-Mead",
                       options={"xatol": 1e-3, "fatol": 1e-5, "maxiter": 400})
        p = res.x
        history.append({"round": k + 1, "params": p.tolist(), "E_vmc_before": run["E"],
                        "err_before": run["err"]})
        print(f"    vòng {k+1}: E_VMC(trước) = {run['E']:.5f} ± {run['err']:.5f}  "
              f"-> tham số mới z1={p[0]:.4f} z2={p[1]:.4f} d={p[2]:.4f} b={p[3]:.4f}")
    return p, history


results = {}
print("A) Monte Carlo biến phân (VMC)")
p_start = np.array([2.69, 0.64, 1.0, 1.0])   # z1, z2 gần số mũ Slater của Clementi; d, b tùy ý
# Hàm thử KHÔNG có Jastrow (chỉ có định thức Slater, tham số ban đầu) để so sánh
no_jastrow = vmc(p_start, W=4000, n_steps=1500, jastrow=False)
print(f"  chỉ định thức Slater (không Jastrow): E = {no_jastrow['E']:.5f} ± {no_jastrow['err']:.5f} Ha")
with Timer() as t:
    p_opt, opt_hist = optimize_params(p_start)
    final = vmc(p_opt, W=4000, n_steps=4000)
print(f"  VMC tối ưu: E = {final['E']:.5f} ± {final['err']:.5f} Ha  ({t.seconds:.0f} s)")
print(f"  tỉ lệ chấp nhận theo electron: {np.round(final['acc'], 2)}")
results["vmc"] = {"E_no_jastrow": no_jastrow["E"], "err_no_jastrow": no_jastrow["err"],
                  "params_opt": p_opt.tolist(), "E": final["E"], "err": final["err"],
                  "acceptance": final["acc"], "opt_history": opt_hist}


# ===========================================================================
# PHẦN B. DMC nút cố định với lấy mẫu theo tầm quan trọng
# ===========================================================================
def limit_drift(F, tau, a=1.0):
    """Giới hạn vận tốc trôi gần nút (Umrigar, Nightingale & Runge 1993):
       v -> v * (-1 + sqrt(1 + 2 a v^2 tau)) / (a v^2 tau), tính cho từng electron."""
    v2 = np.sum(F * F, axis=2, keepdims=True)
    scale = (-1.0 + np.sqrt(1.0 + 2 * a * v2 * tau)) / (a * v2 * tau + 1e-300)
    return F * scale


def dmc(p, tau=0.01, W_target=4000, n_equil=2000, n_steps=10000, record_every=1):
    """Monte Carlo khuếch tán: mỗi walker trôi theo F, khuếch tán ngẫu nhiên, rồi
       nhân bản/biến mất theo trọng số exp(-tau (E_L - E_T)). Không cho walker vượt nút."""
    # Khởi động bằng mẫu VMC
    R = vmc(p, W=W_target, n_steps=50, n_equil=300, keep_every=50)["samples"][:W_target]
    psi, F, EL = evaluate(R, p)
    F = limit_drift(F, tau)
    E_T = EL.mean()
    E_est = E_T
    cut = 2.0 / np.sqrt(tau)            # giới hạn dao động E_L để trọng số không "nổ"
    E_trace, pop_trace = [], []
    sqrt_tau = np.sqrt(tau)
    acc_sum = 0.0
    for it in range(n_equil + n_steps):
        W = len(R)
        chi = rng.normal(size=R.shape)
        Rn = R + tau * F + sqrt_tau * chi
        psin, Fn, ELn = evaluate(Rn, p)
        Fn = limit_drift(Fn, tau)
        # Xác suất chấp nhận Metropolis-Hastings (giảm sai số bước thời gian)
        fwd = np.sum((Rn - R - tau * F) ** 2, axis=(1, 2))
        bwd = np.sum((R - Rn - tau * Fn) ** 2, axis=(1, 2))
        ratio = (psin / psi) ** 2 * np.exp(-(bwd - fwd) / (2 * tau))
        same_sign = np.sign(psin) == np.sign(psi)          # nút cố định: cấm vượt nút
        accept = same_sign & (rng.random(W) < np.minimum(1.0, ratio))
        acc_rate = accept.mean()
        acc_sum += acc_rate
        tau_eff = tau * max(acc_rate, 0.5)
        EL_old = EL.copy()
        R[accept], psi[accept], F[accept], EL[accept] = Rn[accept], psin[accept], Fn[accept], ELn[accept]
        # Trọng số nhánh (branching); E_L bị kẹp trong [E_est - cut, E_est + cut]
        S_old = np.clip(EL_old, E_est - cut, E_est + cut)
        S_new = np.clip(EL, E_est - cut, E_est + cut)
        w = np.exp(-tau_eff * (0.5 * (S_old + S_new) - E_T))
        n_copies = np.minimum((w + rng.random(W)).astype(int), 3)
        idx = np.repeat(np.arange(W), n_copies)
        R, psi, F, EL = R[idx], psi[idx], F[idx], EL[idx]
        # Điều khiển quần thể: điều chỉnh năng lượng thử E_T
        E_mean = EL.mean()
        E_est = 0.99 * E_est + 0.01 * E_mean if it > 50 else E_mean
        E_T = E_est - np.log(len(R) / W_target) / (50 * tau)
        if it >= n_equil:
            E_trace.append(E_mean)
            pop_trace.append(len(R))
    E, err = blocking_error(E_trace, n_blocks=40)
    return {"E": E, "err": err, "tau": tau, "trace": E_trace, "pop": pop_trace,
            "acc": acc_sum / (n_equil + n_steps)}


print("B) Monte Carlo khuếch tán (DMC), nút cố định")
dmc_runs = {}
for tau in [0.04, 0.02, 0.01, 0.005]:
    n_steps = int(300 / tau)            # cùng tổng "thời gian ảo" = 300 Ha^-1
    with Timer() as t:
        d = dmc(p_opt, tau=tau, n_equil=int(10 / tau), n_steps=n_steps)
    dmc_runs[tau] = d
    print(f"  tau = {tau:6.3f}:  E = {d['E']:.5f} ± {d['err']:.5f} Ha   "
          f"(chấp nhận {d['acc']:.3f}, {t.seconds:.0f} s)")

# Ngoại suy tuyến tính về tau -> 0
taus = np.array(sorted(dmc_runs))
Es = np.array([dmc_runs[t]["E"] for t in taus])
errs = np.array([dmc_runs[t]["err"] for t in taus])
wts = 1 / errs**2
A = np.vstack([np.ones_like(taus), taus]).T
cov = np.linalg.inv(A.T @ (wts[:, None] * A))
coef = cov @ (A.T @ (wts * Es))
E0, E0_err = coef[0], np.sqrt(cov[0, 0])
print(f"  ngoại suy tau -> 0:  E = {E0:.5f} ± {E0_err:.5f} Ha   (chính xác {E_EXACT_LI:.5f})")
results["dmc"] = {str(t): {"E": d["E"], "err": d["err"], "acc": d["acc"]} for t, d in dmc_runs.items()}
results["dmc_extrapolated"] = {"E": E0, "err": E0_err, "slope": coef[1]}

E_corr = E_EXACT_LI - E_HF_LIMIT_LI
for name, E in [("VMC", final["E"]), ("DMC (tau->0)", E0)]:
    print(f"  {name}: thu được {100*(E - E_HF_LIMIT_LI)/E_corr:.1f}% năng lượng tương quan")
results["percent_corr"] = {"VMC": 100 * (final["E"] - E_HF_LIMIT_LI) / E_corr,
                           "DMC": 100 * (E0 - E_HF_LIMIT_LI) / E_corr}

# ===========================================================================
# Hình
# ===========================================================================
fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.7))
ax = axes[0]
d = dmc_runs[0.01]
steps = np.arange(len(d["trace"])) * 0.01
ax.plot(steps, d["trace"], color=C["blue"], lw=0.6, alpha=0.8, label="DMC, τ = 0.01")
v = np.array(final["trace"])
ax.axhline(final["E"], color=C["orange"], lw=1.6, ls="-.", label=f"VMC: {final['E']:.4f}")
ax.axhline(E_EXACT_LI, color=TEXT, lw=1.2, ls="--", label=f"chính xác: {E_EXACT_LI:.4f}")
ax.axhline(E_HF_LIMIT_LI, color=NEUTRAL, lw=1.2, ls=":", label=f"giới hạn HF: {E_HF_LIMIT_LI:.4f}")
ax.set_xlabel("Thời gian ảo (Ha$^{-1}$)")
ax.set_ylabel("Năng lượng trung bình quần thể (Ha)")
ax.set_title("(a) Năng lượng DMC dao động quanh đáp án", loc="left")
ax.set_ylim(E_EXACT_LI - 0.03, E_HF_LIMIT_LI + 0.012)
ax.legend(fontsize=7.5, loc="upper right", ncol=2)

ax = axes[1]
ax.errorbar(taus, 1000 * (Es - E_EXACT_LI), yerr=1000 * errs, fmt="o", color=C["blue"],
            markersize=7, markeredgecolor="white", capsize=3, label="DMC")
tt = np.linspace(0, taus.max() * 1.05, 50)
ax.plot(tt, 1000 * (coef[0] + coef[1] * tt - E_EXACT_LI), color=C["blue"], lw=1.2, ls="--")
ax.errorbar([0], [1000 * (E0 - E_EXACT_LI)], yerr=[1000 * E0_err], fmt="D", color=C["orange"],
            markersize=7, markeredgecolor="white", capsize=3, label="ngoại suy τ → 0")
ax.axhline(0, color=TEXT, lw=1)
ax.set_xlabel("Bước thời gian τ (Ha$^{-1}$)")
ax.set_ylabel("E$_{DMC}$ − E$_{chính\\ xác}$ (mHa)")
ax.set_title("(b) Sai số bước thời gian và ngoại suy", loc="left")
ax.legend(fontsize=8)
fig.tight_layout()
save_fig(fig, "05_qmc_li")

# Hình minh họa: "đám mây" vị trí electron lấy mẫu từ |psi|^2
run = vmc(p_opt, W=600, n_steps=40, n_equil=300, keep_every=40)
Rs = run["samples"]
fig, ax = plt.subplots(figsize=(5.2, 5.0))
# Electron không có tên: chỉ tô màu theo spin (2 electron spin lên, 1 electron spin xuống)
up = np.concatenate([Rs[:, 0], Rs[:, 1]])
ax.scatter(up[:, 0], up[:, 1], s=3, color=C["blue"], alpha=0.45, label="electron spin lên (2 mỗi ảnh)")
ax.scatter(Rs[:, 2, 0], Rs[:, 2, 1], s=3, color=C["orange"], alpha=0.55,
           label="electron spin xuống (1 mỗi ảnh)")
ax.plot([0], [0], "k+", markersize=12, mew=2)
ax.set_aspect("equal")
ax.set_xlim(-8, 8)
ax.set_ylim(-8, 8)
ax.set_xlabel("x (bohr)")
ax.set_ylabel("y (bohr)")
ax.set_title("600 \"ảnh chụp\" nguyên tử Li lấy mẫu từ |ψ|²", loc="left")
ax.legend(fontsize=8, loc="upper right", markerscale=3, frameon=True, framealpha=0.9)
save_fig(fig, "05_dam_may_electron")

save_json(results, "05_qmc")
