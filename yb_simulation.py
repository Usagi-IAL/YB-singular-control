    """
Numerical simulation for:
  Wu, Lyuchengfei (2026).
  "Reconsidering the Dynamics of Division of Labor:
   Singular Control and the Yang-Borland Model."

Reproduces all figures in Section 7 of the paper.

Requirements: numpy, matplotlib (standard Anaconda/pip install)
Run: python yb_simulation.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")        # change to "TkAgg" for interactive display
import matplotlib.pyplot as plt

# ── Global parameters ─────────────────────────────────────────────────────────
M      = 100.0   # number of goods / individuals
CLAMP  = 2.0     # maximum |dn/dt| (regularisation)
EPS    = 1e-10   # initial G_0 (small positive; G_0 = 0 exactly is singular)
DT     = 0.3     # Euler step size


def phi(n, a, k, m=M):
    """Static-cost function Phi(n; a, k)."""
    return np.log(k) + (a - 2.0) * np.log(n) + 1.0 / n - 2.0


def D_coeff(n, a, k, m=M):
    """Denominator D(n) = Phi^2 + Phi' - Phi/m  (Kelley condition: D > 0)."""
    ph  = phi(n, a, k, m)
    phd = (a - 2.0) / n - 1.0 / n**2        # d Phi / d n
    return ph**2 + phd - ph / m


def run(a, k, r, m=M, T=380.0):
    """
    Integrate the corrected singular-arc ODE:

        dn/dt = [ r * Phi(n) + (a * exp(n/m) / G) * (1 - m * Phi(n)) ] / D(n)

    together with dG/dt = exp(n/m), G(0) = EPS, n(0) = 1.

    Returns arrays (ts, ns, Gs).
    """
    n, G = 1.0, EPS
    ns, Gs, ts = [n], [G], [0.0]
    t = 0.0

    while t < T and n < m - 0.005:
        ph = phi(n, a, k, m)
        Dv = D_coeff(n, a, k, m)
        if abs(Dv) < 1e-9:
            Dv = 1e-9

        ndot = (r * ph + (a * np.exp(n / m) / G) * (1.0 - m * ph)) / Dv
        ndot = np.clip(ndot, -CLAMP, CLAMP)

        n = np.clip(n + ndot * DT, 1.0, m)
        G += np.exp(n / m) * DT
        t += DT
        ns.append(n); Gs.append(G); ts.append(t)

    # Extend at absorbing state n = m
    if ts[-1] < T - 1.0:
        for te in np.arange(ts[-1] + DT, T + DT, DT):
            Gs.append(Gs[-1] + np.exp(m / m) * DT)
            ns.append(m)
            ts.append(te)

    return np.array(ts), np.array(ns), np.array(Gs)


# ── Derived quantities (YB p.476-477) ─────────────────────────────────────────

def log_u(n, G, a, k, m=M):
    """Log real income:  u_t = C(n_t) * G_t^{am}."""
    n = max(n, 1.001)
    return (-a * m * np.log(m)
            + (n - 1) * np.log(k)
            + (1 - 2*n + a*n) * np.log(n)
            - a * n
            + a * m * np.log(max(G, EPS)))


def log_E(n, G, a, m=M):
    """Log trade dependence E_t = 2(n-1)(L_it)^a / n  (YB p.476)."""
    n = max(n, 1.001)
    return (np.log(2) + np.log(n - 1)
            + (a - 1) * np.log(n)
            - a * n / m
            + a * np.log(max(G, EPS))
            - a * np.log(m))


def log_D(n, G, a, m=M):
    """Log endogenous comparative advantage D_t = (L_it)^a / n  (YB p.477)."""
    n = max(n, 1.001)
    return ((a - 1) * np.log(n)
            - a * n / m
            + a * np.log(max(G, EPS))
            - a * np.log(m))


def norm0(arr):
    """Normalise array to start at 0."""
    return arr - arr[0]


# ── Figure helpers ─────────────────────────────────────────────────────────────

STYLES = [('-', 'C0'), ('--', 'C1'), ('-.', 'C2'), (':', 'C3')]


def make_figure(fname, title, rows, param_fn, T0, panels):
    """
    Parameters
    ----------
    rows     : list of (param_value, label)
    param_fn : callable(param_value) -> (a, k, r)
    T0       : normalisation / display start time
    panels   : list of (ylabel, fn(ts, ns, Gs, a, k) -> array)
    """
    fig, axes = plt.subplots(1, len(panels), figsize=(3.9 * len(panels), 4.2))
    if len(panels) == 1:
        axes = [axes]

    for (pv, lbl), (ls, c) in zip(rows, STYLES):
        a2, k2, r2 = param_fn(pv)
        ts, ns, Gs = run(a2, k2, r2)
        mask = ts >= T0
        mask[-2:] |= (mask.sum() < 2)
        ts2, ns2, Gs2 = ts[mask], ns[mask], Gs[mask]

        for ax, (tit, fn) in zip(axes, panels):
            y = fn(ts2, ns2, Gs2, a2, k2)
            ax.plot(ts2, y, ls=ls, c=c, lw=2, label=lbl)

    if panels[0][0].startswith('$n_t$'):
        axes[0].axhline(M, ls=':', c='gray', lw=0.7)

    for ax, (tit, _) in zip(axes, panels):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8.5)
        ax.set_xlabel('$t$')
        ax.set_title(tit, fontsize=8.8)

    fig.suptitle(title, y=1.02, fontsize=10)
    plt.tight_layout()
    plt.savefig(fname, dpi=140, bbox_inches='tight')
    plt.close()
    print(f"Saved {fname}")


# ── Panel definitions ──────────────────────────────────────────────────────────

p_n = ('$n_t$', lambda t, n, G, a, k: n)
p_u = (r'$\Delta\log u_t$',
       lambda t, n, G, a, k:
           norm0(np.array([log_u(ni, Gi, a, k) for ni, Gi in zip(n, G)])))
p_E = (r'$\Delta\log E_t = 2(n_t{-}1)L_{it}^a/n_t$',
       lambda t, n, G, a, k:
           norm0(np.array([log_E(ni, Gi, a) for ni, Gi in zip(n, G)])))
p_D = (r'$\Delta\log D_t = L_{it}^a/n_t$',
       lambda t, n, G, a, k:
           norm0(np.array([log_D(ni, Gi, a) for ni, Gi in zip(n, G)])))
p_S = (r'$S_t = 1-k/n_t$',
       lambda t, n, G, a, k: 1.0 - k / n)


# ── Figure 1: Phi(n) and D(n) ─────────────────────────────────────────────────

def figure_phi_D():
    """
    Left panel : Phi(n) < 0 for all relevant (a,k) parameter combinations.
    Right panel: D(n)  > 0 for the same combinations (Kelley condition).
    """
    ns = np.linspace(1.01, M, 600)
    param_grid = [
        (1.2, 0.3, '-',  'C0', r'$a=1.2,\,k=0.3$'),
        (1.2, 0.8, '--', 'C0', r'$a=1.2,\,k=0.8$'),
        (1.5, 0.3, '-',  'C1', r'$a=1.5,\,k=0.3$'),
        (1.5, 0.8, '--', 'C1', r'$a=1.5,\,k=0.8$'),
        (2.0, 0.3, '-',  'C2', r'$a=2.0,\,k=0.3$'),
        (2.0, 0.8, '--', 'C2', r'$a=2.0,\,k=0.8$'),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for a, k, ls, c, lbl in param_grid:
        Ph = np.array([phi(n, a, k) for n in ns])
        Dv = np.array([D_coeff(n, a, k) for n in ns])
        axes[0].plot(ns, Ph, ls=ls, c=c, lw=1.8, label=lbl)
        axes[1].plot(ns, Dv, ls=ls, c=c, lw=1.8, label=lbl)
    for ax, ylab, ttl in zip(
            axes,
            [r'$\Phi(n)$', r"$D(n)=\Phi^2+\Phi'-\Phi/m$"],
            [r'$\Phi(n_t)<0$ for all relevant parameters',
             r'$D(n_t)>0$ (Kelley condition holds)']):
        ax.axhline(0, c='k', lw=0.8, ls=':')
        ax.set_xlabel('$n$')
        ax.set_ylabel(ylab)
        ax.set_title(ttl, fontsize=9.5)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7.5, ncol=2)
    plt.tight_layout()
    plt.savefig('yb_phiD.png', dpi=140, bbox_inches='tight')
    plt.close()
    print("Saved yb_phiD.png")


# ── Figure 2: variation in r ───────────────────────────────────────────────────

make_figure(
    'yb_dyn_r.png',
    r'Variation in $r$  ($a=1.2$, $k=0.6$, $m=100$, from $t=50$)',
    [(0.02, '$r=0.02$'), (0.05, '$r=0.05$'),
     (0.10, '$r=0.10$'), (0.20, '$r=0.20$')],
    lambda r: (1.2, 0.6, r),
    T0=50.0,
    panels=[p_n, p_u]
)

# ── Figure 3: variation in a ───────────────────────────────────────────────────

make_figure(
    'yb_dyn_a.png',
    r'Variation in $a$  ($k=0.6$, $r=0.05$, $m=100$, from $t=5$)',
    [(1.1, '$a=1.1$'), (1.4, '$a=1.4$'),
     (1.7, '$a=1.7$'), (2.0, '$a=2.0$')],
    lambda a: (a, 0.6, 0.05),
    T0=5.0,
    panels=[p_n, p_u, p_E, p_D]
)

# ── Figure 4: variation in k ───────────────────────────────────────────────────

def make_k_figure():
    """k-variation with zoomed y-axis on E and D panels."""
    a, r = 1.5, 0.05
    rows = [(0.2, '$k=0.2$'), (0.4, '$k=0.4$'),
            (0.6, '$k=0.6$'), (0.75, '$k=0.75$')]
    T0 = 5.0

    # Collect data to set zoomed y-limits
    data = {}
    for (k, lbl) in rows:
        ts, ns, Gs = run(a, k, r)
        mask = ts >= T0; mask[-2:] |= (mask.sum() < 2)
        ts2, ns2, Gs2 = ts[mask], ns[mask], Gs[mask]
        lE = norm0(np.array([log_E(n, G, a) for n, G in zip(ns2, Gs2)]))
        lD = norm0(np.array([log_D(n, G, a) for n, G in zip(ns2, Gs2)]))
        data[k] = dict(ts=ts2, ns=ns2, Gs=Gs2,
                       lu=norm0(np.array([log_u(n, G, a, k) for n, G in zip(ns2, Gs2)])),
                       lE=lE, lD=lD, Sv=1.0 - k / ns2)

    E_all = np.stack([data[k]['lE'] for k, _ in rows])
    D_all = np.stack([data[k]['lD'] for k, _ in rows])
    E_max, D_max = E_all.max(), D_all.max()
    spread_E = (E_all.max(0) - E_all.min(0)).max()
    spread_D = (D_all.max(0) - D_all.min(0)).max()

    fig, axes = plt.subplots(1, 5, figsize=(18, 4.2))
    for (k, lbl), (ls, c) in zip(rows, STYLES):
        d = data[k]
        for ax, y in zip(axes, [d['ns'], d['lu'], d['lE'], d['lD'], d['Sv']]):
            ax.plot(d['ts'], y, ls=ls, c=c, lw=2, label=lbl)

    axes[0].axhline(M, ls=':', c='gray', lw=0.7)
    # Zoom E and D panels
    axes[2].set_ylim(E_max - 3 * spread_E * 6, E_max + spread_E * 0.5)
    axes[3].set_ylim(D_max - 3 * spread_D * 6, D_max + spread_D * 0.5)

    titles = ['$n_t$', r'$\Delta\log u_t$',
              r'$\Delta\log E_t$  [zoomed]',
              r'$\Delta\log D_t$  [zoomed]',
              r'$S_t = 1-k/n_t$']
    for ax, tit in zip(axes, titles):
        ax.grid(alpha=0.3); ax.legend(fontsize=8.5)
        ax.set_xlabel('$t$'); ax.set_title(tit, fontsize=9)

    fig.suptitle(r'Variation in $k$  ($a=1.5$, $r=0.05$, $m=100$, from $t=5$)',
                 y=1.02, fontsize=10)
    plt.tight_layout()
    plt.savefig('yb_dyn_k.png', dpi=140, bbox_inches='tight')
    plt.close()
    print("Saved yb_dyn_k.png")


make_k_figure()

# ── Figure 5: T*(k, a) parameter space ────────────────────────────────────────

def T_star(a, k, r, T_max=800.0):
    """Time to full specialisation; returns T_max if not reached."""
    ts, ns, _ = run(a, k, r, T=T_max)
    idx = np.where(ns >= M - 0.1)[0]
    return ts[idx[0]] if len(idx) else T_max


def figure_Tstar():
    """
    Coloured heatmap of T*(k,a) for four values of r.
    Bright (yellow) = fast specialisation; dark (purple) = slow / incomplete.
    Cyan contours at T* in {50,75,100,150,200,250}.
    White dashed line at a = 2.
    Only cells where D(1;a,k) > 0 are plotted (Kelley condition guaranteed).
    """
    T_MAX = 350.0
    N_K, N_A = 50, 50
    ks = np.linspace(0.05, 0.95, N_K)
    a_grid = np.linspace(1.05, 2.8, N_A)
    r_vals = [0.02, 0.05, 0.10, 0.20]
    contour_levels = [50, 75, 100, 150, 200, 250]

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))

    for ax, r in zip(axes, r_vals):
        T_mat = np.full((N_A, N_K), np.nan)
        for ia, a in enumerate(a_grid):
            for ik, k in enumerate(ks):
                # Only compute where Kelley condition D(1)>0
                if D_coeff(1.01, a, k) > 0:
                    Ts = T_star(a, k, r, T_max=T_MAX + 50)
                    T_mat[ia, ik] = min(Ts, T_MAX) if not np.isnan(Ts) else T_MAX

        im = ax.pcolormesh(ks, a_grid, T_mat,
                           cmap='viridis_r', vmin=0, vmax=T_MAX,
                           shading='auto')
        # Cyan contours
        valid = ~np.isnan(T_mat)
        if valid.any():
            ax.contour(ks, a_grid, np.where(valid, T_mat, T_MAX),
                       levels=contour_levels,
                       colors='cyan', linewidths=0.9, alpha=0.85)
        # White dashed line at a = 2
        ax.axhline(2.0, ls='--', c='white', lw=1.5)
        plt.colorbar(im, ax=ax, label=r'$T^*$', fraction=0.046, pad=0.04)
        ax.set_xlabel('$k$')
        ax.set_ylabel('$a$')
        ax.set_title(f'$r = {r}$', fontsize=10)

    fig.suptitle(
        r'Time $T^*$ to full specialization over $(k,a)$ parameter space',
        y=1.02, fontsize=11)
    plt.tight_layout()
    plt.savefig('yb_Tstar.png', dpi=140, bbox_inches='tight')
    plt.close()
    print("Saved yb_Tstar.png")


figure_phi_D()
figure_Tstar()
print("All figures generated.")

    
