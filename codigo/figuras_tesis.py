"""
Genera todas las figuras del Capitulo 4 de la tesis a partir de los resultados
guardados: los CSV de litio_confinado_hf.py (base de STO) y los JSON de
malla/barrido.py (solucion sin base). Uso:

    python figuras_tesis.py                      # usa ../datos y escribe en ../figuras
    python figuras_tesis.py ../datos/litio_V0_5.csv ../datos/litio_V0_1.csv \
           --malla ../datos/malla/barrido_V0_5.json ../datos/malla/barrido_V0_1.json \
           --dir ../figuras

El primer CSV es el barrido principal (V0 = 5); el segundo, opcional, se agrega
como comparacion. Con --malla se dibujan ademas la figura del error de la base
(fig_6) y los puntos de la malla en la figura de entropias en momentos (fig_7);
la altura V0 de cada JSON se lee de su nombre (..._V0_<n>.json).

Figura de la tesis  ->  archivo
  4.1  fig_1_energia_total.png          energia del estado base y Ozmen et al.
  4.2  fig_6_error_base.png             E_av - E_av(malla), este trabajo y Ozmen et al.
  4.3  fig_2_energias_orbitales.png     eps_1s y eps_2s
  4.4  fig_3_radio_y_pout.png           <r> y P_out
  4.5  fig_4_optimizacion_zetas.png     exponentes optimos del bloque s
  4.6  fig_5_presion_cuantica.png       presion: Hellmann-Feynman y diferencias finitas
  4.7  fig_1_energia_total_2s_2p.png    1s^2 2s, 1s^2 2p y energia de excitacion
  4.8  nubes_electronicas_Rc_*.png      mapas de densidad del electron de valencia
  4.9  fig_2_entropia_shannon.png       S_r de los orbitales
  4.10 fig_7_entropia_momentos.png      S_p, S_r + S_p y cota BBM

Los nombres de archivo coinciden con los \\includegraphics de final.tex.
"""

import argparse
import csv
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.colors import LogNorm
import numpy as np

from litio_confinado_hf import Basis, N_PER_BLOCK, scf_eav

RAIZ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATOS = os.path.join(RAIZ, "datos")
CSV_DEFECTO = [os.path.join(DATOS, "litio_V0_5.csv"), os.path.join(DATOS, "litio_V0_1.csv")]
MALLA_DEFECTO = [os.path.join(DATOS, "malla", "barrido_V0_5.json"),
                 os.path.join(DATOS, "malla", "barrido_V0_1.json")]

plt.rcParams.update({"font.family": "serif", "axes.labelsize": 13, "font.size": 11,
                     "legend.fontsize": 10, "figure.dpi": 100})

# Ozmen et al., Physica B 656 (2023) 414775, Tabla 1 (energias medias)
OZMEN = {
    1.0: {"Rc": [0.8, 1, 1.5, 2, 2.5, 3, 4, 6, 10],
          "2s": [-6.0701, -6.2489, -6.4612, -6.7123, -6.9688, -7.1304, -7.2924, -7.3976, -7.4281],
          "2p": [-6.0052, -6.2058, -6.5582, -6.8464, -7.0259, -7.1341, -7.2510, -7.3309, -7.3833]},
    5.0: {"Rc": [0.8, 1, 1.5, 2, 2.5, 3, 4, 6, 10],
          "2s": [-1.2519, -2.0080, -4.6135, -6.0210, -6.6637, -6.9822, -7.2547, -7.3951, -7.4178],
          "2p": [-1.6850, -3.5088, -5.7165, -6.5071, -6.8570, -7.0221, -7.2098, -7.3234, -7.3568]},
}


def load(path):
    rows = list(csv.DictReader(open(path)))
    out = {}
    for k in rows[0]:
        if k.startswith("zeta") or k.startswith("conv"):
            out[k] = [r[k] for r in rows]
        else:
            out[k] = np.array([float(r[k]) for r in rows])
    order = np.argsort(out["Rc"])
    return {k: (np.asarray(v)[order] if not isinstance(v, list) else [v[i] for i in order])
            for k, v in out.items()}


def axis_rc(ax):
    ax.set_xlabel(r"Radio de confinamiento $R_c$ (u.a.)")
    ax.grid(True, linestyle="--", alpha=0.5)


def save(fig, d, name):
    fig.tight_layout()
    fig.savefig(os.path.join(d, name), dpi=300)
    plt.close(fig)


BBM = 3.0 * (1.0 + np.log(np.pi))   # cota de Bialynicki-Birula y Mycielski


def cargar_malla(paths):
    """{V0: {"2s": {Rc: dict}, "2p": {...}}} a partir de los JSON de barrido.py."""
    out = {}
    for p in paths:
        m = re.search(r"V0_(\d+(?:\.\d+)?)", os.path.basename(p))
        d = json.load(open(p))
        out[float(m.group(1))] = {c: {float(k): v for k, v in d[c].items()} for c in ("2s", "2p")}
    return out


def _malla_xy(M, conf, clave):
    if M is None or not M[conf] or clave not in next(iter(M[conf].values())):
        return None, None
    R = np.array(sorted(M[conf]))
    return R, np.array([M[conf][r][clave] for r in R])


def _leyenda(ax, M, ncol=1, fontsize=10):
    axis_rc(ax)
    if M is not None:
        ax.plot([], [], "o", mfc="none", mec="k", ms=7, label="malla (sin base)")
    ax.legend(loc="upper center", ncol=ncol, fontsize=fontsize, framealpha=0.9)


def fig_entropia_momentos(sets, malla, d):
    """S_p de los orbitales y sumas S_r + S_p frente a la cota BBM."""
    ncol = len(sets)
    fig, axs = plt.subplots(3, ncol, figsize=(5.6 * ncol, 12.5), squeeze=False)
    for c, (D, V0) in enumerate(sets):
        M = malla.get(V0)
        Rc = D["Rc"]
        # (a) entropia en momentos de cada orbital
        ax = axs[0, c]
        ax.plot(Rc, D["Sp_1s_2s"], "^-", color="gray", ms=4, label=r"$1s$ (en $1s^2 2s$)")
        ax.plot(Rc, D["Sp_val_2s"], "o-", color="darkgreen", ms=4, label=r"$2s$")
        ax.plot(Rc, D["Sp_val_2p"], "s-", color="purple", ms=4, label=r"$2p$ (promedio en $m$)")
        for conf, k, col, mk in (("2s", "Sp_1s", "gray", "^"), ("2s", "Sp_nl", "darkgreen", "o"),
                                 ("2p", "Sp_nl", "purple", "s")):
            x, y = _malla_xy(M, conf, k)
            if x is not None:
                ax.plot(x, y, mk, mfc="none", mec="k", ms=7, mew=0.9)
        ax.set_ylabel(r"Entropía en momentos $S_p$")
        ax.set_title(rf"$V_0 = {V0:g}$")
        ax.set_ylim(-0.5, 8.3)
        _leyenda(ax, M, ncol=2)
        # (b) suma de cada orbital
        ax = axs[1, c]
        ax.plot(Rc, D["S_1s_2s"] + D["Sp_1s_2s"], "^-", color="gray", ms=4, label=r"$1s$ (en $1s^2 2s$)")
        ax.plot(Rc, D["S_val_2s"] + D["Sp_val_2s"], "o-", color="darkgreen", ms=4, label=r"$2s$")
        ax.plot(Rc, D["S_val_2p"] + D["Sp_val_2p"], "s-", color="purple", ms=4, label=r"$2p$ (promedio en $m$)")
        ax.plot(Rc, D["S_val_2p_pz"] + D["Sp_val_2p_pz"], "s--", color="purple", ms=3, alpha=0.5,
                label=r"$2p_z$")
        for conf, tag, col, mk in (("2s", "1s", "gray", "^"), ("2s", "nl", "darkgreen", "o"),
                                   ("2p", "nl", "purple", "s")):
            x, y1 = _malla_xy(M, conf, f"Sr_{tag}")
            if x is not None:
                y2 = _malla_xy(M, conf, f"Sp_{tag}")[1]
                ax.plot(x, y1 + y2, mk, mfc="none", mec="k", ms=7, mew=0.9)
        ax.axhline(BBM, color="crimson", lw=1.2, ls="-.", label=r"cota BBM $3(1+\ln\pi)$")
        ax.axhline(np.log(32 * np.pi ** 3) - 1 / 3, color="gray", lw=0.8, ls=":",
                   label=r"$1s$ hidrogenoide")
        ax.set_ylabel(r"$S_r + S_p$ (orbitales)")
        ax.set_ylim(6.3, 9.45)
        _leyenda(ax, M, ncol=2, fontsize=9)
        # (c) suma de la densidad total, estado base vs excitado
        ax = axs[2, c]
        ax.plot(Rc, D["Sr_tot_2s"] + D["Sp_tot_2s"], "o-", color="navy", ms=4, label=r"$1s^2 2s$")
        ax.plot(Rc, D["Sr_tot_2p"] + D["Sp_tot_2p"], "s--", color="crimson", ms=4, label=r"$1s^2 2p$")
        for conf, col, mk in (("2s", "navy", "o"), ("2p", "crimson", "s")):
            x, y1 = _malla_xy(M, conf, "Sr_tot")
            if x is not None:
                y2 = _malla_xy(M, conf, "Sp_tot")[1]
                ax.plot(x, y1 + y2, mk, mfc="none", mec="k", ms=7, mew=0.9)
        ax.axhline(BBM, color="crimson", lw=1.2, ls="-.", label=r"cota BBM")
        ax.set_ylabel(r"$S_r + S_p$ (densidad total, normalizada a 1)")
        ax.set_ylim(6.3, 8.75)
        _leyenda(ax, M, ncol=2)
    for ax in axs.ravel():
        ax.set_xscale("log")
        ax.set_xticks([0.8, 1, 1.5, 2, 3, 5, 10, 15])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.minorticks_off()
    save(fig, d, "fig_7_entropia_momentos.png")


def fig_error_base(sets, malla, d):
    """Diferencia entre las energias con base y el limite de Hartree-Fock de la malla,
    para este trabajo y para Ozmen et al., en los radios de la Tabla 1 de Ozmen."""
    orden = sorted([s for s in sets if s[1] in malla and s[1] in OZMEN], key=lambda t: -t[1])
    if not orden:
        return
    with plt.rc_context({"mathtext.fontset": "dejavusans", "font.size": 12, "axes.labelsize": 12}):
        fig, axs = plt.subplots(1, len(orden), figsize=(6 * len(orden), 4.6), squeeze=False)
        for ax, (D, V0) in zip(axs[0], orden):
            R = OZMEN[V0]["Rc"]
            M = malla[V0]
            num = np.array([(M["2s"][r]["E"], M["2p"][r]["E"]) for r in R])
            idx = [int(np.argmin(abs(D["Rc"] - r))) for r in R]
            t = np.array([(D["E_2s"][i], D["E_2p"][i]) for i in idx])
            o = np.array(list(zip(OZMEN[V0]["2s"], OZMEN[V0]["2p"])))
            ax.axhspan(-1000, 0, color="0.92", zorder=0)
            ax.axhline(0, color="k", lw=1)
            ax.plot(R, 1e3 * (t[:, 0] - num[:, 0]), "o-", color="navy", label=r"Este trabajo (STOs), $1s^2 2s$")
            ax.plot(R, 1e3 * (t[:, 1] - num[:, 1]), "s--", color="crimson", label=r"Este trabajo (STOs), $1s^2 2p$")
            ax.plot(R, 1e3 * (o[:, 0] - num[:, 0]), "x", color="navy", ms=9, mew=1.6, label=r"Özmen et al., $1s^2 2s$")
            ax.plot(R, 1e3 * (o[:, 1] - num[:, 1]), "+", color="crimson", ms=11, mew=1.6, label=r"Özmen et al., $1s^2 2p$")
            ax.set_title(rf"$V_0 = {V0:g}$")
            ax.set_xlabel(r"Radio de confinamiento $R_c$ (u.a.)")
            ax.set_ylabel(r"$E_{av} - E_{av}^{\,\mathrm{malla}}$ (mHartree)")
            ax.grid(True, ls="--", alpha=0.5)
            vals = np.concatenate([1e3 * (t - num).ravel(), 1e3 * (o - num).ravel()])
            ax.set_ylim(vals.min() * 1.12, max(vals.max(), 10) * 1.3)
            ax.text(0.97, 0.04, "por debajo del límite\nde Hartree-Fock", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=10, color="0.35")
        h, l = axs[0, 0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=4, fontsize=10, frameon=False)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        fig.savefig(os.path.join(d, "fig_6_error_base.png"), dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Figuras del Capitulo 4 de la tesis")
    ap.add_argument("csv", nargs="*", default=CSV_DEFECTO,
                    help="CSV de litio_confinado_hf.py: el principal (V0 = 5) y, opcional, el de comparacion")
    ap.add_argument("--dir", default=os.path.join(RAIZ, "figuras"), help="carpeta de salida")
    ap.add_argument("--malla", nargs="*", default=None,
                    help="JSON de malla/barrido.py; V0 se lee del nombre ...V0_<n>.json")
    a = ap.parse_args()
    if a.malla is None:
        a.malla = [f for f in MALLA_DEFECTO if os.path.exists(f)]
    os.makedirs(a.dir, exist_ok=True)
    A = load(a.csv[0])
    Bc = load(a.csv[1]) if len(a.csv) > 1 else None
    V0a = A["V0"][0]
    V0b = Bc["V0"][0] if Bc else None
    Rc = A["Rc"]

    # ---- Fig 4.1: energia total del estado base ----
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(Rc, A["E_2s"], "o-", color="navy", ms=4, label=rf"Este trabajo, $V_0={V0a:g}$")
    if V0a in OZMEN:
        ax.plot(OZMEN[V0a]["Rc"], OZMEN[V0a]["2s"], "x", color="navy", ms=8,
                label=rf"Özmen et al. (2023), $V_0={V0a:g}$")
    if Bc is not None:
        ax.plot(Bc["Rc"], Bc["E_2s"], "s--", color="darkorange", ms=4, label=rf"Este trabajo, $V_0={V0b:g}$")
        if V0b in OZMEN:
            ax.plot(OZMEN[V0b]["Rc"], OZMEN[V0b]["2s"], "+", color="darkorange", ms=9,
                    label=rf"Özmen et al. (2023), $V_0={V0b:g}$")
    ax.axhline(-7.432727, color="gray", lw=0.8, ls=":", label="Átomo libre (HF)")
    ax.set_ylabel(r"$E_{av}(1s^2 2s)$ (Hartree)")
    axis_rc(ax); ax.legend()
    save(fig, a.dir, "fig_1_energia_total.png")

    sets = [(A, V0a)] + ([(Bc, V0b)] if Bc is not None else [])
    ncol = len(sets)

    # ---- Fig 4.3: energias orbitales ----
    fig, axs = plt.subplots(1, ncol, figsize=(5.5 * ncol, 4.5), squeeze=False)
    for ax, (D, V0) in zip(axs[0], sets):
        ax.plot(D["Rc"], D["eps1s_2s"], "o-", color="darkred", ms=4, label=r"$\varepsilon_{1s}$")
        ax.plot(D["Rc"], D["eps2s"], "s-", color="darkgreen", ms=4, label=r"$\varepsilon_{2s}$")
        ax.axhline(0, color="k", lw=0.7)
        ax.axhline(V0, color="gray", lw=0.8, ls=":", label=r"$V_0$")
        ax.set_ylabel("Energía orbital (Hartree)")
        ax.set_title(rf"$1s^2 2s$, $V_0 = {V0:g}$")
        axis_rc(ax); ax.legend()
    save(fig, a.dir, "fig_2_energias_orbitales.png")

    # ---- Fig 4.4: radio medio y probabilidad de penetracion ----
    fig, axs = plt.subplots(2, ncol, figsize=(5.5 * ncol, 8), squeeze=False)
    for c, (D, V0) in enumerate(sets):
        ax1, ax2 = axs[0, c], axs[1, c]
        ax1.plot(D["Rc"], D["r_1s_2s"], "o-", ms=4, label=r"$\langle r\rangle_{1s}$")
        ax1.plot(D["Rc"], D["r_val_2s"], "s-", ms=4, label=r"$\langle r\rangle_{2s}$")
        ax1.plot(D["Rc"], D["Rc"], "k:", lw=0.8, label=r"$\langle r\rangle = R_c$")
        ax1.set_ylim(0, max(D["r_val_2s"]) * 1.15)
        ax1.set_ylabel(r"$\langle r\rangle$ (u.a.)"); ax1.set_title(rf"$1s^2 2s$, $V_0 = {V0:g}$")
        axis_rc(ax1); ax1.legend()
        ax2.semilogy(D["Rc"], np.maximum(D["Pout_1s_2s"], 1e-16), "o-", ms=4, label=r"$P_{out}(1s)$")
        ax2.semilogy(D["Rc"], np.maximum(D["Pout_val_2s"], 1e-16), "s-", ms=4, label=r"$P_{out}(2s)$")
        ax2.set_ylabel(r"$P_{out}$"); ax2.set_ylim(1e-12, 2)
        axis_rc(ax2); ax2.legend()
    save(fig, a.dir, "fig_3_radio_y_pout.png")

    # ---- Fig 4.5: exponentes zeta del bloque s (estado base) ----
    Z = np.array([[float(v) for v in z.split()] for z in A["zeta_2s"]])
    fig, ax = plt.subplots(figsize=(7, 5))
    for i in range(Z.shape[1]):
        ax.semilogy(Rc, Z[:, i], "o-", ms=3.5, label=rf"$\zeta_{i + 1}$")
    ax.set_ylabel(r"Exponente $\zeta$ (u.a.)")
    ax.set_title(rf"Exponentes óptimos del bloque $s$, $1s^2 2s$, $V_0 = {V0a:g}$")
    axis_rc(ax); ax.legend(ncol=5)
    save(fig, a.dir, "fig_4_optimizacion_zetas.png")

    # ---- Fig 4.6: presion (Hellmann-Feynman vs diferencias finitas) ----
    fig, ax = plt.subplots(figsize=(7, 5))
    for D, V0, c in ((A, V0a, "purple"), (Bc, V0b, "teal")):
        if D is None:
            continue
        ax.semilogy(D["Rc"], D["P_HF_2s"], "o-", color=c, ms=4, label=rf"Hellmann-Feynman, $V_0={V0:g}$")
        m = D["P_FD_2s"] > 0
        ax.semilogy(D["Rc"][m], D["P_FD_2s"][m], "x", color=c, ms=7,
                    label=rf"Diferencias finitas, $V_0={V0:g}$")
    ax.set_ylabel(r"Presión $P$ (Hartree/bohr$^3$)")
    ax2 = ax.secondary_yaxis("right", functions=(lambda p: p * 29421.02648438959,
                                                  lambda g: g / 29421.02648438959))
    ax2.set_ylabel("Presión (GPa)")
    axis_rc(ax); ax.legend()
    save(fig, a.dir, "fig_5_presion_cuantica.png")

    # ---- Fig 4.7: 1s^2 2s vs 1s^2 2p y energia de excitacion ----
    fig, axs = plt.subplots(1, ncol + 1, figsize=(5.2 * (ncol + 1), 4.5))
    for ax, (D, V0) in zip(axs[:ncol], sets):
        ax.plot(D["Rc"], D["E_2s"], "o-", color="navy", ms=4, label=r"$1s^2 2s$")
        ax.plot(D["Rc"], D["E_2p"], "s--", color="crimson", ms=4, label=r"$1s^2 2p$")
        if V0 in OZMEN:
            ax.plot(OZMEN[V0]["Rc"], OZMEN[V0]["2s"], "x", color="navy", ms=7, label=r"Özmen $2s$")
            ax.plot(OZMEN[V0]["Rc"], OZMEN[V0]["2p"], "+", color="crimson", ms=8, label=r"Özmen $2p$")
        ax.set_ylabel(r"$E_{av}$ (Hartree)"); ax.set_title(rf"$V_0 = {V0:g}$")
        axis_rc(ax); ax.legend()
    ax = axs[-1]
    for (D, V0), st in zip(sets, ("d-", "o--")):
        ax.plot(D["Rc"], D["E_2p"] - D["E_2s"], st, ms=4, label=rf"$V_0 = {V0:g}$")
    ax.axhline(0, color="k", lw=0.7)
    ax.axhline(0.067655, color="gray", lw=0.8, ls=":", label="Átomo libre")
    ax.set_ylabel(r"$\Delta E = E(1s^2 2p) - E(1s^2 2s)$ (Hartree)")
    ax.set_ylim(-0.3, 0.15)
    axis_rc(ax); ax.legend()
    save(fig, a.dir, "fig_1_energia_total_2s_2p.png")

    # ---- Fig 4.9: entropia de Shannon S_r de los orbitales ----
    fig, axs = plt.subplots(1, ncol, figsize=(5.5 * ncol, 4.5), squeeze=False)
    for ax, (D, V0) in zip(axs[0], sets):
        ax.plot(D["Rc"], D["S_val_2s"], "o-", color="darkgreen", ms=4, label=r"$2s$")
        ax.plot(D["Rc"], D["S_val_2p"], "s-", color="purple", ms=4, label=r"$2p$ (promedio en $m$)")
        ax.plot(D["Rc"], D["S_val_2p_pz"], "s--", color="purple", ms=3, alpha=0.5, label=r"$2p_z$")
        ax.plot(D["Rc"], D["S_1s_2s"], "^-", color="gray", ms=4, label=r"$1s$ (en $1s^2 2s$)")
        ax.set_ylabel(r"Entropía de Shannon $S_r$"); ax.set_title(rf"$V_0 = {V0:g}$")
        axis_rc(ax); ax.legend()
    save(fig, a.dir, "fig_2_entropia_shannon.png")

    # ---- Fig 4.10: entropia en momentos y desigualdad BBM; Fig 4.2: error de la base ----
    malla = cargar_malla(a.malla)
    fig_entropia_momentos(sets, malla, a.dir)
    fig_error_base(sets, malla, a.dir)

    # ---- Fig 4.8: mapas 2D de la densidad del electron de valencia ----
    for Rsel in (10.0, 1.0):
        i = int(np.argmin(abs(Rc - Rsel)))
        R0 = Rc[i]
        zs = np.array([float(v) for v in A["zeta_2s"][i].split()])
        zp = np.array([float(v) for v in A["zeta_2p"][i].split()])
        B2s = Basis(zs, zp[N_PER_BLOCK:])
        B2p = Basis(zp[:N_PER_BLOCK], zp[N_PER_BLOCK:])
        r2s = scf_eav(B2s, R0, V0a, "2s")
        r2p = scf_eav(B2p, R0, V0a, "2p")
        L = 1.6 * R0 if R0 > 3 else 4.0
        x = np.linspace(-L, L, 401)
        X, Zg = np.meshgrid(x, x)
        Rr = np.sqrt(X ** 2 + Zg ** 2)
        cos = np.where(Rr > 0, Zg / np.where(Rr > 0, Rr, 1), 0.0)
        R2s = (B2s.radial(Rr.ravel()) @ r2s["c2"]).reshape(Rr.shape)
        R2p = (B2p.radial(Rr.ravel()) @ r2p["c2"]).reshape(Rr.shape)
        rho_s = R2s ** 2 / (4 * np.pi)
        rho_p = R2p ** 2 * 3 * cos ** 2 / (4 * np.pi)
        fig, axs = plt.subplots(1, 2, figsize=(10, 4.6))
        vmax = max(rho_s.max(), rho_p.max())
        for ax, rho, t in ((axs[0], rho_s, r"$|\psi_{2s}|^2$ en $1s^2 2s$"),
                           (axs[1], rho_p, r"$|\psi_{2p_z}|^2$ en $1s^2 2p$")):
            im = ax.pcolormesh(X, Zg, np.maximum(rho, vmax * 1e-6), shading="auto",
                               norm=LogNorm(vmin=vmax * 1e-6, vmax=vmax), cmap="inferno")
            th = np.linspace(0, 2 * np.pi, 300)
            ax.plot(R0 * np.cos(th), R0 * np.sin(th), "w--", lw=1.2)
            ax.set_aspect("equal"); ax.set_title(t)
            ax.set_xlabel(r"$x$ (u.a.)"); ax.set_ylabel(r"$z$ (u.a.)")
        fig.colorbar(im, ax=axs, shrink=0.85, label=r"densidad (u.a.$^{-3}$)")
        fig.suptitle(rf"$R_c = {R0:g}$ u.a., $V_0 = {V0a:g}$")
        fig.savefig(os.path.join(a.dir, f"nubes_electronicas_Rc_{Rsel:.1f}.png"), dpi=300,
                    bbox_inches="tight")
        plt.close(fig)
    print("Figuras en", a.dir)


if __name__ == "__main__":
    main()
