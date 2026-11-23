"""Mejor energía Hartree-Fock alcanzable con la base de la tesis
(cinco STO s con n = 1, y cinco STO p con n = 2), evaluando el funcional en la
malla numérica (las integrales son "exactas" para la base dada).

Para exponentes fijos se resuelve Roothaan en el subespacio de la base usando
las mismas matrices de Fock que el código numérico; los exponentes se optimizan
con las restricciones de la tesis: zeta_i - zeta_{i+1} >= 0.4 y lambda_min(S) >= 1e-7.

Uso:  python base_sto.py <2s|2p> <V0> <Rc> <h>      (Rc <= 0: átomo libre)
      python base_sto.py 2s 5 2 0.02

Es un cálculo lento (miles de ciclos SCF en la malla): cerca de una hora por caso (61 min para 2s 5 2 0.02, E = -5.92996).
"""
import sys
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize
from hf_litio_confinado import (Malla, energia, fock, rota, energia_2p, fock_2p)
from scipy.optimize import minimize_scalar


def base(m, zetas, n):
    # P = r^n exp(-zeta r)  ->  y = P / sqrt(r)
    X = np.array([m.r**n * np.exp(-z * m.r) for z in zetas]).T / np.sqrt(m.r)[:, None]
    nrm = np.sqrt(m.h * np.sum(m.Bd[:, None] * X**2, axis=0))
    return X / nrm


def sub_eig(A, S, cb=None, k=0):
    """k-ésimo estado de A c = e S c, opcionalmente S-ortogonal a cb."""
    if cb is not None:
        Scb = S @ cb
        Q = np.eye(len(cb)) - np.outer(cb, Scb) / (cb @ Scb)
        A = Q.T @ A @ Q + 1e6 * np.outer(Scb, Scb) / (cb @ Scb)
    w, v = eigh(A, S)
    return v[:, k]


def hf_2s_en_base(m, zetas, maxit=60):
    X = base(m, zetas, 1)
    Bd = m.Bd
    S = m.h * X.T @ (Bd[:, None] * X)
    if np.linalg.eigvalsh(S / np.sqrt(np.outer(np.diag(S), np.diag(S)))).min() < 1e-7:
        return np.inf
    # arranque: hamiltoniano de un electrón
    H = m.h * X.T @ m.Aloc @ X
    w, v = eigh(H, S)
    c1, c2 = v[:, 0], v[:, 1]
    E_old = np.inf
    for _ in range(maxit):
        y1, y2 = X @ c1, X @ c2
        y1 /= np.sqrt(m.prod(y1, y1)); y2 /= np.sqrt(m.prod(y2, y2))
        A1, A2 = fock(m, y1, y2)
        F1, F2 = m.h * X.T @ A1 @ X, m.h * X.T @ A2 @ X
        c1n = sub_eig(F1, S, c2, 0)
        c2n = sub_eig(F2, S, c1n, 0)
        # mezcla y ortonormalización
        c1 = 0.6 * np.sign(c1n @ S @ c1) * c1n + 0.4 * c1
        c2 = 0.6 * np.sign(c2n @ S @ c2) * c2n + 0.4 * c2
        c1 /= np.sqrt(c1 @ S @ c1)
        c2 = c2 - (c2 @ S @ c1) * c1
        c2 /= np.sqrt(c2 @ S @ c2)
        y1, y2 = X @ c1, X @ c2
        res = minimize_scalar(lambda t: energia(m, *rota(y1, y2, t)),
                              bounds=(-0.3, 0.3), method="bounded")
        ct, st = np.cos(res.x), np.sin(res.x)
        c1, c2 = ct * c1 + st * c2, -st * c1 + ct * c2
        E = res.fun
        if abs(E - E_old) < 1e-10:
            break
        E_old = E
    return E


def hf_2p_en_base(m, zs, zp, maxit=60):
    Xs, Xp = base(m, zs, 1), base(m, zp, 2)
    Ss = m.h * Xs.T @ (m.Bd[:, None] * Xs)
    Sp = m.h * Xp.T @ (m.Bd[:, None] * Xp)
    for S in (Ss, Sp):
        if np.linalg.eigvalsh(S / np.sqrt(np.outer(np.diag(S), np.diag(S)))).min() < 1e-7:
            return np.inf
    c1 = eigh(Xs.T @ m.Aloc @ Xs, Ss)[1][:, 0]
    cp = eigh(Xp.T @ m.Aloc_p @ Xp, Sp)[1][:, 0]
    E_old = np.inf
    for _ in range(maxit):
        y1, yp = Xs @ c1, Xp @ cp
        y1 /= np.sqrt(m.prod(y1, y1)); yp /= np.sqrt(m.prod(yp, yp))
        A1, Ap = fock_2p(m, y1, yp)
        n1 = eigh(Xs.T @ A1 @ Xs, Ss)[1][:, 0]
        npp = eigh(Xp.T @ Ap @ Xp, Sp)[1][:, 0]
        c1 = 0.6 * np.sign(n1 @ Ss @ c1) * n1 + 0.4 * c1
        cp = 0.6 * np.sign(npp @ Sp @ cp) * npp + 0.4 * cp
        c1 /= np.sqrt(c1 @ Ss @ c1); cp /= np.sqrt(cp @ Sp @ cp)
        E = energia_2p(m, Xs @ c1, Xp @ cp)
        if abs(E - E_old) < 1e-10:
            break
        E_old = E
    return E


def zetas_de(p):
    """p = (log z_min, log incrementos): garantiza z_i - z_{i+1} >= 0.4."""
    z = [np.exp(p[0])]
    for q in p[1:]:
        z.append(z[-1] + 0.4 + np.exp(q))
    return np.array(z[::-1])


def optimiza(m, f, npar, semillas, popsize=12, maxiter=60):
    obj = lambda p: f(zetas_de(p))
    best = (np.inf, None)
    for k, s in enumerate(semillas):
        r = minimize(obj, s, method="Nelder-Mead",
                     options=dict(maxiter=700, xatol=1e-3, fatol=1e-8, adaptive=True))
        print(f"  punto de partida {k + 1} de {len(semillas)}: E = {r.fun:.6f}", flush=True)
        if r.fun < best[0]:
            best = (r.fun, r.x)
    r = minimize(obj, best[1], method="Nelder-Mead",
                 options=dict(maxiter=1500, xatol=1e-5, fatol=1e-10, adaptive=True))
    if r.fun < best[0]:
        best = (r.fun, r.x)
    return best[0], zetas_de(best[1])


def semillas_s():
    # progresiones geométricas razonables (en la parametrización de zetas_de)
    out = []
    for zmin, inc in ((0.6, [0.3, 1.0, 2.0, 4.0]), (1.5, [0.5, 1.0, 2.0, 4.0])):
        out.append(np.log([zmin] + inc))
    return out


if __name__ == "__main__":
    conf = sys.argv[1]
    V0 = float(sys.argv[2]); Rc = float(sys.argv[3]); h = float(sys.argv[4])
    Rcarg = np.inf if Rc <= 0 else Rc
    m = Malla(Z=3.0, Rc=Rcarg, V0=V0, h=h)
    if conf == "2s":
        E, z = optimiza(m, lambda z: hf_2s_en_base(m, z), 5, semillas_s())
    else:
        # se optimizan los 10 exponentes
        def f(p):
            return hf_2p_en_base(m, zetas_de(p[:5]), zetas_de(p[5:]))
        obj = f
        best = (np.inf, None)
        for s in semillas_s():
            for sp in (np.log([0.3, 0.3, 0.8, 1.5, 3.0]), np.log([0.6, 0.4, 1.0, 2.0, 4.0])):
                p0 = np.concatenate([s, sp])
                r = minimize(obj, p0, method="Nelder-Mead",
                             options=dict(maxiter=4000, xatol=1e-4, fatol=1e-9, adaptive=True))
                if r.fun < best[0]:
                    best = (r.fun, r.x)
        r = minimize(obj, best[1], method="Nelder-Mead",
                     options=dict(maxiter=6000, xatol=1e-5, fatol=1e-10, adaptive=True))
        if r.fun < best[0]:
            best = (r.fun, r.x)
        E, z = best[0], (zetas_de(best[1][:5]), zetas_de(best[1][5:]))
    print(f"RESULT conf={conf} V0={V0} Rc={Rc} E_base={E:.7f} zetas={z}", flush=True)
