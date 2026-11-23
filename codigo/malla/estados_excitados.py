"""1s^2 nl con valencia de l arbitraria (3d, 4f, ...) en la malla, y 1s^2 3s.

E_av = 2 I_1s + I_nl + F0(1s,1s) + 2 F0(1s,nl) - G^l(1s,nl)/(2l+1)
"""
import sys
import numpy as np
import warnings
from scipy.linalg import eigh
warnings.filterwarnings("ignore")
from hf_litio_confinado import Malla, iteracion_inversa, nodos


def nucleo(m, k):
    r = m.r
    rmx = np.maximum.outer(r, r); rmn = np.minimum.outer(r, r)
    return m.h * np.outer(r**2, r**2) * rmn**k / rmx**(k + 1)


def Xk(m, Sk, k, yb):
    X = np.outer(yb, yb) * Sk
    X[np.diag_indices(m.N)] -= (2 * k + 1) * (m.h**2 / 12.0) * m.r**3 * yb**2
    return X


def energia_l(m, l, Sk, y1, yv):
    h = m.h; w = 1.0 / (2 * l + 1)
    Al = m.Aloc + np.eye(m.N) * 0.5 * l * (l + 1)
    I1 = h * y1 @ m.Aloc @ y1; Iv = h * yv @ Al @ yv
    F11 = h * np.sum(m.Bd * y1**2 * m.J(y1)); F1v = h * np.sum(m.Bd * y1**2 * m.J(yv))
    G = h * y1 @ Xk(m, Sk, l, yv) @ y1
    return 2 * I1 + Iv + F11 + 2 * F1v - w * G


def fock_l(m, l, Sk, y1, yv):
    w = 1.0 / (2 * l + 1); B = m.Bd
    Al = m.Aloc + np.eye(m.N) * 0.5 * l * (l + 1)
    A1 = m.Aloc + np.diag(B * (m.J(y1) + m.J(yv))) - 0.5 * w * Xk(m, Sk, l, yv)
    Av = Al + np.diag(B * 2.0 * m.J(y1)) - w * Xk(m, Sk, l, y1)
    return A1, Av


def guess_l(m, l):
    idx = m.r > 1e-4; Bs = np.diag(m.Bd[idx])
    def est(A):
        _, v = eigh(A[np.ix_(idx, idx)], Bs, subset_by_index=[0, 0])
        y = np.zeros(m.N); y[idx] = v[:, 0]
        return m.normaliza(y if y[idx][0] >= 0 else -y)
    y1 = est(m.Aloc)
    Al = m.Aloc + np.eye(m.N) * 0.5 * l * (l + 1)
    return y1, est(Al + np.diag(m.Bd * 2.0 * m.J(y1)))


def scf_l(m, l, alfa=0.7, tol=1e-11, maxit=300):
    Sk = nucleo(m, l)
    y1, yv = guess_l(m, l)
    E0 = energia_l(m, l, Sk, y1, yv)
    for it in range(1, maxit + 1):
        A1, Av = fock_l(m, l, Sk, y1, yv)
        n1 = iteracion_inversa(m, A1, y1); nv = iteracion_inversa(m, Av, yv)
        y1 = m.normaliza(alfa * n1 + (1 - alfa) * y1)
        yv = m.normaliza(alfa * nv + (1 - alfa) * yv)
        E = energia_l(m, l, Sk, y1, yv)
        if abs(E - E0) < tol and it > 5:
            break
        E0 = E
    A1, Av = fock_l(m, l, Sk, y1, yv)
    ev = m.h * yv @ Av @ yv
    pout = m.h * np.sum((m.Bd * yv**2)[m.r > m.Rc]) if np.isfinite(m.Rc) else 0.0
    return E, ev, pout, it, nodos(yv)


if __name__ == "__main__":
    l = int(sys.argv[1]); V0 = float(sys.argv[2]); Rc = float(sys.argv[3])
    Rc = np.inf if Rc <= 0 else Rc
    m = Malla(Z=3.0, Rc=Rc, V0=V0, h=0.02, rmax=150.0)
    E, ev, pout, it, nd = scf_l(m, l)
    print(f"l={l} V0={V0} Rc={Rc}: E={E:.6f} e_nl={ev:.5f} Pout={pout:.3f} it={it} nodos={nd}", flush=True)
