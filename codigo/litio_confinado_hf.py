"""
Hartree-Fock-Roothaan para el atomo de litio confinado en un pozo esferico finito,
    V_c(r) = V0 * Theta(r - Rc),
con una base global de orbitales tipo Slater (STO) cuyos exponentes se optimizan
para cada radio. Es el programa principal de la tesis

    A. Valenzuela Leyva, "Estudio de la Estructura Electronica y Propiedades del
    Atomo de Litio Confinado mediante el Metodo de Hartree-Fock", Licenciatura en
    Fisica, Universidad de Sonora (2026).

El modelo es el de Ozmen et al., Physica B 656 (2023) 414775. Correspondencia entre
el codigo y la tesis:

  Sec. 2.8 y Ap. C   Energia media de configuracion E_av de 1s^2 nl (nl = 2s, 2p) y
                     operadores de Fock de capa abierta            -> scf_eav
  Sec. 3.2           Base de 10 STO: 5 tipo s (n = 1) y 5 tipo p (n = 2) -> Basis
  Sec. 3.3 y Ap. B   Integrales de un electron: S, T, V_nuc y V_conf (gamma
                     incompleta superior)                          -> one_electron_matrices
  Sec. 3.4 y Ap. B   Integrales radiales de Slater R^k en forma cerrada y pesos
                     angulares con simbolos 3-j (Ap. A)            -> radial_eri, three_j_000
  Sec. 3.5           Ciclo SCF por bloques de simetria; operador de Guest y Saunders
                     para 1s^2 2s; regularizacion S + eta*I        -> scf_eav
  Sec. 3.5 (subsec.) Optimizacion de exponentes: SLSQP, Evolucion Diferencial y
                     arranques en progresion geometrica, con las restricciones
                     zeta_i - zeta_{i+1} >= 0.4 y lambda_min(S) >= 1e-7 -> StateOptimizer
  Sec. 3.6           Barrido descendente en Rc con propagacion en cascada, pulido con
                     los radios vecinos y propiedades: <r>, P_out, presion
                     (Hellmann-Feynman y diferencias finitas)      -> run_sweep, polish, build_row
  Sec. 2.9 y Ap. E   Entropias de Shannon en posiciones (S_r) y en momentos (S_p) de
                     cada orbital y de la densidad total; la transformada de Fourier
                     de los STO es analitica. Desigualdad de Bialynicki-Birula y
                     Mycielski: S_r + S_p >= 3(1 + ln pi)          -> shannon_entropies, check_bbm

Todas las integrales de uno y dos electrones, <r> y P_out son analiticas. Solo las
entropias se integran numericamente (regla del trapecio). Unidades atomicas en todo
el codigo.

Uso (ver el README):
    python litio_confinado_hf.py --V0 5 --salida datos/litio     # barrido completo
    python litio_confinado_hf.py --pulir datos/litio_V0_5.csv    # pulido con radios vecinos
    python litio_confinado_hf.py --recalcular datos/litio_V0_5.csv
    python litio_confinado_hf.py --libre                         # atomo libre (Tabla 4.1)
    python litio_confinado_hf.py --prueba-momentos
"""

import argparse
import csv
import math
import time

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import LinearConstraint, NonlinearConstraint, differential_evolution, minimize
from scipy.special import gammaincc, gammaln, spherical_jn

# Regla del trapecio: np.trapezoid (NumPy >= 2.0) o np.trapz (versiones anteriores)
trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

# =====================================================================
# 0. Parametros globales
# =====================================================================
Z_NUC = 3.0
ETA_TIKHONOV = 1e-8      # Sec. 3.5: S + eta*I solo estabiliza la diagonalizacion
DELTA_MIN = 0.4          # Sec. 3.5: zeta_i - zeta_{i+1} >= 0.4
SCF_TOL = 1e-10          # |Delta E| (Hartree); 1e-6 no basta para los gradientes numericos de SLSQP
SCF_TOL_P = 1e-8         # RMS de la matriz de densidad total P = 2 D_1s + D_nl
SCF_MAXITER = 300
FIELD_DELTA = 1e-3       # campo finito F = -delta, 0, +delta (solo para --polarizabilidad)
N_PER_BLOCK = 5          # 5 s + 5 p = 10 STOs
LOG_SMIN = -7.0          # log10 del menor eigenvalor permitido de S (por bloque, funciones normalizadas)


# =====================================================================
# 1. Simbolo 3-j con proyecciones nulas (Apendice A de la tesis)
# =====================================================================
def three_j_000(l1, l2, l3):
    """(l1 l2 l3; 0 0 0) por la formula de factoriales. Cero si falla
    la condicion del triangulo o la de paridad."""
    J = l1 + l2 + l3
    if J % 2 or l3 < abs(l1 - l2) or l3 > l1 + l2:
        return 0.0
    g = J // 2
    f = math.factorial
    val = math.sqrt(f(2 * g - 2 * l1) * f(2 * g - 2 * l2) * f(2 * g - 2 * l3) / f(2 * g + 1))
    val *= f(g) / (f(g - l1) * f(g - l2) * f(g - l3))
    return (-1) ** g * val


# =====================================================================
# 2. Base STO global
#    chi(r) = N r^(n-1) e^(-zeta r) Y_l0,   n = l + 1,   N = (2 zeta)^(n+1/2)/sqrt((2n)!)
# =====================================================================
class Basis:
    def __init__(self, zeta_s, zeta_p):
        zs = np.asarray(zeta_s, float)
        zp = np.asarray(zeta_p, float)
        self.zeta = np.concatenate([zs, zp])
        self.l = np.array([0] * len(zs) + [1] * len(zp))
        self.n = self.l + 1
        self.idx_s = np.arange(len(zs))
        self.idx_p = np.arange(len(zs), len(zs) + len(zp))
        self.norm = (2 * self.zeta) ** (self.n + 0.5) / np.sqrt(
            [math.factorial(2 * k) for k in self.n])
        self.size = len(self.zeta)

    def radial(self, r):
        """Matriz (len(r), nbas) con la parte radial R_mu(r) de cada STO."""
        r = np.asarray(r)[:, None]
        return self.norm * r ** (self.n - 1) * np.exp(-self.zeta * r)

    def momentum(self, p):
        """Matriz (len(p), nbas) con la parte radial F_mu(p) de cada STO en el espacio
        de momentos (Apendice E de la tesis):
            F(p) = sqrt(2/pi) int_0^inf j_l(p r) R(r) r^2 dr
        Para n = l + 1 la integral es cerrada:
            int_0^inf r^(l+2) e^(-zeta r) j_l(p r) dr = 2^(l+1) (l+1)! zeta p^l / (zeta^2 + p^2)^(l+2)
        La fase (-i)^l de la transformada no cambia la densidad y se omite."""
        assert np.all(self.n == self.l + 1), "formula cerrada valida solo para n = l + 1"
        p = np.asarray(p)[:, None]
        fact = np.array([2.0 ** (l + 1) * math.factorial(l + 1) for l in self.l])
        return (math.sqrt(2.0 / math.pi) * self.norm * fact * self.zeta * p ** self.l
                / (self.zeta ** 2 + p ** 2) ** (self.l + 2))


# =====================================================================
# 3. Integrales de un electron (analiticas)
# =====================================================================
def _gamma_over_pow(s, z):
    """Gamma(s) / z^s  (s puede ser arreglo de enteros positivos)."""
    return np.exp(gammaln(s) - s * np.log(z))


def one_electron_matrices(B, Rc, V0, Z=Z_NUC):
    """S, T, V_nuc, V_conf y el dipolo z, todos con integrales cerradas.
    V_conf usa la gamma incompleta superior (Apendice B)."""
    na, nb = np.meshgrid(B.n, B.n, indexing="ij")
    za, zb = np.meshgrid(B.zeta, B.zeta, indexing="ij")
    la, lb = np.meshgrid(B.l, B.l, indexing="ij")
    NN = np.outer(B.norm, B.norm)
    zab = za + zb
    nab = na + nb
    same_l = (la == lb).astype(float)

    S = NN * _gamma_over_pow(nab + 1, zab) * same_l
    Vn = -Z * NN * _gamma_over_pow(nab, zab) * same_l

    # -1/2 nabla^2 chi_b = -N/2 [ (n(n-1)-l(l+1)) r^(n-3) - 2 zeta n r^(n-2) + zeta^2 r^(n-1) ] e^(-zeta r) Y
    c1 = nb * (nb - 1) - lb * (lb + 1)
    t1 = np.where(c1 != 0, c1 * _gamma_over_pow(np.maximum(nab - 1, 1), zab), 0.0)
    t2 = -2 * zb * nb * _gamma_over_pow(nab, zab)
    t3 = zb ** 2 * _gamma_over_pow(nab + 1, zab)
    T = -0.5 * NN * (t1 + t2 + t3) * same_l
    T = 0.5 * (T + T.T)

    # <a| V0 Theta(r-Rc) |b> = V0 N_a N_b Gamma(n_a+n_b+1, zeta_ab Rc) / zeta_ab^(n_a+n_b+1)
    Vc = V0 * NN * gammaincc(nab + 1, zab * Rc) * _gamma_over_pow(nab + 1, zab) * same_l

    # <a| z |b> = <a| r cos(theta) |b>; angular s-p_z = 1/sqrt(3)
    sp = ((la == 0) & (lb == 1)) | ((la == 1) & (lb == 0))
    Dz = NN * _gamma_over_pow(nab + 2, zab) / math.sqrt(3.0) * sp

    return S, T, Vn, Vc, Dz


# =====================================================================
# 4. Integrales radiales bi-electronicas R^k (analiticas)
#    R^k[ab|cd] = int int chi_a chi_b (1) chi_c chi_d (2) r<^k / r>^(k+1) r1^2 r2^2 dr1 dr2
# =====================================================================
def _J_half(A, al, B, be, k):
    """int_0^inf dr2 r2^(B-k-1) e^(-be r2) int_0^r2 dr1 r1^(A+k) e^(-al r1)
    = m!/al^(m+1) [ Gamma(p)/be^p - sum_{j=0}^m al^j/j! Gamma(p+j)/(al+be)^(p+j) ],
    con m = A + k, p = B - k (enteros, p >= 1)."""
    m = A + k
    p = B - k
    s = al + be
    first = _gamma_over_pow(p, be)
    acc = np.zeros(np.broadcast(A, al, B, be).shape)
    mmax = int(np.max(m))
    for j in range(mmax + 1):
        term = np.exp(j * np.log(al) - gammaln(j + 1) + gammaln(p + j) - (p + j) * np.log(s))
        acc = acc + np.where(j <= m, term, 0.0)
    return np.exp(gammaln(m + 1) - (m + 1) * np.log(al)) * (first - acc)


def radial_eri(B, kmax=2):
    """Tensor Rk[k, a, b, c, d] para k = 0..kmax (sin parte angular)."""
    nb = B.size
    NN = np.outer(B.norm, B.norm)
    A = (B.n[:, None] + B.n[None, :]).ravel()          # potencia de r (incluye r^2)
    ZA = (B.zeta[:, None] + B.zeta[None, :]).ravel()
    W = NN.ravel()
    Ag, Bg = np.meshgrid(A, A, indexing="ij")
    Zg, Yg = np.meshgrid(ZA, ZA, indexing="ij")
    Wg = np.outer(W, W)
    Lpair = (B.l[:, None] + B.l[None, :]).ravel()
    Lmin = np.minimum.outer(Lpair, Lpair)
    out = np.zeros((kmax + 1, nb, nb, nb, nb))
    for k in range(kmax + 1):
        ok = Lmin >= k            # sin esto p = B - k podria ser < 1 en combinaciones no usadas
        Ak = np.where(ok, Ag, k + 2)
        Bk = np.where(ok, Bg, k + 2)
        val = _J_half(Ak, Zg, Bk, Yg, k) + _J_half(Bk, Yg, Ak, Zg, k)
        out[k] = np.where(ok, Wg * val, 0.0).reshape(nb, nb, nb, nb)
    return out


def full_eri(B, Rk):
    """(ab|cd) completo para funciones con m = 0, via coeficientes de Gaunt:
    (ab|cd) = sum_k sqrt((2la+1)(2lb+1)(2lc+1)(2ld+1)) (la k lb;000)^2 (lc k ld;000)^2 R^k."""
    l = B.l
    eri = np.zeros(Rk.shape[1:])
    for k in range(Rk.shape[0]):
        g = np.array([[three_j_000(a, k, b) ** 2 * math.sqrt((2 * a + 1) * (2 * b + 1))
                       for b in l] for a in l])
        eri += np.einsum("ab,cd,abcd->abcd", g, g, Rk[k])
    return eri


# =====================================================================
# 5. Operadores esfericamente promediados (formalismo de energia media)
# =====================================================================
def w_exchange(l_open):
    """Peso del intercambio 1s-nl en E_av: sum_k (0 k l; 000)^2 G^k  ->  solo k = l."""
    return {l_open: three_j_000(0, l_open, l_open) ** 2}


def coulomb(Rk, D):
    """J[a,b] = sum_cd D[c,d] R^0[ab|cd]  (parte esferica)."""
    return np.einsum("abcd,cd->ab", Rk[0], D)


def exchange(Rk, D, weights):
    """K~[a,b] = sum_k w_k sum_cd D[c,d] R^k[ac|db]."""
    K = 0.0
    for k, w in weights.items():
        K = K + w * np.einsum("acdb,cd->ab", Rk[k], D)
    return K


# =====================================================================
# 6. SCF ROHF / energia media de configuracion para 1s^2 nl (nl = 2s o 2p)
# =====================================================================
def _guest_saunders(Fa, Fb, C, S, i_c, i_o):
    """Fock efectivo ROHF (Guest-Saunders) para 1 capa cerrada + 1 abierta
    dentro del mismo bloque de simetria."""
    Fa_mo = C.T @ Fa @ C
    Fb_mo = C.T @ Fb @ C
    R = 0.5 * (Fa_mo + Fb_mo)
    nv = [i for i in range(C.shape[1]) if i not in (i_c, i_o)]
    R[i_c, i_o] = R[i_o, i_c] = Fb_mo[i_c, i_o]
    for v in nv:
        R[i_o, v] = R[v, i_o] = Fa_mo[i_o, v]
    SC = S @ C
    return SC @ R @ SC.T


def _renorm(C, S):
    """Normaliza columnas con la S verdadera. La regularizacion de Tikhonov solo
    estabiliza el eigensolver; si se normalizara con S + eta*I, los vectores casi
    en el nucleo de S tendrian norma real ~0 y la energia colapsaria (no variacional)."""
    C = np.atleast_2d(C.T).T
    return C / np.sqrt(np.einsum("ai,ab,bi->i", C, S, C))


def scf_eav(B, Rc, V0, config, Rk=None, tol=SCF_TOL):
    """Resuelve el campo autoconsistente para 1s^2 2s (config='2s') o 1s^2 2p (config='2p').
    Regresa un diccionario con E_av, orbitales, energias orbitales, etc."""
    S, T, Vn, Vc, _ = one_electron_matrices(B, Rc, V0)
    H = T + Vn + Vc
    if Rk is None:
        Rk = radial_eri(B, kmax=2 if config == "2p" else 0)
    Sreg = S + ETA_TIKHONOV * np.eye(B.size)
    l_open = 0 if config == "2s" else 1
    w = w_exchange(l_open)
    ids, idp = B.idx_s, B.idx_p
    Ss = Sreg[np.ix_(ids, ids)]
    Hs = H[np.ix_(ids, ids)]

    # Arranque sin repulsion entre electrones: diagonalizamos H^core por bloques
    S_s = S[np.ix_(ids, ids)]
    S_p = S[np.ix_(idp, idp)]
    _, Cs = eigh(Hs, Ss)
    Cs[:, :2] = _renorm(Cs[:, :2], S_s)
    c1 = np.zeros(B.size); c1[ids] = Cs[:, 0]
    c2 = np.zeros(B.size)
    if config == "2s":
        c2[ids] = Cs[:, 1]
    else:
        Sp, Hp = Sreg[np.ix_(idp, idp)], H[np.ix_(idp, idp)]
        _, Cp = eigh(Hp, Sp)
        c2[idp] = _renorm(Cp[:, :1], S_p)[:, 0]

    E_old = np.inf
    P_old = np.zeros((B.size, B.size))
    converged = False
    for it in range(SCF_MAXITER):
        D1, D2 = np.outer(c1, c1), np.outer(c2, c2)
        P = 2 * D1 + D2
        dP_rms = np.sqrt(np.mean((P - P_old) ** 2))
        P_old = P
        J1, J2 = coulomb(Rk, D1), coulomb(Rk, D2)
        K1, K2 = exchange(Rk, D1, w), exchange(Rk, D2, w)

        # Energia media de configuracion (ec. 3 de Ozmen):
        # E_av = 2 I_1s + I_nl + F0(1s,1s) + 2 F0(1s,nl) - sum_k (0 k l;000)^2 G^k(1s,nl)
        E = (2 * np.sum(H * D1) + np.sum(H * D2) + np.sum(J1 * D1)
             + 2 * np.sum(J1 * D2) - np.sum(K1 * D2))

        # Criterios de convergencia (Sec. 3.5): energia y RMS de la matriz de densidad
        if abs(E - E_old) < tol and dP_rms < SCF_TOL_P and it > 2:
            converged = True
            break
        E_old = E

        # Operadores de Fock que anulan la variacion de E_av
        F1 = H + J1 + J2 - 0.5 * K2          # orbital 1s (2 electrones)
        F2 = H + 2 * J1 - K1                  # orbital nl (1 electron)

        if config == "2s":
            # Ambos orbitales en el bloque s: ROHF de Guest-Saunders con
            # F_alpha = h + 2J1 + J2 - K1 - K2,  F_beta = h + 2J1 + J2 - K1
            Fa = (H + 2 * J1 + J2 - K1 - K2)[np.ix_(ids, ids)]
            Fb = (H + 2 * J1 + J2 - K1)[np.ix_(ids, ids)]
            Feff = _guest_saunders(Fa, Fb, Cs, Ss, 0, 1)
            _, Cs = eigh(Feff, Ss)
            Cs[:, :2] = _renorm(Cs[:, :2], S_s)
            c1 = np.zeros(B.size); c1[ids] = Cs[:, 0]
            c2 = np.zeros(B.size); c2[ids] = Cs[:, 1]
        else:
            # Bloques independientes: 1s en el bloque s, 2p en el bloque p
            _, Cs = eigh(F1[np.ix_(ids, ids)], Ss)
            _, Cp = eigh(F2[np.ix_(idp, idp)], Sreg[np.ix_(idp, idp)])
            c1 = np.zeros(B.size); c1[ids] = _renorm(Cs[:, :1], S_s)[:, 0]
            c2 = np.zeros(B.size); c2[idp] = _renorm(Cp[:, :1], S_p)[:, 0]

    eps1 = c1 @ F1 @ c1        # ec. 4 de Ozmen
    eps2 = c2 @ F2 @ c2
    return dict(E=E, c1=c1, c2=c2, eps1=eps1, eps2=eps2, converged=converged,
                iters=it, Rk=Rk, S=S, H=H, Vc=Vc, T=T, Vn=Vn)


# =====================================================================
# 7. Polarizabilidad por campo finito (opcional; no se usa en la tesis)
# =====================================================================
def scf_field(B, Rc, V0, F, guess, eri, tol=1e-12):
    """SCF ROHF para 1s^2 2s con H' = F z. El campo mezcla s y p_z, por lo
    que aqui se usa la base completa (sin bloques) y los ERI con m = 0.
    El orbital abierto se elige por maxima superposicion con el 2s de
    referencia (MOM), para no colapsar al 2p cuando este queda por debajo."""
    S, T, Vn, Vc, Dz = one_electron_matrices(B, Rc, V0)
    H = T + Vn + Vc + F * Dz
    Sreg = S + ETA_TIKHONOV * np.eye(B.size)
    c1, c2 = guess["c1"].copy(), guess["c2"].copy()
    c2_ref = guess["c2"]
    _, Call = eigh(H, Sreg)
    # base de orbitales completa con c1 y c2 al frente: el resto se proyecta
    # fuera de {c1, c2} y se ortonormaliza (ortogonalizacion canonica)
    Occ = np.column_stack([c1, c2])
    M = Call - Occ @ (Occ.T @ Sreg @ Call)
    w, U = np.linalg.eigh(M.T @ Sreg @ M)
    keep = np.argsort(w)[::-1][:B.size - 2]
    V = M @ U[:, keep] / np.sqrt(w[keep])
    C = np.column_stack([c1, c2, V])
    E_old = np.inf
    for it in range(SCF_MAXITER):
        c1, c2 = C[:, 0], C[:, 1]
        D1, D2 = np.outer(c1, c1), np.outer(c2, c2)
        Dt = 2 * D1 + D2
        Jt = np.einsum("abcd,cd->ab", eri, Dt)
        K1 = np.einsum("acbd,cd->ab", eri, D1)
        K2 = np.einsum("acbd,cd->ab", eri, D2)
        Fa = H + Jt - K1 - K2
        Fb = H + Jt - K1
        E = 0.5 * (np.sum((D1 + D2) * (H + Fa)) + np.sum(D1 * (H + Fb)))
        if abs(E - E_old) < tol and it > 2:
            break
        E_old = E
        Feff = _guest_saunders(Fa, Fb, C, Sreg, 0, 1)
        _, Cn = eigh(Feff, Sreg)
        ov = np.abs(Cn.T @ Sreg @ c2_ref)
        ov[0] = -1
        i_o = int(np.argmax(ov))
        order = [0, i_o] + [i for i in range(B.size) if i not in (0, i_o)]
        C = Cn[:, order]
        C[:, :2] = _renorm(C[:, :2], S)
    return E


def polarizability(B, Rc, V0, ref, delta=FIELD_DELTA):
    Rk = radial_eri(B, kmax=2)
    eri = full_eri(B, Rk)
    Ep = scf_field(B, Rc, V0, +delta, ref, eri)
    E0 = scf_field(B, Rc, V0, 0.0, ref, eri)
    Em = scf_field(B, Rc, V0, -delta, ref, eri)
    return -(Ep - 2 * E0 + Em) / delta ** 2, E0


# =====================================================================
# 8. Optimizacion variacional de los exponentes (Sec. 3.5 de la tesis)
# =====================================================================
def _sep_constraint(nvar, blocks):
    """Matriz A tal que A @ zeta >= DELTA_MIN impone zeta_i - zeta_{i+1} >= 0.4
    dentro de cada bloque (exponentes ordenados de mayor a menor)."""
    rows = []
    for start, size in blocks:
        for i in range(size - 1):
            r = np.zeros(nvar)
            r[start + i], r[start + i + 1] = 1.0, -1.0
            rows.append(r)
    return np.array(rows)


class StateOptimizer:
    """Optimiza los exponentes que afectan la energia de cada configuracion:
       1s^2 2s -> solo los 5 exponentes s (las p no entran en E_av),
       1s^2 2p -> los 5 s y los 5 p (10 parametros)."""

    def __init__(self, config, V0, bounds_s=(0.05, 40.0), bounds_p=(0.05, 30.0)):
        self.config = config
        self.V0 = V0
        self.nvar = N_PER_BLOCK if config == "2s" else 2 * N_PER_BLOCK
        self.bounds = [bounds_s] * N_PER_BLOCK
        blocks = [(0, N_PER_BLOCK)]
        if config == "2p":
            self.bounds += [bounds_p] * N_PER_BLOCK
            blocks.append((N_PER_BLOCK, N_PER_BLOCK))
        self.A = _sep_constraint(self.nvar, blocks)
        self.zeta_p_fixed = np.array([3.0, 2.0, 1.2, 0.7, 0.3])

    def log_smin(self, x):
        """log10 del menor eigenvalor de S en cada bloque. La separacion absoluta
        zeta_i - zeta_{i+1} >= 0.4 no basta: con exponentes grandes las funciones se
        vuelven casi paralelas (p. ej. 13.2 y 12.8). Esta restriccion mantiene
        cond(S) <= 1e7 y evita el colapso numerico del SCF."""
        x = np.asarray(x)
        B = self.basis(x)
        S = one_electron_matrices(B, 1.0, 0.0)[0]
        out = [np.log10(max(np.linalg.eigvalsh(S[np.ix_(B.idx_s, B.idx_s)])[0], 1e-300))]
        if self.config == "2p":
            out.append(np.log10(max(np.linalg.eigvalsh(S[np.ix_(B.idx_p, B.idx_p)])[0], 1e-300)))
        return np.array(out)

    def basis(self, x):
        if self.config == "2s":
            return Basis(x[:N_PER_BLOCK], self.zeta_p_fixed)
        return Basis(x[:N_PER_BLOCK], x[N_PER_BLOCK:])

    def energy(self, x, Rc):
        x = np.asarray(x)
        # Las restricciones de separacion las imponen DE (LinearConstraint) y SLSQP;
        # no se penalizan aqui para no romper los gradientes por diferencias finitas.
        if np.any(x <= 0) or np.any(self.log_smin(x) < LOG_SMIN - 2.0):
            return 1e3
        B = self.basis(x)
        try:
            res = scf_eav(B, Rc, self.V0, self.config)
        except (np.linalg.LinAlgError, ValueError):
            return 1e3
        E = res["E"]
        return E if np.isfinite(E) else 1e3

    def global_search(self, Rc, seed, maxiter=60, popsize=12, rng=0):
        lc = LinearConstraint(self.A, DELTA_MIN, np.inf)
        nc = NonlinearConstraint(self.log_smin, LOG_SMIN, np.inf)
        de = differential_evolution(self.energy, self.bounds, args=(Rc,),
                                    constraints=(lc, nc), x0=seed, maxiter=maxiter,
                                    popsize=popsize, tol=1e-8, seed=rng,
                                    polish=False, init="sobol")
        return de.x

    def local_refine(self, Rc, x0):
        cons = [{"type": "ineq", "fun": lambda x: self.A @ x - DELTA_MIN,
                 "jac": lambda x: self.A},
                {"type": "ineq", "fun": lambda x: self.log_smin(x) - LOG_SMIN}]
        res = minimize(self.energy, x0, args=(Rc,), method="SLSQP",
                       bounds=self.bounds, constraints=cons,
                       options={"ftol": 1e-10, "maxiter": 300, "eps": 1e-6})
        ok = (np.all(self.A @ res.x >= DELTA_MIN - 1e-6)
              and np.all(self.log_smin(res.x) >= LOG_SMIN - 1e-3))
        x = res.x if (ok and res.fun <= self.energy(x0, Rc)) else x0
        return x, self.energy(x, Rc)


# =====================================================================
# 9. Propiedades de los orbitales (Sec. 3.6): P_in, P_out, <r> (analiticas) y S_r
# =====================================================================
R_GRID = np.concatenate([np.linspace(0.0, 1.0, 4001)[:-1], np.linspace(1.0, 60.0, 20001)])


def r_matrix(B):
    """<a| r |b> = N_a N_b (n_a + n_b + 2)! / (zeta_a + zeta_b)^(n_a + n_b + 3) (mismo l)."""
    nab = B.n[:, None] + B.n[None, :]
    zab = B.zeta[:, None] + B.zeta[None, :]
    return np.outer(B.norm, B.norm) * _gamma_over_pow(nab + 2, zab) * (B.l[:, None] == B.l[None, :])


def orbital_observables(B, c, Rc, r=R_GRID):
    """P_in, P_out, <r>, entropia de Shannon radial y R(Rc)^2 de un orbital.
    P_out = <Theta(r - Rc)> y <r> se calculan con las integrales analiticas de la base
    (P_out es el valor esperado de V_conf con V0 = 1). Solo la entropia,
    S_radial = -int R^2 ln(R^2) r^2 dr, se integra con la regla del trapecio; la entropia
    total es S_radial + S_angular, con S_angular = ln(4 pi) para una densidad esferica
    (orbital s, o p promediado sobre m como en la energia media de configuracion) y
    ln(4 pi/3) + 2/3 para p_z."""
    S, _, _, Theta, _ = one_electron_matrices(B, Rc, 1.0)
    norm = c @ S @ c
    P_out = (c @ Theta @ c) / norm
    r_mean = (c @ r_matrix(B) @ c) / norm
    R2 = (B.radial(r) @ c) ** 2
    R2 = R2 / trapz(R2 * r ** 2, r)
    logR2 = np.log(np.where(R2 > 1e-300, R2, 1e-300))
    S_rad = -trapz(R2 * logR2 * r ** 2, r)
    Rw = (B.radial(np.array([Rc])) @ c)[0]
    return dict(P_in=1.0 - P_out, P_out=P_out, r_mean=r_mean,
                S_rad=S_rad, R2_wall=Rw ** 2 / norm)


S_ANG_SPH = math.log(4 * math.pi)
S_ANG_PZ = math.log(4 * math.pi / 3) + 2.0 / 3.0

# ---------------------------------------------------------------------
# 9b. Entropias de Shannon en posiciones y en momentos (Sec. 2.9 y Apendice E)
#     Para una densidad esferica rho = f(q)/(4 pi), con int f q^2 dq = 1 (q = r o p):
#         S = -int f ln f q^2 dq + ln(4 pi)
#     La malla en p es logaritmica: la densidad de momentos de un orbital s
#     decae como p^-8 y casi toda la entropia esta entre p ~ 1e-2 y p ~ 1e2.
# ---------------------------------------------------------------------
P_GRID = np.exp(np.linspace(math.log(1e-4), math.log(2e3), 8001))
BBM_COTA = 3.0 * (1.0 + math.log(math.pi))          # 6.4342...


def radial_entropy(f, q):
    """-int f ln f q^2 dq para una densidad radial f(q) (se renormaliza con el trapecio)."""
    f = f / trapz(f * q ** 2, q)
    logf = np.log(np.where(f > 1e-300, f, 1e-300))
    return -trapz(f * logf * q ** 2, q)


def shannon_entropies(B, c1, cv, r=R_GRID, p=P_GRID):
    """Entropias de Shannon de la configuracion 1s^2 nl con los orbitales c1 (1s) y cv (nl).
    Devuelve S_r y S_p (con la parte angular esferica ln 4 pi) del 1s, del orbital de
    valencia (promediado en m si es p) y de la densidad total normalizada a 1,
        sigma(r) = [2 |R_1s|^2 + |R_nl|^2] / (3 * 4 pi),   Pi(p) igual con F en lugar de R,
    y la norma de la densidad de momentos de cada orbital como control (debe dar 1)."""
    R1, Rv = B.radial(r) @ c1, B.radial(r) @ cv
    F1, Fv = B.momentum(p) @ c1, B.momentum(p) @ cv
    # normalizamos en r con la misma regla que en orbital_observables
    n1, nv = trapz(R1 ** 2 * r ** 2, r), trapz(Rv ** 2 * r ** 2, r)
    R1, Rv, F1, Fv = R1 / math.sqrt(n1), Rv / math.sqrt(nv), F1 / math.sqrt(n1), Fv / math.sqrt(nv)
    out = {}
    for tag, fr, fp in (("1s", R1 ** 2, F1 ** 2), ("val", Rv ** 2, Fv ** 2),
                        ("tot", (2 * R1 ** 2 + Rv ** 2) / 3.0, (2 * F1 ** 2 + Fv ** 2) / 3.0)):
        out[f"Sr_{tag}"] = radial_entropy(fr, r) + S_ANG_SPH
        out[f"Sp_{tag}"] = radial_entropy(fp, p) + S_ANG_SPH
        out[f"normp_{tag}"] = trapz(fp * p ** 2, p)
    return out


def check_bbm(rows):
    """Resumen de la desigualdad de Bialynicki-Birula y Mycielski en un barrido:
    para cada densidad (1s, valencia, 2p_z y total) de cada configuracion, el minimo de
    S_r + S_p sobre todos los radios y su distancia a la cota 3(1 + ln pi)."""
    sumas = {
        "1s (1s2 2s)": ("S_1s_2s", "Sp_1s_2s"), "2s": ("S_val_2s", "Sp_val_2s"),
        "total 1s2 2s": ("Sr_tot_2s", "Sp_tot_2s"),
        "1s (1s2 2p)": ("S_1s_2p", "Sp_1s_2p"), "2p (prom. m)": ("S_val_2p", "Sp_val_2p"),
        "2p_z": ("S_val_2p_pz", "Sp_val_2p_pz"), "total 1s2 2p": ("Sr_tot_2p", "Sp_tot_2p"),
    }
    print(f"Desigualdad BBM: S_r + S_p >= 3(1 + ln pi) = {BBM_COTA:.6f}")
    ok = True
    for nombre, (a, b) in sumas.items():
        st = [(r[a] + r[b], r["Rc"]) for r in rows]
        smin, rmin = min(st)
        ok &= smin > BBM_COTA
        print(f"  {nombre:14s}: min S_r + S_p = {smin:.4f} en Rc = {rmin:g}  "
              f"(margen {smin - BBM_COTA:+.4f})")
    print("  Se cumple en todos los casos." if ok else "  *** NO se cumple en algun caso ***")
    return ok


def momentum_selftest():
    """Pruebas del calculo en momentos (Apendice E):
    (1) STO 1s con exponente zeta (hidrogenoide):
          S_r = 3 + ln pi - 3 ln zeta,   S_p = ln(32 pi^2) - 10/3 + 3 ln zeta,
        de modo que S_r + S_p = ln(32 pi^3) - 1/3 = 6.5666 no depende de zeta;
    (2) la formula cerrada de F(p) contra la transformada de Hankel por cuadratura;
    (3) la norma en el espacio de momentos (teorema de Plancherel)."""
    from scipy.integrate import quad
    print(f"Cota BBM 3(1 + ln pi) = {BBM_COTA:.6f}")
    for z in (1.0, 2.7, 6.0):
        B = Basis([z], [])
        c = np.array([1.0])
        Sr = radial_entropy((B.radial(R_GRID) @ c) ** 2, R_GRID) + S_ANG_SPH
        Sp = radial_entropy((B.momentum(P_GRID) @ c) ** 2, P_GRID) + S_ANG_SPH
        Sr_ex = 3 + math.log(math.pi) - 3 * math.log(z)
        Sp_ex = math.log(32 * math.pi ** 2) - 10.0 / 3.0 + 3 * math.log(z)
        print(f"  1s zeta={z:4.1f}: S_r={Sr:.7f} (exacto {Sr_ex:.7f})  S_p={Sp:.7f} "
              f"(exacto {Sp_ex:.7f})  suma={Sr + Sp:.7f} (exacto {Sr_ex + Sp_ex:.7f})")
    B = Basis([1.3, 4.0], [0.9, 2.5])
    for mu in range(B.size):
        l = B.l[mu]
        for pv in (0.3, 1.7, 6.0):
            f = lambda rr: spherical_jn(l, pv * rr) * (B.radial(np.array([rr]))[0, mu]) * rr ** 2
            num = math.sqrt(2 / math.pi) * quad(f, 0, 80, limit=400)[0]
            ana = B.momentum(np.array([pv]))[0, mu]
            assert abs(num - ana) < 1e-9 * max(1, abs(ana)), (mu, pv, num, ana)
        normp = trapz(B.momentum(P_GRID)[:, mu] ** 2 * P_GRID ** 2, P_GRID)
        print(f"  STO n={B.n[mu]} l={l} zeta={B.zeta[mu]}: F(p) analitica = cuadratura; "
              f"int |F|^2 p^2 dp = {normp:.10f}")


# =====================================================================
# 10. Barrido en Rc (Sec. 3.6: barrido descendente con propagacion en cascada)
# =====================================================================
def _even_tempered_seeds(nvar):
    """Semillas 'even-tempered' zeta_i = top / ratio^i (con separacion >= 0.4)."""
    seeds = []
    for top in (3.0, 6.0, 10.0):
        for ratio in (1.5, 2.2):
            x = top / ratio ** np.arange(N_PER_BLOCK)
            x = np.maximum(x, DELTA_MIN * np.arange(N_PER_BLOCK - 1, -1, -1) + 0.15)
            seeds.append(x if nvar == N_PER_BLOCK else np.concatenate([x, 0.6 * x]))
    return seeds


def _best_of(opt, Rc, seed, de_maxiter):
    """Mejor resultado entre: (i) cascada + SLSQP, (ii) DE + SLSQP y (iii) para el
    1s^2 2s, multi-arranque SLSQP desde semillas even-tempered. El paisaje E(zeta)
    es multimodal bajo confinamiento y ninguna estrategia sola es confiable."""
    xa, Ea = opt.local_refine(Rc, seed)
    if de_maxiter <= 0:
        return xa, Ea
    if opt.config == "2s":
        for x0 in _even_tempered_seeds(opt.nvar):
            xm, Em = opt.local_refine(Rc, x0)
            if Em < Ea:
                xa, Ea = xm, Em
    # El 1s^2 2s (5 parametros, solo k = 0) es barato: busqueda global mas amplia.
    # Su paisaje de energia es el mas multimodal bajo confinamiento fuerte.
    if opt.config == "2s":
        xg = opt.global_search(Rc, xa, maxiter=4 * de_maxiter, popsize=25)
    else:
        xg = opt.global_search(Rc, xa, maxiter=de_maxiter)
    xb, Eb = opt.local_refine(Rc, xg)
    return (xa, Ea) if Ea <= Eb else (xb, Eb)


def build_row(V0, Rc, x2s, x2p, opt2s, opt2p, do_polar=False):
    r2s = scf_eav(opt2s.basis(x2s), Rc, V0, "2s")
    B2p = opt2p.basis(x2p)
    r2p = scf_eav(B2p, Rc, V0, "2p")

    row = dict(Rc=Rc, V0=V0, E_2s=r2s["E"], E_2p=r2p["E"],
               eps1s_2s=r2s["eps1"], eps2s=r2s["eps2"],
               eps1s_2p=r2p["eps1"], eps2p=r2p["eps2"],
               conv_2s=r2s["converged"], conv_2p=r2p["converged"])
    for tag, res, B in (("2s", r2s, opt2s.basis(x2s)), ("2p", r2p, B2p)):
        o1 = orbital_observables(B, res["c1"], Rc)
        o2 = orbital_observables(B, res["c2"], Rc)
        row[f"Pin_1s_{tag}"], row[f"Pout_1s_{tag}"], row[f"r_1s_{tag}"] = o1["P_in"], o1["P_out"], o1["r_mean"]
        row[f"Pin_val_{tag}"], row[f"Pout_val_{tag}"], row[f"r_val_{tag}"] = o2["P_in"], o2["P_out"], o2["r_mean"]
        # Entropia de Shannon del orbital de valencia (densidad esferica) y del 1s
        row[f"S_val_{tag}"] = o2["S_rad"] + S_ANG_SPH
        row[f"S_1s_{tag}"] = o1["S_rad"] + S_ANG_SPH
        if tag == "2p":
            row["S_val_2p_pz"] = o2["S_rad"] + S_ANG_PZ
        # Entropias en el espacio de momentos y de la densidad total (Apendice E).
        # La parte angular en momentos es la misma que en posiciones.
        sh = shannon_entropies(B, res["c1"], res["c2"])
        row[f"Sp_1s_{tag}"] = sh["Sp_1s"]
        row[f"Sp_val_{tag}"] = sh["Sp_val"]
        if tag == "2p":
            row["Sp_val_2p_pz"] = sh["Sp_val"] - S_ANG_SPH + S_ANG_PZ
        row[f"Sr_tot_{tag}"] = sh["Sr_tot"]
        row[f"Sp_tot_{tag}"] = sh["Sp_tot"]
        # Presion por Hellmann-Feynman con la densidad TOTAL en la pared:
        # P = V0 rho(Rc) = (V0 / 4 pi) [ 2 R_1s(Rc)^2 + R_nl(Rc)^2 ]
        row[f"P_HF_{tag}"] = V0 / (4 * math.pi) * (2 * o1["R2_wall"] + o2["R2_wall"])
        row[f"P_HF_val_{tag}"] = V0 / (4 * math.pi) * o2["R2_wall"]
        row[f"Pout_tot_{tag}"] = (2 * o1["P_out"] + o2["P_out"]) / 3.0

    if do_polar:
        # Base de la polarizabilidad: exponentes s del 1s^2 2s + exponentes p del 1s^2 2p
        Bpol = Basis(x2s, x2p[N_PER_BLOCK:])
        ref = scf_eav(Bpol, Rc, V0, "2s")
        row["alpha_zz"], _ = polarizability(Bpol, Rc, V0, ref)
    row["zeta_2s"] = " ".join(f"{z:.5f}" for z in x2s)
    row["zeta_2p"] = " ".join(f"{z:.5f}" for z in x2p)
    return row


def run_sweep(V0, Rc_list, do_polar=False, de_maxiter=40, verbose=True):
    opt2s = StateOptimizer("2s", V0)
    opt2p = StateOptimizer("2p", V0)
    seed2s = np.array([7.0, 4.2, 2.6, 1.0, 0.5])
    seed2p = np.concatenate([seed2s, [3.4, 1.9, 1.1, 0.6, 0.15]])
    rows = []
    for Rc in Rc_list:
        t0 = time.time()
        # Evolucion Diferencial (con la semilla en cascada dentro de la
        # poblacion inicial) + refinamiento SLSQP. Se conserva el mejor de
        # "cascada + SLSQP" y "DE + SLSQP": la cascada sola puede quedar atrapada
        # en minimos locales a confinamiento fuerte.
        x2s, E2s = _best_of(opt2s, Rc, seed2s, de_maxiter)
        x2p, E2p = _best_of(opt2p, Rc, seed2p, de_maxiter)
        seed2s, seed2p = x2s, x2p

        row = build_row(V0, Rc, x2s, x2p, opt2s, opt2p, do_polar)
        r2s = dict(E=row["E_2s"], eps1=row["eps1s_2s"], eps2=row["eps2s"])
        r2p = dict(E=row["E_2p"], eps2=row["eps2p"])
        rows.append(row)
        if verbose:
            msg = (f"Rc={Rc:6.3f}  E(1s2 2s)={r2s['E']:.6f}  E(1s2 2p)={r2p['E']:.6f}  "
                   f"e1s={r2s['eps1']:.4f} e2s={r2s['eps2']:.4f} e2p={r2p['eps2']:.4f}")
            if do_polar:
                msg += f"  a_zz={row['alpha_zz']:.3f}"
            print(msg + f"  ({time.time() - t0:.1f} s)", flush=True)
    return rows


def free_atom(de_maxiter=40, verbose=True):
    """Atomo de litio libre (V0 = 0), Tabla 4.1 de la tesis. Para cada configuracion se
    optimizan los exponentes desde varias semillas (SLSQP), se refina con Evolucion
    Diferencial + SLSQP y se conserva la menor energia. Devuelve un diccionario por
    configuracion con E_av, energias orbitales, <r> y los exponentes optimos."""
    Rc = 100.0                       # con V0 = 0 el radio no interviene
    seed_s = np.array([7.0, 4.2, 2.6, 1.0, 0.5])
    seed_p = np.array([3.4, 1.9, 1.1, 0.6, 0.15])
    out = {}
    for config in ("2s", "2p"):
        opt = StateOptimizer(config, 0.0)
        seeds = [seed_s if config == "2s" else np.concatenate([seed_s, seed_p])]
        seeds += _even_tempered_seeds(opt.nvar)
        x, E = None, np.inf
        for x0 in seeds:
            xm, Em = opt.local_refine(Rc, x0)
            if Em < E:
                x, E = xm, Em
        if de_maxiter > 0:
            xg = opt.global_search(Rc, x, maxiter=de_maxiter)
            xm, Em = opt.local_refine(Rc, xg)
            if Em < E:
                x, E = xm, Em
        B = opt.basis(x)
        res = scf_eav(B, Rc, 0.0, config)
        o1 = orbital_observables(B, res["c1"], Rc)
        o2 = orbital_observables(B, res["c2"], Rc)
        out[config] = dict(E=res["E"], eps1s=res["eps1"], eps_val=res["eps2"],
                           r_1s=o1["r_mean"], r_val=o2["r_mean"], zeta=x,
                           log_smin=opt.log_smin(x))
        if verbose:
            print(f"1s^2 {config}:  E_av = {res['E']:.6f}   eps_1s = {res['eps1']:.6f}   "
                  f"eps_{config} = {res['eps2']:.6f}   <r>_1s = {o1['r_mean']:.4f}   "
                  f"<r>_{config} = {o2['r_mean']:.4f}")
            print("    exponentes:", " ".join(f"{z:.5f}" for z in x),
                  "   log10 lambda_min(S):", " ".join(f"{v:.2f}" for v in out[config]["log_smin"]))
    if verbose:
        dE = out["2p"]["E"] - out["2s"]["E"]
        print(f"Energia de excitacion 2s -> 2p: {dE:.6f} Hartree = {dE * 27.211386:.4f} eV")
    return out


def add_fd_pressure(rows):
    """Presion por diferencias finitas, P = -(1/4 pi Rc^2) dE/dRc, como comprobacion
    independiente de Hellmann-Feynman (np.gradient: centrales de 2o orden en malla no uniforme)."""
    Rc = np.array([r["Rc"] for r in rows])
    order = np.argsort(Rc)
    for tag in ("2s", "2p"):
        E = np.array([rows[i][f"E_{tag}"] for i in order])
        dE = np.gradient(E, Rc[order])
        P = -dE / (4 * np.pi * Rc[order] ** 2)
        for k, i in enumerate(order):
            rows[i][f"P_FD_{tag}"] = P[k]
    return rows


def polish(rows, V0, passes=2, verbose=True, extra=None):
    """Pulido posterior al barrido: para cada R_c se reoptimiza con SLSQP partiendo de
    los exponentes de los radios vecinos (en ambos sentidos) y se conserva el mejor.
    Corrige minimos locales que la cascada descendente no puede evitar."""
    opt2s = StateOptimizer("2s", V0)
    opt2p = StateOptimizer("2p", V0)
    rows = sorted(rows, key=lambda r: -r["Rc"])
    X = {t: [np.array([float(v) for v in r[f"zeta_{t}"].split()]) for r in rows] for t in ("2s", "2p")}
    E = {t: [r[f"E_{t}"] for r in rows] for t in ("2s", "2p")}
    # semillas externas (p. ej. de otra corrida) para los radios que coincidan
    for er in (extra or []):
        for i, r in enumerate(rows):
            if abs(er["Rc"] - r["Rc"]) < 1e-9:
                for t, opt in (("2s", opt2s), ("2p", opt2p)):
                    x, e = opt.local_refine(r["Rc"], np.array([float(v) for v in er[f"zeta_{t}"].split()]))
                    if e < E[t][i] - 1e-9:
                        X[t][i], E[t][i] = x, e
    for _ in range(passes):
        changed = False
        for i, r in enumerate(rows):
            for t, opt in (("2s", opt2s), ("2p", opt2p)):
                for j in (i - 1, i + 1):
                    if 0 <= j < len(rows):
                        x, e = opt.local_refine(r["Rc"], X[t][j])
                        if e < E[t][i] - 1e-9:
                            X[t][i], E[t][i], changed = x, e, True
        if not changed:
            break
    new = []
    for i, r in enumerate(rows):
        nr = build_row(V0, r["Rc"], X["2s"][i], X["2p"][i], opt2s, opt2p)
        if verbose and (abs(nr["E_2s"] - r["E_2s"]) > 1e-7 or abs(nr["E_2p"] - r["E_2p"]) > 1e-7):
            print(f"Rc={r['Rc']:6.3f}  2s: {r['E_2s']:.6f} -> {nr['E_2s']:.6f}   "
                  f"2p: {r['E_2p']:.6f} -> {nr['E_2p']:.6f}", flush=True)
        new.append(nr)
    return add_fd_pressure(new)


def load_csv(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k, v in r.items():
            if not (k.startswith("zeta") or k.startswith("conv")):
                r[k] = float(v)
    return rows


def save_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def make_figures(rows, prefix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "axes.labelsize": 13, "font.size": 11})
    Rc = np.array([r["Rc"] for r in rows])
    V0 = rows[0]["V0"]
    g = lambda k: np.array([r[k] for r in rows])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(Rc, g("E_2s"), "o-", label=r"$1s^2 2s$")
    ax.plot(Rc, g("E_2p"), "s--", label=r"$1s^2 2p$")
    ax.set(xlabel=r"$R_c$ (u.a.)", ylabel=r"$E_{av}$ (Hartree)", title=f"Energia media, $V_0$ = {V0:g}")
    ax.grid(alpha=0.4); ax.legend(); fig.tight_layout()
    fig.savefig(f"{prefix}_energias.png", dpi=300)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    axs[0].plot(Rc, g("eps1s_2s"), "o-", label=r"$\varepsilon_{1s}$")
    axs[0].plot(Rc, g("eps2s"), "s-", label=r"$\varepsilon_{2s}$")
    axs[0].set_title(r"$1s^2 2s$")
    axs[1].plot(Rc, g("eps1s_2p"), "o-", label=r"$\varepsilon_{1s}$")
    axs[1].plot(Rc, g("eps2p"), "s-", label=r"$\varepsilon_{2p}$")
    axs[1].set_title(r"$1s^2 2p$")
    for a in axs:
        a.set(xlabel=r"$R_c$ (u.a.)", ylabel="Energia orbital (Hartree)"); a.grid(alpha=0.4); a.legend()
    fig.tight_layout(); fig.savefig(f"{prefix}_energias_orbitales.png", dpi=300)

    fig, ax = plt.subplots(figsize=(7, 5))
    for k, lab, st in (("Pin_1s_2s", "1s-In", "k-"), ("Pout_1s_2s", "1s-Out", "k--"),
                       ("Pin_val_2s", "2s-In", "C0-"), ("Pout_val_2s", "2s-Out", "C0--"),
                       ("Pin_val_2p", "2p-In", "C1-"), ("Pout_val_2p", "2p-Out", "C1--")):
        ax.plot(Rc, g(k), st, marker="o", ms=3, label=lab)
    ax.set(xlabel=r"$R_c$ (u.a.)", ylabel="Probabilidad", title=f"$P_{{in}}$ y $P_{{out}}$, $V_0$ = {V0:g}")
    ax.grid(alpha=0.4); ax.legend(ncol=2); fig.tight_layout()
    fig.savefig(f"{prefix}_probabilidades.png", dpi=300)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(Rc, g("r_1s_2s"), "o-", label=r"$\langle r\rangle_{1s}$")
    ax.plot(Rc, g("r_val_2s"), "s-", label=r"$\langle r\rangle_{2s}$")
    ax.plot(Rc, g("r_val_2p"), "^-", label=r"$\langle r\rangle_{2p}$")
    ax.set(xlabel=r"$R_c$ (u.a.)", ylabel=r"$\langle r\rangle$ (u.a.)"); ax.grid(alpha=0.4); ax.legend()
    fig.tight_layout(); fig.savefig(f"{prefix}_radio_medio.png", dpi=300)

    if "alpha_zz" in rows[0]:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(Rc, g("alpha_zz"), "d-")
        ax.set(xlabel=r"$R_c$ (u.a.)", ylabel=r"$\alpha_{zz}$ (u.a.)",
               title=r"Polarizabilidad dipolar, $1s^2 2s$"); ax.grid(alpha=0.4)
        fig.tight_layout(); fig.savefig(f"{prefix}_polarizabilidad.png", dpi=300)


# =====================================================================
# 11. Ejecucion
# =====================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Li confinado en pozo esferico finito (HFR, bases globales)")
    ap.add_argument("--V0", type=float, default=5.0)
    ap.add_argument("--Rc", type=float, nargs="+",
                    default=[15.0, 12.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.5, 5.0, 4.5, 4.0, 3.5, 3.0,
                             2.75, 2.5, 2.25, 2.0, 1.75, 1.5, 1.25, 1.0, 0.9, 0.8])
    ap.add_argument("--polarizabilidad", action="store_true",
                    help="calcula tambien alpha_zz por campo finito (no se usa en la tesis)")
    ap.add_argument("--de-maxiter", type=int, default=40,
                    help="generaciones de Evolucion Diferencial por radio (0 = solo cascada + SLSQP)")
    ap.add_argument("--salida", default="litio")
    ap.add_argument("--semillas", metavar="CSV", nargs="*", default=[],
                    help="con --pulir: CSV de otras corridas cuyos exponentes se usan como semillas")
    ap.add_argument("--pulir", metavar="CSV",
                    help="no hace barrido: pule un CSV existente reoptimizando desde los radios vecinos")
    ap.add_argument("--recalcular", metavar="CSV",
                    help="no optimiza: recalcula todas las propiedades con los exponentes guardados en el CSV "
                         "(por ejemplo, para agregar las entropias en momentos)")
    ap.add_argument("--libre", action="store_true",
                    help="atomo libre (V0 = 0): optimiza los exponentes y reproduce la Tabla 4.1 de la tesis")
    ap.add_argument("--prueba-momentos", action="store_true",
                    help="pruebas del calculo en el espacio de momentos y de la desigualdad BBM")
    args = ap.parse_args()

    if args.prueba_momentos:
        momentum_selftest()
        raise SystemExit

    if args.libre:
        free_atom(de_maxiter=args.de_maxiter)
        raise SystemExit

    if args.recalcular:
        old = load_csv(args.recalcular)
        V0 = old[0]["V0"]
        opt2s, opt2p = StateOptimizer("2s", V0), StateOptimizer("2p", V0)
        rows = []
        for r in old:
            x2s = np.array([float(v) for v in r["zeta_2s"].split()])
            x2p = np.array([float(v) for v in r["zeta_2p"].split()])
            nr = build_row(V0, r["Rc"], x2s, x2p, opt2s, opt2p)
            dE = max(abs(nr["E_2s"] - r["E_2s"]), abs(nr["E_2p"] - r["E_2p"]))
            print(f"Rc={r['Rc']:6.3f}  |dE| = {dE:.1e}  "
                  f"2s: S_r+S_p = {nr['S_val_2s'] + nr['Sp_val_2s']:.4f}   "
                  f"2p: S_r+S_p = {nr['S_val_2p'] + nr['Sp_val_2p']:.4f}", flush=True)
            rows.append(nr)
        rows = add_fd_pressure(rows)
        save_csv(rows, args.recalcular)
        check_bbm(rows)
        print(f"CSV actualizado: {args.recalcular}")
        raise SystemExit

    if args.pulir:
        extra = [r for f in args.semillas for r in load_csv(f)]
        viejas = load_csv(args.pulir)
        rows = polish(viejas, viejas[0]["V0"], extra=extra)     # V0 se lee del propio CSV
        save_csv(rows, args.pulir)
        make_figures(rows, args.pulir[:-4])
        print(f"CSV pulido: {args.pulir}")
        raise SystemExit

    Rcs = sorted(args.Rc, reverse=True)   # barrido descendente obligatorio (cascada)
    rows = run_sweep(args.V0, Rcs, do_polar=args.polarizabilidad, de_maxiter=args.de_maxiter)
    rows = add_fd_pressure(rows)
    save_csv(rows, f"{args.salida}_V0_{args.V0:g}.csv")
    check_bbm(rows)
    make_figures(rows, f"{args.salida}_V0_{args.V0:g}")
    print(f"Resultados en {args.salida}_V0_{args.V0:g}.csv y figuras *.png")
