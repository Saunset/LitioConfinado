"""Entropías de Shannon en posiciones (S_r) y en momentos (S_p) para orbitales en la malla.

En la malla x = ln r el orbital se guarda como y(x), con u(r) = r R(r) = r^(1/2) y(x).
  Posiciones:  R(r)^2 = y^2 / r,      int R^2 r^2 dr = h sum r^2 y^2 = 1
               S_r = -h sum r^2 y^2 ln(y^2 / r) + ln(4 pi)
  Momentos:    F(p) = sqrt(2/pi) int j_l(p r) R(r) r^2 dr = sqrt(2/pi) int j_l(p r) u(r) r dr
               S_p = -int F^2 ln F^2 p^2 dp + ln(4 pi)      (regla del trapecio en p)
               La malla logarítmica no sirve para esta integral cuando p es grande: en
               r ~ 1 su paso es dr = r h ~ 0.02 y j_l(p r) oscila más rápido que eso para
               p > ~50. Por eso interpolamos u(r) con splines cúbicos a una malla uniforme
               con dr = 0.0025 (válida hasta p ~ pi/dr ~ 1250) y usamos p <= 500. Usamos un
               spline dentro y otro fuera de la cavidad, porque u'' salta en r = Rc.
La parte angular ln(4 pi) corresponde a una densidad esférica (orbital s, o p promediado
en m como en la energía media de configuración). Para la densidad total de 1s^2 nl,
normalizada a 1, se usa (2 R_1s^2 + R_nl^2)/3 y (2 F_1s^2 + F_nl^2)/3.
"""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import spherical_jn

P = np.exp(np.linspace(np.log(1e-4), np.log(5e2), 3001))
DR = 0.0025
S_ANG = np.log(4 * np.pi)
COTA_BBM = 3 * (1 + np.log(np.pi))



def _u_uniforme(m, y):
    """u(r) = r^(1/2) y en una malla uniforme fina, con splines separados en r < Rc y r > Rc."""
    u = np.sqrt(m.r) * y
    big = np.nonzero(np.abs(u) > 1e-13 * np.abs(u).max())[0]
    rcut = m.r[big[-1]]
    ru = np.arange(0.0, rcut + DR, DR)
    uu = np.zeros_like(ru)
    if np.isfinite(m.Rc) and m.Rc < rcut:
        ins, out = m.r <= m.Rc * (1 + 1e-12), m.r >= m.Rc * (1 - 1e-12)
        si = CubicSpline(np.concatenate([[0.0], m.r[ins]]), np.concatenate([[0.0], u[ins]]))
        so = CubicSpline(m.r[out], u[out])
        uu = np.where(ru <= m.Rc, si(ru), so(ru))
    else:
        uu = CubicSpline(np.concatenate([[0.0], m.r]), np.concatenate([[0.0], u]))(ru)
    return ru, uu


def transformada(m, y, l, bloque=200):
    """F(p) = sqrt(2/pi) int j_l(p r) u(r) r dr en la malla P (trapecio uniforme)."""
    ru, uu = _u_uniforme(m, y)
    w = np.full_like(ru, DR)
    w[0] = w[-1] = DR / 2
    g = w * uu * ru
    F = np.empty_like(P)
    for i in range(0, P.size, bloque):
        F[i:i + bloque] = spherical_jn(l, np.outer(P[i:i + bloque], ru)) @ g
    return np.sqrt(2 / np.pi) * F


def _s(f, q, w):
    """-int f ln f w dq con f ya normalizada (w = q^2 dq)."""
    lf = np.log(np.where(f > 1e-300, f, 1e-300))
    return -np.sum(w * f * lf)


def entropias(m, y1, yv, lv):
    """S_r y S_p del 1s, del orbital de valencia (l = lv) y de la densidad total."""
    wr = m.h * m.r ** 3            # r^2 dr = r^3 dx
    R1, Rv = y1 ** 2 / m.r, yv ** 2 / m.r
    F1 = transformada(m, y1, 0)
    Fv = transformada(m, yv, lv)
    wp = np.zeros_like(P)          # pesos del trapecio en p, multiplicados por p^2
    dp = np.diff(P)
    wp[:-1] += dp / 2
    wp[1:] += dp / 2
    wp *= P ** 2
    out = {}
    for tag, fr, fp in (("1s", R1, F1 ** 2), ("nl", Rv, Fv ** 2),
                        ("tot", (2 * R1 + Rv) / 3, (2 * F1 ** 2 + Fv ** 2) / 3)):
        nr, npn = np.sum(wr * fr), np.sum(wp * fp)
        out[f"Sr_{tag}"] = _s(fr / nr, m.r, wr) + S_ANG
        out[f"Sp_{tag}"] = _s(fp / npn, P, wp) + S_ANG
        out[f"normp_{tag}"] = npn
    return out


if __name__ == "__main__":
    # Prueba: orbitales STO puestos en la malla, comparados con la transformada exacta.
    import math
    from hf_litio_confinado import Malla
    m = Malla(Z=3.0, Rc=2.0, V0=5.0, h=0.02)
    for z in (1.0, 2.7):
        y = np.sqrt(m.r) * 2 * z ** 1.5 * np.exp(-z * m.r)        # 1s: R = 2 z^(3/2) e^(-z r)
        y = m.normaliza(y)
        e = entropias(m, y, y, 0)
        Sr = 3 + math.log(math.pi) - 3 * math.log(z)
        Sp = math.log(32 * math.pi ** 2) - 10 / 3 + 3 * math.log(z)
        print(f"STO 1s zeta={z}: S_r={e['Sr_1s']:.7f} (exacto {Sr:.7f})  "
              f"S_p={e['Sp_1s']:.7f} (exacto {Sp:.7f})  norma p = {e['normp_1s']:.8f}")
    # 2p STO: R = N r e^(-z r); F exacta = sqrt(2/pi) N 8 z p / (z^2 + p^2)^3
    z = 1.3
    N = (2 * z) ** 2.5 / math.sqrt(24)
    y = m.normaliza(np.sqrt(m.r) * N * m.r * np.exp(-z * m.r))
    F_num = transformada(m, y, 1)
    F_ex = math.sqrt(2 / math.pi) * N * 8 * z * P / (z ** 2 + P ** 2) ** 3
    print(f"STO 2p zeta={z}: max |F_malla - F_exacta| = {np.max(np.abs(F_num - F_ex)):.2e}")
