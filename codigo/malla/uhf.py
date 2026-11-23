"""Hartree-Fock no restringido (UHF) para el litio 1s^2 2s en la malla.

En el cálculo de la tesis los dos electrones 1s comparten el mismo orbital espacial
(Hartree-Fock restringido de capa abierta). Aquí dejamos que 1s(alfa) y 1s(beta) sean
distintos: el electrón 1s(alfa) tiene intercambio con el 2s(alfa) y el 1s(beta) no, así
que sus orbitales se separan un poco (polarización de espín de la capa interna).

Energía (a = 1s alfa, b = 1s beta, c = 2s alfa):
    E = I_a + I_b + I_c + F0(a,b) + F0(a,c) + F0(b,c) - G0(a,c)
Ecuaciones:
    [h + J_b + J_c - K_c] a = e_a a      (a ortogonal a c)
    [h + J_a + J_c      ] b = e_b b
    [h + J_a + J_b - K_a] c = e_c c      (c ortogonal a a)

Es la prueba que se menciona en las perspectivas de la tesis (Capítulo 5).

Uso:
    python uhf.py 0 0          # átomo libre
    python uhf.py 5 2          # V0 = 5, Rc = 2
"""
import sys

import numpy as np

from hf_litio_confinado import Malla, iteracion_inversa, ortonormaliza, scf, nodos


def energia_uhf(m, a, b, c):
    h, B = m.h, m.Bd
    Jb, Jc = m.J(b), m.J(c)
    I = sum(h * y @ m.Aloc @ y for y in (a, b, c))
    Fab = h * np.sum(B * a**2 * Jb)
    Fac = h * np.sum(B * a**2 * Jc)
    Fbc = h * np.sum(B * b**2 * Jc)
    Gac = h * a @ m.X(c) @ a
    return I + Fab + Fac + Fbc - Gac


def scf_uhf(m, y1, y2, alfa=0.7, tol=1e-12, maxit=300):
    """Parte de la solución restringida (y1 = 1s, y2 = 2s)."""
    a, b, c = y1.copy(), y1.copy(), y2.copy()
    E_old = energia_uhf(m, a, b, c)
    for it in range(1, maxit + 1):
        B = m.Bd
        Ja, Jb, Jc = m.J(a), m.J(b), m.J(c)
        Aa = m.Aloc + np.diag(B * (Jb + Jc)) - m.X(c)
        Ab = m.Aloc + np.diag(B * (Ja + Jc))
        Ac = m.Aloc + np.diag(B * (Ja + Jb)) - m.X(a)
        na = iteracion_inversa(m, Aa, a, yb=c)
        nb = iteracion_inversa(m, Ab, b)
        nc = iteracion_inversa(m, Ac, c, yb=a)
        a, c = ortonormaliza(m, alfa * na + (1 - alfa) * a, alfa * nc + (1 - alfa) * c)
        b = m.normaliza(alfa * nb + (1 - alfa) * b)
        E = energia_uhf(m, a, b, c)
        if abs(E - E_old) < tol and it > 5:
            break
        E_old = E
    return a, b, c, E, it


if __name__ == "__main__":
    V0, Rc = float(sys.argv[1]), float(sys.argv[2])
    h = float(sys.argv[3]) if len(sys.argv) > 3 else 0.02
    m = Malla(Z=3.0, Rc=Rc, V0=V0, h=h) if Rc > 0 else Malla(Z=3.0, h=h)
    y1, y2, E_r, _ = scf(m)
    a, b, c, E_u, it = scf_uhf(m, y1, y2)
    print(f"V0 = {V0:g}, Rc = {Rc if Rc > 0 else 'inf'}:")
    print(f"  E restringido (ROHF) = {E_r:.8f}")
    print(f"  E no restringido     = {E_u:.8f}   ({it} ciclos; nodos a, b, c = {nodos(a)}, {nodos(b)}, {nodos(c)})")
    print(f"  diferencia           = {E_u - E_r:.2e} Hartree")
    print(f"  traslape <1s alfa|1s beta> = {m.prod(a, b):.8f}")
