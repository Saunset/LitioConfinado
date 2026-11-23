"""
Hartree-Fock numérico (sin base) para el átomo de litio 1s^2 2s (^2S)
confinado por un escalón de potencial  V_c(r) = V0 * Theta(r - Rc).

Variable logarítmica:  rho = ln r,   P(r) = r^{1/2} y(rho)
Ecuación para cada orbital (l = 0):

    -1/2 y'' + [1/8 + r^2 V(r)] y + r^{3/2} (términos de Hartree e intercambio)
        = eps r^2 y   (+ multiplicadores fuera de la diagonal)

Discretización:
  * y'' con diferencias finitas de 5 puntos  (error O(h^4)).
  * Integrales  ∫ P_a P_b / r_>  con trapecio + corrección de Euler-Maclaurin
    en el quiebre r' = r  (error O(h^4)).
  * Escalón en Rc: Rc es un nodo de la malla, V(Rc) = V0/2, y se corrige la
    cuadratura de  ∫ r^2 V y^2  por el salto (Euler-Maclaurin).
  * Pared impenetrable (V0 = inf): y(Rc) = 0 con reflexión impar.
  * r_min = 1e-10: cerca del origen y ~ r^{1/2} decae lento en rho y cortar
    antes introduce un error ~ Z^3 r_min en la energía.
  * El funcional de energía discreto y las matrices de Fock son consistentes:
    las ecuaciones de Fock son exactamente la derivada del funcional discreto.

Autoconsistencia:
  * 1s se obtiene en el complemento ortogonal de 2s y viceversa, por
    iteración inversa con un sistema bordeado (no por diagonalización densa,
    que con B = diag(r^2) y r_min = 1e-10 pierde toda la precisión).
  * En cada iteración se optimiza el ángulo de rotación 1s-2s, lo que impone
    la condición  w_1s eps_{1s,2s} = w_2s eps_{2s,1s}  (2 eps_12 = eps_21).

Uso:
    python hf_litio_confinado.py                     # pruebas de verificación (Tabla D.1)
    python hf_litio_confinado.py --Rc 2 --V0 1       # escalón finito: 1s^2 2s y 1s^2 2p
    python hf_litio_confinado.py --Rc 2              # pared impenetrable (solo 1s^2 2s)
    python hf_litio_confinado.py --Rc 2 --V0 1 --h 0.01

Con un solo punto de partida el ciclo puede converger a otra solución autoconsistente;
para los resultados de la tesis se usa barrido.py, que prueba varios puntos de partida.

Unidades atómicas de Hartree.
"""

import argparse
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar, brentq
from scipy.integrate import solve_ivp


# ----------------------------------------------------------------------
# Malla y operadores
# ----------------------------------------------------------------------
class Malla:
    def __init__(self, Z=3.0, Rc=np.inf, V0=0.0, h=0.02, rmin=1e-10, rmax=50.0):
        self.Z, self.Rc, self.V0, self.h = Z, Rc, V0, h
        self.pared = np.isfinite(Rc) and np.isinf(V0)

        # anclamos la malla para que Rc caiga exactamente en un nodo
        x0 = np.log(Rc) if np.isfinite(Rc) else 0.0
        kmin = int(np.ceil((np.log(rmin) - x0) / h))
        if self.pared:
            kmax = -1                      # la pared (y = 0) no es incógnita
        else:
            rmax = max(rmax, (Rc + 40.0) if np.isfinite(Rc) else rmax)
            kmax = int(np.floor((np.log(rmax) - x0) / h))
        self.x = x0 + h * np.arange(kmin, kmax + 1)
        self.r = np.exp(self.x)
        N = self.N = self.r.size

        # segunda derivada, 5 puntos, fantasmas nulos a la izquierda
        c = 1.0 / (12 * h * h)
        D2 = (np.diag(np.full(N, -30.0))
              + np.diag(np.full(N - 1, 16.0), 1) + np.diag(np.full(N - 1, 16.0), -1)
              + np.diag(np.full(N - 2, -1.0), 2) + np.diag(np.full(N - 2, -1.0), -2)) * c
        if self.pared:
            # reflexión impar en la pared: y_{N+1} = -y_{N-1}
            D2[-1, -1] += 1.0 * c
        self.D2 = D2

        # potencial local
        r = self.r
        Vc = np.zeros(N)
        if np.isfinite(Rc) and not self.pared:
            Vc[r > Rc * (1 + 1e-12)] = V0
            Vc[np.isclose(r, Rc, rtol=1e-12)] = 0.5 * V0   # valor medio en el escalón
        self.V = -Z / r + Vc

        # operador de un electrón en la forma A y = eps B y
        self.T = -0.5 * D2 + np.diag(np.full(N, 1.0 / 8.0))
        self.Aloc = self.T + np.diag(r**2 * self.V)
        if np.isfinite(Rc) and not self.pared and V0 != 0.0:
            self.Aloc += self._correccion_escalon()
        self.Bd = r**2                                    # B diagonal

        # núcleo de Coulomb  S_ij = h r_i^2 r_j^2 / max(r_i, r_j)
        rmaxij = np.maximum.outer(r, r)
        self.S = h * np.outer(r**2, r**2) / rmaxij
        self.M = h * (r**2)[None, :] / rmaxij              # para J(r_i)
        # núcleo k = 1 (para G^1 en 1s^2 2p):  r_< / r_>^2
        rminij = np.minimum.outer(r, r)
        self.S1 = h * np.outer(r**2, r**2) * rminij / rmaxij**2
        # término centrífugo l(l+1)/2 en la forma y (l = 1)
        self.Aloc_p = self.Aloc + np.eye(N) * 1.0

    def _correccion_escalon(self):
        """Corrección de Euler-Maclaurin a  ∫ r^2 V y^2 drho  en el salto de V.

        El trapecio con V(Rc) = V0/2 deja el error -(h^2/12)[f'_- - f'_+],
        con f = r^2 V y^2. Como y y y' son continuas,
            f'_- - f'_+ = -2 V0 Rc^2 (y_c^2 + y_c y'_c),
        así que se suma la forma cuadrática  (h^2/6) V0 Rc^2 (y_c^2 + y_c y'_c).
        Se escribe simétrica para que A siga siendo hermítica.
        """
        h, N = self.h, self.N
        c = int(np.argmin(np.abs(self.r - self.Rc)))
        d = np.zeros(N)
        d[[c - 2, c - 1, c + 1, c + 2]] = np.array([1.0, -8.0, 8.0, -1.0]) / (12 * h)
        e = np.zeros(N)
        e[c] = 1.0
        return (h / 6.0) * self.V0 * self.Rc**2 * (
            np.outer(e, e) + 0.5 * (np.outer(e, d) + np.outer(d, e)))

    # producto interno  <a|b> = ∫ P_a P_b dr = h Σ r^2 y_a y_b
    def prod(self, a, b):
        return self.h * np.sum(self.Bd * a * b)

    def normaliza(self, y):
        return y / np.sqrt(self.prod(y, y))

    # Y^0_{ab}(r)/r = ∫ P_a P_b / r_>  dr'
    def J(self, ya, yb=None):
        yb = ya if yb is None else yb
        f = ya * yb
        return self.M @ f - (self.h**2 / 12.0) * self.r * f

    # operador de intercambio con el orbital b (simétrico)
    def X(self, yb):
        Xm = np.outer(yb, yb) * self.S
        Xm[np.diag_indices(self.N)] -= (self.h**2 / 12.0) * self.r**3 * yb**2
        return Xm

    # operador de intercambio de rango k = 1 con el orbital b
    # corrección de Euler-Maclaurin en el quiebre: -(h^2/12)(2k+1) r y_a y_b
    def X1(self, yb):
        Xm = np.outer(yb, yb) * self.S1
        Xm[np.diag_indices(self.N)] -= 3.0 * (self.h**2 / 12.0) * self.r**3 * yb**2
        return Xm

    # P(r) a partir de y
    def P(self, y):
        return np.sqrt(self.r) * y


# ----------------------------------------------------------------------
# Funcional de energía y matrices de Fock para 1s^2 2s
# ----------------------------------------------------------------------
def energia(m, y1, y2, detalle=False):
    h = m.h
    J1, J2 = m.J(y1), m.J(y2)
    I1 = h * y1 @ m.Aloc @ y1
    I2 = h * y2 @ m.Aloc @ y2
    F11 = h * np.sum(m.Bd * y1**2 * J1)
    F12 = h * np.sum(m.Bd * y1**2 * J2)
    G12 = h * y1 @ m.X(y2) @ y1
    E = 2 * I1 + I2 + F11 + 2 * F12 - G12
    if detalle:
        T = h * (2 * y1 @ m.T @ y1 + y2 @ m.T @ y2)
        return E, dict(I1=I1, I2=I2, F11=F11, F12=F12, G12=G12, T=T)
    return E


def fock(m, y1, y2):
    B = m.Bd
    A1 = m.Aloc + np.diag(B * (m.J(y1) + m.J(y2))) - 0.5 * m.X(y2)
    A2 = m.Aloc + np.diag(B * 2.0 * m.J(y1)) - m.X(y1)
    return A1, A2


def iteracion_inversa(m, A, y, yb=None, nit=2):
    """Iteración inversa para A y = eps B y.

    Si se da yb, la solución se restringe al complemento B-ortogonal de yb
    mediante el sistema bordeado
        [A - eps B   B yb] [z]   [B y]
        [(B yb)^T     0  ] [mu] = [ 0 ]
    No se usa la factorización de Cholesky de B (que con r_min ~ 1e-10 es
    numéricamente inútil); la iteración inversa sí es precisa.
    """
    Bd = m.Bd
    N = m.N
    for _ in range(nit):
        eps = (y @ A @ y) / (y @ (Bd * y))
        if yb is None:
            z = np.linalg.solve(A - eps * np.diag(Bd), Bd * y)
        else:
            K = np.zeros((N + 1, N + 1))
            K[:N, :N] = A - eps * np.diag(Bd)
            K[:N, N] = K[N, :N] = Bd * yb
            rhs = np.append(Bd * y, 0.0)
            z = np.linalg.solve(K, rhs)[:N]
        y = alinea(m.normaliza(z), y)
    return y


def alinea(y, yref):
    return y if np.dot(y, yref) >= 0 else -y


def ortonormaliza(m, y1, y2):
    """Löwdin simétrico en la métrica B."""
    Smat = np.array([[m.prod(y1, y1), m.prod(y1, y2)],
                     [m.prod(y2, y1), m.prod(y2, y2)]])
    w, U = np.linalg.eigh(Smat)
    X = U @ np.diag(w**-0.5) @ U.T
    Y = np.vstack([y1, y2]).T @ X
    return Y[:, 0], Y[:, 1]


def rota(y1, y2, t):
    c, s = np.cos(t), np.sin(t)
    return c * y1 + s * y2, -s * y1 + c * y2


def nodos(y):
    yy = y[np.abs(y) > 1e-8 * np.abs(y).max()]
    return int(np.sum(np.diff(np.sign(yy)) != 0))


def guess(m, k2=1):
    """Orbitales iniciales con el número de nodos correcto.

    1s: estado base del hamiltoniano de un electrón (núcleo + confinamiento).
    2s: segundo estado del mismo hamiltoniano apantallado por 2 J_1s.
    La diagonalización densa se hace solo en r > 1e-4 (ahí sí es precisa) y
    sirve para identificar el estado; la iteración inversa refina después.
    """
    idx = m.r > 1e-4
    Bsub = np.diag(m.Bd[idx])

    def estado(A, k):
        _, v = eigh(A[np.ix_(idx, idx)], Bsub, subset_by_index=[k, k])
        y = np.zeros(m.N)
        y[idx] = v[:, 0]
        if y[idx][0] < 0:
            y = -y
        return m.normaliza(y)

    y1 = estado(m.Aloc, 0)
    y2 = estado(m.Aloc + np.diag(m.Bd * 2.0 * m.J(y1)), k2)
    y2 = m.normaliza(y2 - m.prod(y1, y2) * y1)
    return y1, y2


def scf(m, alfa=0.7, tol=1e-12, maxit=200, inicio=None, verbose=False):
    y1, y2 = guess(m) if inicio is None else ortonormaliza(m, *inicio)
    E_old = energia(m, y1, y2)
    for it in range(1, maxit + 1):
        A1, A2 = fock(m, y1, y2)
        n1 = iteracion_inversa(m, A1, y1, yb=y2)
        n2 = iteracion_inversa(m, A2, y2, yb=y1)
        y1, y2 = ortonormaliza(m, alfa * n1 + (1 - alfa) * y1,
                                  alfa * n2 + (1 - alfa) * y2)
        # optimización del ángulo de rotación 1s-2s
        res = minimize_scalar(lambda t: energia(m, *rota(y1, y2, t)),
                              bounds=(-0.2, 0.2), method="bounded",
                              options=dict(xatol=1e-12))
        y1, y2 = rota(y1, y2, res.x)
        E = energia(m, y1, y2)
        if verbose:
            print(f"  it {it:3d}  E = {E:.12f}  dE = {E - E_old:.2e}")
        if abs(E - E_old) < tol and it > 5:
            break
        E_old = E
    if nodos(y1) != 0 or nodos(y2) != 1:
        print(f"  AVISO: nodos 1s = {nodos(y1)}, 2s = {nodos(y2)} (se esperaba 0 y 1)")
    return y1, y2, E, it


def analiza(m, y1, y2):
    A1, A2 = fock(m, y1, y2)
    h = m.h
    e1 = h * y1 @ A1 @ y1
    e2 = h * y2 @ A2 @ y2
    e12 = h * y2 @ A1 @ y1          # <2s|F_1s|1s>
    e21 = h * y1 @ A2 @ y2          # <1s|F_2s|2s>
    E, d = energia(m, y1, y2, detalle=True)
    return dict(E=E, e1s=e1, e2s=e2, e12=e12, e21=e21, T=d["T"],
                virial=-(E - d["T"]) / d["T"])


# ----------------------------------------------------------------------
# Configuración 1s^2 2p (^2P)
#   E_av = 2 I(1s) + I(2p) + F0(1s,1s) + 2 F0(1s,2p) - (1/3) G1(1s,2p)
# 1s y 2p tienen l distinta: son ortogonales por la parte angular y no hay
# multiplicadores fuera de la diagonal.
# ----------------------------------------------------------------------
def energia_2p(m, y1, yp):
    h = m.h
    J1, Jp = m.J(y1), m.J(yp)
    I1 = h * y1 @ m.Aloc @ y1
    Ip = h * yp @ m.Aloc_p @ yp
    F11 = h * np.sum(m.Bd * y1**2 * J1)
    F1p = h * np.sum(m.Bd * y1**2 * Jp)
    G1 = h * y1 @ m.X1(yp) @ y1
    return 2 * I1 + Ip + F11 + 2 * F1p - G1 / 3.0


def fock_2p(m, y1, yp):
    B = m.Bd
    A1 = m.Aloc + np.diag(B * (m.J(y1) + m.J(yp))) - m.X1(yp) / 6.0
    Ap = m.Aloc_p + np.diag(B * 2.0 * m.J(y1)) - m.X1(y1) / 3.0
    return A1, Ap


def guess_2p(m, kp=0):
    idx = m.r > 1e-4
    Bsub = np.diag(m.Bd[idx])

    def estado(A, k):
        _, v = eigh(A[np.ix_(idx, idx)], Bsub, subset_by_index=[k, k])
        y = np.zeros(m.N)
        y[idx] = v[:, 0]
        if y[idx][0] < 0:
            y = -y
        return m.normaliza(y)

    y1 = estado(m.Aloc, 0)
    yp = estado(m.Aloc_p + np.diag(m.Bd * 2.0 * m.J(y1)), kp)
    return y1, yp


def scf_2p(m, alfa=0.7, tol=1e-12, maxit=200, inicio=None, verbose=False):
    y1, yp = guess_2p(m) if inicio is None else inicio
    E_old = energia_2p(m, y1, yp)
    for it in range(1, maxit + 1):
        A1, Ap = fock_2p(m, y1, yp)
        n1 = iteracion_inversa(m, A1, y1)
        npp = iteracion_inversa(m, Ap, yp)
        y1 = m.normaliza(alfa * n1 + (1 - alfa) * y1)
        yp = m.normaliza(alfa * npp + (1 - alfa) * yp)
        E = energia_2p(m, y1, yp)
        if verbose:
            print(f"  it {it:3d}  E = {E:.12f}  dE = {E - E_old:.2e}")
        if abs(E - E_old) < tol and it > 5:
            break
        E_old = E
    return y1, yp, E, it


def analiza_2p(m, y1, yp):
    A1, Ap = fock_2p(m, y1, yp)
    h = m.h
    return dict(E=energia_2p(m, y1, yp),
                e1s=h * y1 @ A1 @ y1, e2p=h * yp @ Ap @ yp)


def p_fuera(m, y):
    """Probabilidad de encontrar al electrón en r > Rc (nodo en Rc con peso 1/2)."""
    w = m.h * m.Bd * y**2
    fuera = m.r > m.Rc * (1 + 1e-12)
    borde = np.isclose(m.r, m.Rc, rtol=1e-12)
    return np.sum(w[fuera]) + 0.5 * np.sum(w[borde])


def r_medio(m, y):
    return m.h * np.sum(m.Bd * m.r * y**2)


# ----------------------------------------------------------------------
# Verificación independiente: un electrón, disparo con solve_ivp
# ----------------------------------------------------------------------
def un_electron_fd(m):
    P = m.r * np.exp(-m.Z * m.r)
    if m.pared:
        P = P * (1 - m.r / m.Rc)
    y = iteracion_inversa(m, m.Aloc, m.normaliza(P / np.sqrt(m.r)), nit=10)
    return (y @ m.Aloc @ y) / (y @ (m.Bd * y))


def un_electron_disparo(Z, Rc, V0, e_guess, ventana=0.05, rmax=60.0):
    """P'' = 2(V - eps) P. Hacia afuera desde 0 hasta Rc y hacia adentro desde rmax."""
    def V(r):
        return -Z / r + (V0 if r > Rc else 0.0)

    def rhs(r, u):
        return [u[1], 2.0 * (V(r) - u[2]) * u[0], 0.0]

    opts = dict(method="DOP853", rtol=1e-12, atol=1e-30)

    def W(eps):
        r0 = 1e-7
        sol = solve_ivp(rhs, [r0, Rc], [r0 - Z * r0**2, 1 - 2 * Z * r0, eps], **opts)
        Lout = sol.y[1, -1] / sol.y[0, -1]
        k = np.sqrt(2 * (V0 - eps))
        Rf = max(rmax, Rc + 40 / k)
        sol = solve_ivp(rhs, [Rf, Rc], [1e-30, -k * 1e-30, eps], **opts)
        Lin = sol.y[1, -1] / sol.y[0, -1]
        return Lout - Lin

    return brentq(W, e_guess - ventana, e_guess + ventana, xtol=1e-14)


# ----------------------------------------------------------------------
def pruebas():
    print("=" * 70)
    print("1) Un electrón, pared impenetrable (resultado exacto)")
    print("   H con Rc = 2: el estado base es la 2s libre truncada, E = -1/8")
    for h in (0.04, 0.02, 0.01):
        m = Malla(Z=1.0, Rc=2.0, V0=np.inf, h=h)
        e = un_electron_fd(m)
        print(f"   h = {h:<6} E = {e:.12f}   error = {e + 0.125:.2e}")

    print("\n2) Un electrón (Z = 3), escalón finito: diferencias finitas vs disparo")
    for Rc, V0 in ((1.0, 1.0), (0.5, 2.0), (0.4, 10.0)):
        m = Malla(Z=3.0, Rc=Rc, V0=V0, h=0.01)
        efd = un_electron_fd(m)
        edp = un_electron_disparo(3.0, Rc, V0, efd)
        print(f"   Rc = {Rc:<4} V0 = {V0:<5} FD = {efd:.10f}   disparo = {edp:.10f}"
              f"   dif = {efd - edp:.1e}")

    print("\n3) Li libre: límite Hartree-Fock")
    print("   Referencia: E = -7.432727, eps_1s = -2.477741, eps_2s = -0.196323")
    for h in (0.04, 0.02, 0.01):
        m = Malla(Z=3.0, h=h)
        y1, y2, E, it = scf(m)
        a = analiza(m, y1, y2)
        print(f"   h = {h:<5} E = {a['E']:.9f}  eps1s = {a['e1s']:.6f}"
              f"  eps2s = {a['e2s']:.6f}  -V/T = {a['virial']:.8f}  ({it} it)")
    print(f"   Estacionariedad: 2 eps_12 = {2 * a['e12']:.9f},  eps_21 = {a['e21']:.9f}")
    m = Malla(Z=3.0, h=0.02)
    y1, yp, E, it = scf_2p(m)
    a = analiza_2p(m, y1, yp)
    print(f"   1s^2 2p, h = 0.02: E = {a['E']:.9f}  eps1s = {a['e1s']:.6f}  eps2p = {a['e2p']:.6f}  ({it} it)")

    print("\n4) Li confinado: convergencia en h")
    for Rc, V0 in ((2.0, np.inf), (2.0, 1.0)):
        for h in (0.04, 0.02, 0.01):
            m = Malla(Z=3.0, Rc=Rc, V0=V0, h=h)
            y1, y2, E, it = scf(m)
            a = analiza(m, y1, y2)
            print(f"   Rc = {Rc} V0 = {V0:<4} h = {h:<5} E = {a['E']:.10f}"
                  f"  eps2s = {a['e2s']:.7f}  ({it} it)")


def corrida(Rc, V0, h):
    m = Malla(Z=3.0, Rc=Rc, V0=V0, h=h)
    y1, y2, E, it = scf(m)
    a = analiza(m, y1, y2)
    print(f"Li confinado: Rc = {Rc}, V0 = {V0}, h = {h}, N = {m.N}")
    print(f"  E          = {a['E']:.10f}")
    print(f"  eps_1s     = {a['e1s']:.8f}")
    print(f"  eps_2s     = {a['e2s']:.8f}")
    print(f"  2 eps_12   = {2 * a['e12']:.6e}   eps_21 = {a['e21']:.6e}")
    print(f"  iteraciones = {it}")
    if np.isfinite(Rc) and np.isfinite(V0) and V0 > a["e2s"]:
        # cola fuera de la caja: P_2s ~ r^{1/k} exp(-k r)
        k = np.sqrt(2 * (V0 - a["e2s"]))
        P2 = m.P(y2)
        sel = (m.r > Rc + 3 / k) & (m.r < Rc + 8 / k)
        pend = np.gradient(np.log(np.abs(P2[sel])), m.r[sel])
        teo = -k + 1 / (k * m.r[sel])
        print(f"  cola 2s: d lnP/dr numérico vs -k + 1/(k r): "
              f"max dif = {np.abs(pend - teo).max():.1e}  (k = {k:.4f})")
    print(f"  <r>_1s     = {r_medio(m, y1):.6f}   <r>_2s = {r_medio(m, y2):.6f}")
    if np.isfinite(Rc) and np.isfinite(V0):
        print(f"  P_out(1s)  = {p_fuera(m, y1):.6f}   P_out(2s) = {p_fuera(m, y2):.6f}")
        rho = np.interp(Rc, m.r, (2 * m.P(y1)**2 + m.P(y2)**2) / (4 * np.pi * m.r**2))
        print(f"  presión    = V0 rho(Rc) = {V0 * rho:.6e} u.a. = {V0 * rho * 29421.02648:.2f} GPa")
        # configuración excitada 1s^2 2p
        z1, zp, E2p, itp = scf_2p(m)
        b = analiza_2p(m, z1, zp)
        print("Configuración 1s^2 2p:")
        print(f"  E          = {b['E']:.10f}   (E(2p) - E(2s) = {b['E'] - a['E']:+.6f})")
        print(f"  eps_1s     = {b['e1s']:.8f}")
        print(f"  eps_2p     = {b['e2p']:.8f}")
        print(f"  <r>_2p     = {r_medio(m, zp):.6f}   P_out(2p) = {p_fuera(m, zp):.6f}")
        print(f"  iteraciones = {itp}   (nodos de u_1s, u_2p: {nodos(z1)}, {nodos(zp)})")
    return m, y1, y2, a


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--Rc", type=float, default=None)
    ap.add_argument("--V0", type=float, default=np.inf)
    ap.add_argument("--h", type=float, default=0.02)
    ap.add_argument("--pruebas", action="store_true")
    args = ap.parse_args()
    if args.pruebas or args.Rc is None:
        pruebas()
    if args.Rc is not None:
        corrida(args.Rc, args.V0, args.h)
