"""Bases 'even-tempered' grandes (zeta_k = a b^k) evaluadas en la malla.
Solo se optimizan coeficientes lineales; la dependencia lineal se trata con
ortogonalización canónica (se descartan direcciones con valor propio de S < tol)."""
import sys, numpy as np, warnings
warnings.filterwarnings("ignore")
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar
from hf_litio_confinado import Malla, energia, fock, rota, energia_2p, fock_2p

def base(m, zetas, n):
    X = np.array([m.r**n*np.exp(-z*m.r) for z in zetas]).T/np.sqrt(m.r)[:, None]
    return X/np.sqrt(m.h*np.sum(m.Bd[:, None]*X**2, axis=0))

def canon(m, X, tol=1e-11):
    S = m.h*X.T@(m.Bd[:, None]*X)
    w, U = np.linalg.eigh(S)
    k = w > tol*w.max()
    return X@(U[:, k]/np.sqrt(w[k]))          # columnas ortonormales en la métrica B

def sub(Q, A, cb=None):
    F = Q.T@A@Q  # Q es ortonormal en la métrica h*B
    if cb is not None:
        P = np.eye(len(cb)) - np.outer(cb, cb)
        F = P@F@P + 1e6*np.outer(cb, cb)
    return eigh(F)[1][:, 0]

def hf2s(m, Q):
    h = m.h
    c = eigh(h*Q.T@m.Aloc@Q)[1]; c1, c2 = c[:, 0], c[:, 1]
    Eo = np.inf
    for _ in range(200):
        y1, y2 = Q@c1, Q@c2
        A1, A2 = fock(m, y1, y2)
        n1 = sub(Q, h*A1, c2); n2 = sub(Q, h*A2, n1)
        c1 = 0.6*np.sign(n1@c1)*n1 + 0.4*c1; c1 /= np.linalg.norm(c1)
        c2 = 0.6*np.sign(n2@c2)*n2 + 0.4*c2; c2 -= (c2@c1)*c1; c2 /= np.linalg.norm(c2)
        y1, y2 = Q@c1, Q@c2
        r = minimize_scalar(lambda t: energia(m, *rota(y1, y2, t)), bounds=(-.3, .3), method="bounded",
                            options=dict(xatol=1e-10))
        ct, st = np.cos(r.x), np.sin(r.x); c1, c2 = ct*c1+st*c2, -st*c1+ct*c2
        if abs(r.fun-Eo) < 1e-11: break
        Eo = r.fun
    return r.fun

def hf2p(m, Qs, Qp):
    h = m.h
    c1 = eigh(h*Qs.T@m.Aloc@Qs)[1][:, 0]; cp = eigh(h*Qp.T@m.Aloc_p@Qp)[1][:, 0]
    Eo = np.inf
    for _ in range(200):
        A1, Ap = fock_2p(m, Qs@c1, Qp@cp)
        n1 = eigh(h*Qs.T@A1@Qs)[1][:, 0]; npp = eigh(h*Qp.T@Ap@Qp)[1][:, 0]
        c1 = 0.6*np.sign(n1@c1)*n1+0.4*c1; c1 /= np.linalg.norm(c1)
        cp = 0.6*np.sign(npp@cp)*npp+0.4*cp; cp /= np.linalg.norm(cp)
        E = energia_2p(m, Qs@c1, Qp@cp)
        if abs(E-Eo) < 1e-11: break
        Eo = E
    return E

if __name__ == "__main__":
    V0, Rc, h = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
    m = Malla(Z=3.0, Rc=Rc, V0=V0, h=h) if Rc > 0 else Malla(Z=3.0, h=h)
    for N in (10, 20, 30):
        z = 0.08*(200/0.08)**(np.arange(N)/(N-1))
        Qs = canon(m, base(m, z, 1)); Qp = canon(m, base(m, z, 2))
        print(f"ET V0={V0} Rc={Rc} N={N} rango_s={Qs.shape[1]} E2s={hf2s(m,Qs):.6f} E2p={hf2p(m,Qs,Qp):.6f}", flush=True)
