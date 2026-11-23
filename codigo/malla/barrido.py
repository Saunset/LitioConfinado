"""Barrido en Rc para Li confinado (escalón finito), configuraciones 1s^2 2s y 1s^2 2p.
En cada radio se prueban varios puntos de partida y se conserva la solución
autoconsistente de menor energía con el número de nodos correcto."""
import sys, json, warnings
import numpy as np
from hf_litio_confinado import (Malla, scf, guess, analiza, scf_2p, guess_2p,
                                analiza_2p, nodos, p_fuera, r_medio)
from entropias import entropias
warnings.filterwarnings("ignore")

V0 = float(sys.argv[1]); h = float(sys.argv[2]); salida = sys.argv[3]
RCS = [15, 10, 8, 6, 5, 4.5, 4.25, 4, 3.75, 3.5, 3.25, 3, 2.75, 2.5, 2.25, 2,
       1.75, 1.5, 1.25, 1.1, 1, 0.9, 0.85, 0.8]

def a_malla(m, sol):
    """Interpola P de otra malla a esta y devuelve y."""
    r0, P0 = sol
    P = np.interp(m.r, r0, P0, left=0.0, right=0.0)
    return P / np.sqrt(m.r)

def resolver(m, conf, inicio):
    try:
        if conf == "2s":
            y1, y2, E, it = scf(m, inicio=inicio)
            ok = nodos(y1) == 0 and nodos(y2) == 1 and it < 200
        else:
            y1, y2, E, it = scf_2p(m, inicio=inicio)
            ok = nodos(y1) == 0 and nodos(y2) == 0 and it < 200
        if not ok or not np.isfinite(E):
            return None
        return E, y1, y2
    except Exception:
        return None

def candidatos(m, conf, vecinos):
    c = []
    if conf == "2s":
        for k in (1, 2, 3):
            try: c.append(guess(m, k2=k))
            except Exception: pass
    else:
        for k in (0, 1):
            try: c.append(guess_2p(m, kp=k))
            except Exception: pass
    for v in vecinos:
        c.append((m.normaliza(a_malla(m, v[0])), m.normaliza(a_malla(m, v[1]))))
    return c

res = {"2s": {}, "2p": {}}
mejores = {"2s": {}, "2p": {}}
for conf in ("2s", "2p"):
    for paso, orden in enumerate((RCS, RCS[::-1])):
        prev = None
        for Rc in orden:
            m = Malla(Z=3.0, Rc=Rc, V0=V0, h=h)
            vec = [prev] if prev is not None else []
            if paso == 1 and Rc in mejores[conf]:
                vec.append(mejores[conf][Rc][1])
            best = None
            starts = candidatos(m, conf, vec) if paso == 0 else candidatos(m, conf, vec)[-len(vec):] if vec else []
            for ini in starts:
                r_ = resolver(m, conf, ini)
                if r_ is not None and (best is None or r_[0] < best[0] - 1e-10):
                    best = r_
            if best is None:
                prev = None
                continue
            E, y1, y2 = best
            sol = ((m.r.copy(), m.P(y1)), (m.r.copy(), m.P(y2)))
            if Rc not in mejores[conf] or E < mejores[conf][Rc][0] - 1e-10:
                a = analiza(m, y1, y2) if conf == "2s" else analiza_2p(m, y1, y2)
                d = dict(E=float(E), e1s=float(a["e1s"]),
                         enl=float(a["e2s"] if conf == "2s" else a["e2p"]),
                         pout1s=float(p_fuera(m, y1)), poutnl=float(p_fuera(m, y2)),
                         r1s=float(r_medio(m, y1)), rnl=float(r_medio(m, y2)),
                         rho_Rc=float(np.interp(Rc, m.r, (2*m.P(y1)**2 + m.P(y2)**2)/(4*np.pi*m.r**2))))
                # Entropías de Shannon en posiciones y momentos (1s, valencia y densidad total)
                d.update({k: float(v) for k, v in entropias(m, y1, y2, 0 if conf == "2s" else 1).items()})
                mejores[conf][Rc] = (E, sol)
                res[conf][str(Rc)] = d
                print(f"V0={V0} {conf} Rc={Rc:<5} paso {paso}: E={E:.7f} enl={d['enl']:.5f} "
                      f"Pout={d['poutnl']:.3f}", flush=True)
            prev = mejores[conf][Rc][1]

json.dump(res, open(salida, "w"), indent=1)
