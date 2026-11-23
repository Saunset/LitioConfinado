"""Pruebas rapidas del programa de STO (litio_confinado_hf.py). Uso:

    python pruebas.py                 # ~1 minuto
    python pruebas.py --datos ../datos

Comprueba, contra resultados exactos o independientes:
  1. simbolos 3-j con proyecciones nulas (Apendice A);
  2. integrales de un electron de un STO 1s: T = zeta^2/2, V_nuc = -Z zeta, y el
     confinamiento V_conf contra integracion numerica (Apendice B);
  3. integrales de dos electrones contra las formulas cerradas hidrogenoides:
     F0(1s,1s) = 5 zeta/8,  F0(1s,2p)  y  G1(1s,2p) (Apendice B);
  4. el atomo hidrogenoide (un electron) y el ion Li+ (1s^2) con un solo exponente;
  5. la transformada de Fourier de los STO, las entropias exactas del 1s y el teorema
     de Plancherel (Apendice E);
  6. que los CSV de datos/ se reproducen con los exponentes que guardan, y que la
     desigualdad S_r + S_p >= 3(1 + ln pi) se cumple en todos los radios.
Termina con codigo 0 si todas las pruebas pasan.
"""
import argparse
import math
import os
import sys

import numpy as np
from scipy.integrate import quad

import litio_confinado_hf as hf

fallas = []


def prueba(nombre, valor, esperado, tol):
    ok = abs(valor - esperado) <= tol
    print(f"  [{'OK' if ok else 'FALLA'}] {nombre}: {valor:.10g} (esperado {esperado:.10g}, tol {tol:g})")
    if not ok:
        fallas.append(nombre)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datos"))
    a = ap.parse_args()

    print("1. Simbolos 3-j (l1 l2 l3; 0 0 0)")
    prueba("(0 0 0)", hf.three_j_000(0, 0, 0), 1.0, 1e-14)
    prueba("(0 1 1)", hf.three_j_000(0, 1, 1), -1 / math.sqrt(3), 1e-14)
    prueba("(1 1 2)", hf.three_j_000(1, 1, 2), math.sqrt(2 / 15), 1e-14)
    prueba("(1 1 1) = 0 por paridad", hf.three_j_000(1, 1, 1), 0.0, 0)
    prueba("(0 2 1) = 0 por el triangulo", hf.three_j_000(0, 2, 1), 0.0, 0)

    print("2. Integrales de un electron")
    z, Rc, V0 = 2.7, 1.3, 5.0
    B = hf.Basis([z], [1.1])
    S, T, Vn, Vc, _ = hf.one_electron_matrices(B, Rc, V0)
    prueba("S(1s,1s)", S[0, 0], 1.0, 1e-13)
    prueba("T(1s,1s) = zeta^2/2", T[0, 0], z ** 2 / 2, 1e-12)
    prueba("V_nuc(1s,1s) = -Z zeta", Vn[0, 0], -3 * z, 1e-12)
    prueba("V_conf(1s,1s) = V0 e^(-2 zeta Rc)(1 + 2 zeta Rc + 2 zeta^2 Rc^2)", Vc[0, 0],
           V0 * math.exp(-2 * z * Rc) * (1 + 2 * z * Rc + 2 * z ** 2 * Rc ** 2), 1e-13)
    num = V0 * quad(lambda r: (B.radial(np.array([r]))[0, 1]) ** 2 * r ** 2, Rc, 80)[0]
    prueba("V_conf(2p,2p) contra cuadratura", Vc[1, 1], num, 1e-9)
    prueba("T(2p,2p) = zeta^2/2", T[1, 1], 1.1 ** 2 / 2, 1e-12)

    print("3. Integrales de dos electrones (formulas hidrogenoides)")
    al, be = 2.7, 1.3                      # 1s ~ e^(-al r),  2p ~ r e^(-be r/2)
    B = hf.Basis([al], [be / 2])
    Rk = hf.radial_eri(B, kmax=2)
    prueba("F0(1s,1s) = 5 alfa/8", Rk[0, 0, 0, 0, 0], 5 * al / 8, 1e-13)
    prueba("F0(1s,2p)", Rk[0, 0, 0, 1, 1], be / 4 - be ** 5 * (6 * al + be) / (4 * (2 * al + be) ** 5), 1e-13)
    prueba("G1(1s,2p) = 112 a^3 b^5/(2a+b)^7", Rk[1, 0, 1, 1, 0], 112 * al ** 3 * be ** 5 / (2 * al + be) ** 7, 1e-13)
    eri = hf.full_eri(B, Rk)
    prueba("(1s 2p|2p 1s) = G1/3", eri[0, 1, 1, 0], Rk[1, 0, 1, 1, 0] / 3, 1e-14)

    print("4. Sistemas con solucion conocida")
    B = hf.Basis([3.0], [1.0])
    S, T, Vn, Vc, _ = hf.one_electron_matrices(B, 50.0, 0.0)
    prueba("Li2+ (un electron), E = -Z^2/2", (T + Vn)[0, 0], -4.5, 1e-12)
    zopt = 3 - 5 / 16                       # exponente optimo de 1s^2 con un solo STO
    B = hf.Basis([zopt], [1.0])
    S, T, Vn, Vc, _ = hf.one_electron_matrices(B, 50.0, 0.0)
    E = 2 * (T + Vn)[0, 0] + hf.radial_eri(B, 0)[0, 0, 0, 0, 0]
    prueba("Li+ (1s^2) con un STO, E = -(Z - 5/16)^2", E, -zopt ** 2, 1e-12)

    print("5. Espacio de momentos (Apendice E)")
    for z in (1.0, 2.7):
        B = hf.Basis([z], [])
        c = np.array([1.0])
        Sr = hf.radial_entropy((B.radial(hf.R_GRID) @ c) ** 2, hf.R_GRID) + hf.S_ANG_SPH
        Sp = hf.radial_entropy((B.momentum(hf.P_GRID) @ c) ** 2, hf.P_GRID) + hf.S_ANG_SPH
        prueba(f"S_r(1s, zeta={z})", Sr, 3 + math.log(math.pi) - 3 * math.log(z), 5e-6)
        prueba(f"S_p(1s, zeta={z})", Sp, math.log(32 * math.pi ** 2) - 10 / 3 + 3 * math.log(z), 5e-6)
        prueba(f"S_r + S_p (zeta={z}) = ln(32 pi^3) - 1/3", Sr + Sp, math.log(32 * math.pi ** 3) - 1 / 3, 5e-6)
    B = hf.Basis([1.3], [0.9])
    for mu, l in ((0, 0), (1, 1)):
        pv = 1.7
        from scipy.special import spherical_jn
        num = math.sqrt(2 / math.pi) * quad(lambda r: spherical_jn(l, pv * r) * B.radial(np.array([r]))[0, mu] * r ** 2,
                                            0, 80, limit=400)[0]
        prueba(f"F(p) analitica contra cuadratura (l={l})", B.momentum(np.array([pv]))[0, mu], num, 1e-9)
        prueba(f"Plancherel: int F^2 p^2 dp (l={l})",
               hf.trapz(B.momentum(hf.P_GRID)[:, mu] ** 2 * hf.P_GRID ** 2, hf.P_GRID), 1.0, 5e-6)

    print("6. Datos guardados")
    for V0 in (5, 1):
        path = os.path.join(a.datos, f"litio_V0_{V0}.csv")
        if not os.path.exists(path):
            print(f"  (no se encontro {path}; se omite)")
            continue
        rows = hf.load_csv(path)
        opt2s, opt2p = hf.StateOptimizer("2s", float(V0)), hf.StateOptimizer("2p", float(V0))
        dmax = 0.0
        for r in rows:
            x2s = np.array([float(v) for v in r["zeta_2s"].split()])
            x2p = np.array([float(v) for v in r["zeta_2p"].split()])
            dmax = max(dmax, abs(opt2s.energy(x2s, r["Rc"]) - r["E_2s"]), abs(opt2p.energy(x2p, r["Rc"]) - r["E_2p"]))
        prueba(f"V0 = {V0}: energias reproducidas con los exponentes guardados (max |dE|)", dmax, 0.0, 2e-6)
        smin = min(min(r[a_] + r[b_] for r in rows) for a_, b_ in
                   (("S_1s_2s", "Sp_1s_2s"), ("S_val_2s", "Sp_val_2s"), ("Sr_tot_2s", "Sp_tot_2s"),
                    ("S_1s_2p", "Sp_1s_2p"), ("S_val_2p_pz", "Sp_val_2p_pz"), ("Sr_tot_2p", "Sp_tot_2p")))
        ok = smin > hf.BBM_COTA
        print(f"  [{'OK' if ok else 'FALLA'}] V0 = {V0}: min(S_r + S_p) = {smin:.4f} > 3(1 + ln pi) = {hf.BBM_COTA:.4f}")
        if not ok:
            fallas.append(f"BBM V0={V0}")

    print()
    if fallas:
        print("FALLARON:", "; ".join(fallas))
        sys.exit(1)
    print("Todas las pruebas pasaron.")


if __name__ == "__main__":
    main()
