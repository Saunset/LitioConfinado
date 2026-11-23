"""Tabla 4.6 de la tesis: compara las entropias S_r, S_p y S_t = S_r + S_p de la base de
STO (CSV de litio_confinado_hf.py) con las de la malla (JSON de malla/barrido.py), y
comprueba la cota de Bialynicki-Birula y Mycielski en todos los puntos de la malla.
Tambien imprime, para la malla, E(1s2 2p) - E(1s2 2s) y la diferencia de las sumas
entropicas totales en R_c <= 4 (discusion "estado base y estado excitado").

Uso:
    python comparar_entropias_malla.py           # usa los archivos de ../datos
    python comparar_entropias_malla.py ../datos/litio_V0_5.csv ../datos/malla/barrido_V0_5.json \
                                       ../datos/litio_V0_1.csv ../datos/malla/barrido_V0_1.json
(los argumentos van por pares CSV JSON)."""
import json
import sys

import numpy as np

from figuras_tesis import CSV_DEFECTO, MALLA_DEFECTO, load, BBM

SEL = [3, 2, 1.25, 1.0, 0.8]
# clave en el CSV (STO) -> clave en el JSON (malla), por configuracion
PARES = {
    "2s": {"St_1s": (("S_1s_2s", "Sp_1s_2s"), ("Sr_1s", "Sp_1s")),
           "St_val": (("S_val_2s", "Sp_val_2s"), ("Sr_nl", "Sp_nl")),
           "St_tot": (("Sr_tot_2s", "Sp_tot_2s"), ("Sr_tot", "Sp_tot")),
           "Sr_val": (("S_val_2s",), ("Sr_nl",)), "Sp_val": (("Sp_val_2s",), ("Sp_nl",))},
    "2p": {"St_1s": (("S_1s_2p", "Sp_1s_2p"), ("Sr_1s", "Sp_1s")),
           "St_val": (("S_val_2p", "Sp_val_2p"), ("Sr_nl", "Sp_nl")),
           "St_tot": (("Sr_tot_2p", "Sp_tot_2p"), ("Sr_tot", "Sp_tot")),
           "Sr_val": (("S_val_2p",), ("Sr_nl",)), "Sp_val": (("Sp_val_2p",), ("Sp_nl",))},
}


def main():
    args = sys.argv[1:] or [f for par in zip(CSV_DEFECTO, MALLA_DEFECTO) for f in par]
    filas = []
    for csvp, jsp in zip(args[::2], args[1::2]):
        D = load(csvp)
        M = json.load(open(jsp))
        V0 = D["V0"][0]
        print(f"===== V0 = {V0:g} =====")
        # cota BBM en la malla
        peor = (np.inf, None)
        for conf in ("2s", "2p"):
            for R, d in M[conf].items():
                for tag in ("1s", "nl", "tot"):
                    st = d[f"Sr_{tag}"] + d[f"Sp_{tag}"]
                    if st < peor[0]:
                        peor = (st, (conf, R, tag))
                    for k in ("normp_1s", "normp_nl"):
                        if abs(d[k] - 1) > 1e-4:
                            print("  AVISO norma en p:", conf, R, k, d[k])
        print(f"  malla: minimo S_t = {peor[0]:.4f} en {peor[1]} (cota {BBM:.4f})")
        # diferencias STO - malla en los radios comunes
        for conf in ("2s", "2p"):
            Rm = {float(k): v for k, v in M[conf].items()}
            for q, (kc, km) in PARES[conf].items():
                dif = []
                for i, R in enumerate(D["Rc"]):
                    if R in Rm:
                        sto = sum(D[k][i] for k in kc)
                        mal = sum(Rm[R][k] for k in km)
                        dif.append((abs(sto - mal), R))
                mx = max(dif)
                print(f"  {conf} {q:7s}: max |STO - malla| = {mx[0]:.4f} (Rc = {mx[1]:g}); "
                      f"mediana = {np.median([d[0] for d in dif]):.4f}")
            # orden de las sumas totales en la malla
        R2s = {float(k): v for k, v in M["2s"].items()}
        R2p = {float(k): v for k, v in M["2p"].items()}
        print("  malla: Rc, dE = E(2p)-E(2s), dS_t(total) = S_t(2p)-S_t(2s)")
        for R in sorted(set(R2s) & set(R2p)):
            if R <= 4:
                a, b = R2s[R], R2p[R]
                print(f"    {R:5g}  dE = {b['E'] - a['E']:+.4f}   dSt = "
                      f"{(b['Sr_tot'] + b['Sp_tot']) - (a['Sr_tot'] + a['Sp_tot']):+.4f}")
        # filas de la tabla LaTeX
        for R in SEL:
            i = int(np.argmin(abs(D["Rc"] - R)))
            if R not in R2s or R not in R2p:
                continue
            c = []
            for conf, Rm in (("2s", R2s), ("2p", R2p)):
                for q in ("St_val", "St_tot"):
                    kc, km = PARES[conf][q]
                    c += [sum(D[k][i] for k in kc), sum(Rm[R][k] for k in km)]
            filas.append((V0, R, c))
    print("\n% ---- filas para la tabla (S_t: STO, malla) ----")
    V0prev = None
    for V0, R, c in filas:
        if V0 != V0prev:
            print(rf"\multicolumn{{9}}{{c}}{{$V_0 = {V0:g}$}} \\ \hline")
            V0prev = V0
        print(f"${R:g}$ & " + " & ".join(f"${x:.3f}$" for x in c) + r" \\")


if __name__ == "__main__":
    main()
