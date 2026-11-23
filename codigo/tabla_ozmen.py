"""Tabla 4.2 de la tesis en LaTeX: energias medias de configuracion en la malla (limite
de Hartree-Fock), con la base de STO (este trabajo) y de Ozmen et al. (2023).

Uso:
    python tabla_ozmen.py                        # usa los archivos de ../datos
    python tabla_ozmen.py ../datos/litio_V0_1.csv ../datos/litio_V0_5.csv \
           --malla ../datos/malla/barrido_V0_1.json ../datos/malla/barrido_V0_5.json

Sin --malla se imprime la tabla sin la columna de la malla. Al final imprime, como
comentarios, las diferencias con el limite que se citan en el texto de la Seccion 4.2.
"""
import argparse
import os

import numpy as np

from figuras_tesis import CSV_DEFECTO, MALLA_DEFECTO, OZMEN, cargar_malla, load

ap = argparse.ArgumentParser()
ap.add_argument("csv", nargs="*", default=CSV_DEFECTO)
ap.add_argument("--malla", nargs="*", default=None)
a = ap.parse_args()
if a.malla is None:
    a.malla = [f for f in MALLA_DEFECTO if os.path.exists(f)]

csvs = {float(load(p)["V0"][0]): load(p) for p in a.csv}
malla = cargar_malla(a.malla)
con_malla = all(V0 in malla for V0 in csvs)

if con_malla:
    print(r"\begin{tabular}{cc|rrr|rrr}")
    print(r"\hline")
    print(r" & & \multicolumn{3}{c|}{$E_{\mathrm{av}}(1s^2 2s)$} & \multicolumn{3}{c}{$E_{\mathrm{av}}(1s^2 2p)$} \\")
    print(r"$V_0$ & $R_c$ & Malla & STO & Özmen & Malla & STO & Özmen \\ \hline")
else:
    print(r"\begin{tabular}{cc|rr|rr}")
    print(r"\hline")
    print(r" & & \multicolumn{2}{c|}{$E_{\mathrm{av}}(1s^2 2s)$} & \multicolumn{2}{c}{$E_{\mathrm{av}}(1s^2 2p)$} \\")
    print(r"$V_0$ & $R_c$ & STO & Özmen & STO & Özmen \\ \hline")

notas = []
for V0 in sorted(csvs):
    D = csvs[V0]
    for k, R in enumerate(OZMEN[V0]["Rc"]):
        i = int(np.argmin(abs(D["Rc"] - R)))
        assert abs(D["Rc"][i] - R) < 1e-6
        o2s, o2p = OZMEN[V0]["2s"][k], OZMEN[V0]["2p"][k]
        if con_malla:
            m2s, m2p = malla[V0]["2s"][R]["E"], malla[V0]["2p"][R]["E"]
            print(f"{V0:g} & {R:g} & {m2s:.4f} & {D['E_2s'][i]:.4f} & {o2s:.4f} & "
                  f"{m2p:.4f} & {D['E_2p'][i]:.4f} & {o2p:.4f} \\\\")
            notas.append((V0, R, 1e3 * (D["E_2s"][i] - m2s), 1e3 * (o2s - m2s),
                          1e3 * (D["E_2p"][i] - m2p), 1e3 * (o2p - m2p)))
        else:
            print(f"{V0:g} & {R:g} & {D['E_2s'][i]:.4f} & {o2s:.4f} & {D['E_2p'][i]:.4f} & {o2p:.4f} \\\\")
    print(r"\hline")
print(r"\end{tabular}")

if con_malla:
    print("%\n% Diferencias con el limite de Hartree-Fock (mHartree): STO y Ozmen")
    print("%   V0    Rc   2s: STO   Ozmen    2p: STO    Ozmen")
    for V0, R, a2s, b2s, a2p, b2p in notas:
        print(f"%  {V0:3g} {R:5g} {a2s:9.1f} {b2s:7.1f} {a2p:10.1f} {b2p:8.1f}")
    n_abajo = sum(1 for t in notas if t[5] < 0)
    print(f"% Ozmen 1s2 2p por debajo del limite en {n_abajo} de {len(notas)} radios")
