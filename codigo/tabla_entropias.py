"""Tabla 4.5 de la tesis en LaTeX: entropias de Shannon S_r, S_p y sumas S_t = S_r + S_p
con la base de STO, a partir de los CSV de litio_confinado_hf.py. Como comentarios
imprime el minimo y el maximo de cada suma y su distancia a la cota 3(1 + ln pi).

Uso:
    python tabla_entropias.py                                # usa los CSV de ../datos
    python tabla_entropias.py ../datos/litio_V0_5.csv ../datos/litio_V0_1.csv
"""
import sys
import numpy as np
from figuras_tesis import CSV_DEFECTO, load, BBM

SEL = {5.0: [15, 5, 3, 2, 1.5, 1.25, 1.0, 0.8], 1.0: [15, 5, 3, 2.25, 2, 1.5, 1.0, 0.8]}


def f(x):
    return f"${x:.3f}$" if x >= 0 else f"$-{abs(x):.3f}$"


for path in (sys.argv[1:] or CSV_DEFECTO):
    D = load(path)
    V0 = D["V0"][0]
    print(f"% ---- V0 = {V0:g} ----")
    print(rf"\multicolumn{{10}}{{c}}{{$V_0 = {V0:g}$}} \\ \hline")
    for R in SEL[V0]:
        i = int(np.argmin(abs(D["Rc"] - R)))
        c = [D["S_val_2s"][i], D["Sp_val_2s"][i], D["S_val_2s"][i] + D["Sp_val_2s"][i],
             D["S_1s_2s"][i] + D["Sp_1s_2s"][i], D["Sr_tot_2s"][i] + D["Sp_tot_2s"][i],
             D["S_val_2p"][i], D["Sp_val_2p"][i], D["S_val_2p"][i] + D["Sp_val_2p"][i],
             D["Sr_tot_2p"][i] + D["Sp_tot_2p"][i]]
        print(f"${D['Rc'][i]:g}$ & " + " & ".join(f(x) for x in c) + r" \\")
    # minimos (para el texto)
    for k, (a, b) in {"2s": ("S_val_2s", "Sp_val_2s"), "1s(2s)": ("S_1s_2s", "Sp_1s_2s"),
                      "tot2s": ("Sr_tot_2s", "Sp_tot_2s"), "2p": ("S_val_2p", "Sp_val_2p"),
                      "2pz": ("S_val_2p_pz", "Sp_val_2p_pz"), "1s(2p)": ("S_1s_2p", "Sp_1s_2p"),
                      "tot2p": ("Sr_tot_2p", "Sp_tot_2p")}.items():
        St = D[a] + D[b]
        j = int(np.argmin(St)); J = int(np.argmax(St))
        print(f"%   {k:7s} min S_t = {St[j]:.4f} en Rc = {D['Rc'][j]:g};  max = {St[J]:.4f} en Rc = {D['Rc'][J]:g};"
              f"  margen min sobre BBM = {St[j] - BBM:.4f}")
