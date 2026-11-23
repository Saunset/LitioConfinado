"""Tabla 4.4 de la tesis: valores de u = r R_2s y de sus derivadas a la izquierda y a
la derecha de la pared, para la configuracion 1s^2 2s con la base de STO. Muestra que
con funciones base globales u y u' son continuas en r = Rc, y que u'' tambien lo es,
cuando en la solucion exacta deberia saltar 2 V0 u(Rc).

Uso:
    python tabla_continuidad.py                               # V0 = 5, Rc = 3, 2, 1.25
    python tabla_continuidad.py ../datos/litio_V0_1.csv --Rc 3 2
"""
import argparse

import numpy as np

from figuras_tesis import CSV_DEFECTO
from litio_confinado_hf import Basis, load_csv, scf_eav

ap = argparse.ArgumentParser()
ap.add_argument("csv", nargs="?", default=CSV_DEFECTO[0])
ap.add_argument("--Rc", type=float, nargs="+", default=[3.0, 2.0, 1.25])
a = ap.parse_args()

rows = {r["Rc"]: r for r in load_csv(a.csv)}
h = 1e-3                                   # paso de las derivadas laterales
print(" Rc     u(Rc)    u' izq / der        u'' izq / der     salto exacto 2 V0 u(Rc)")
for Rc in a.Rc:
    r = rows[Rc]
    V0 = r["V0"]
    zs = np.array([float(v) for v in r["zeta_2s"].split()])
    B = Basis(zs, [3.0, 2.0, 1.2, 0.7, 0.3])          # el bloque p no interviene en 1s^2 2s
    res = scf_eav(B, Rc, V0, "2s")
    u = lambda x: x * (B.radial(np.atleast_1d(x)) @ res["c2"])[0]
    # formulas laterales de tres y cuatro puntos (error de orden h^2)
    d1 = lambda s: s * (-3 * u(Rc) + 4 * u(Rc + s * h) - u(Rc + 2 * s * h)) / (2 * h)
    d2 = lambda s: (2 * u(Rc) - 5 * u(Rc + s * h) + 4 * u(Rc + 2 * s * h) - u(Rc + 3 * s * h)) / h ** 2
    print(f"{Rc:5g}  {u(Rc):7.3f}   {d1(-1):8.4f} / {d1(1):8.4f}   {d2(-1):8.3f} / {d2(1):8.3f}   {2 * V0 * u(Rc):8.2f}")
