# Átomo de litio confinado: Hartree-Fock con orbitales tipo Slater y en una malla radial

Código, datos y figuras de la tesis de licenciatura

> **Estudio de la Estructura Electrónica y Propiedades del Átomo de Litio Confinado mediante el Método de Hartree-Fock**
> Andrés Valenzuela Leyva — Departamento de Física, Universidad de Sonora (2026).
> Director de tesis: Dr. Adrian Duarte Alcaraz.

El repositorio contiene dos programas independientes que resuelven el mismo problema, el átomo de litio dentro de una cavidad esférica de radio $R_c$ con una barrera de potencial finita de altura $V_0$:

| Programa | Carpeta | Qué hace |
|---|---|---|
| **Base de STO** | `codigo/litio_confinado_hf.py` | Hartree-Fock-Roothaan con 10 orbitales tipo Slater cuyos exponentes se optimizan en cada radio. Todas las integrales son analíticas. Es el programa principal de la tesis. |
| **Malla radial** | `codigo/malla/` | Las mismas ecuaciones resueltas sin funciones base, en una malla de puntos radiales. Da el límite de Hartree-Fock y sirve para medir el error de la base. |

Todo está en **unidades atómicas de Hartree** (longitudes en radios de Bohr, energías en Hartree).

---

## Contenido

1. [El problema físico](#1-el-problema-físico)
2. [Estructura del repositorio](#2-estructura-del-repositorio)
3. [Instalación](#3-instalación)
4. [Inicio rápido](#4-inicio-rápido)
5. [Programa con base de STO](#5-programa-con-base-de-sto)
6. [Programa en la malla radial](#6-programa-en-la-malla-radial)
7. [Formato de los datos](#7-formato-de-los-datos)
8. [Cómo reproducir cada tabla y figura de la tesis](#8-cómo-reproducir-cada-tabla-y-figura-de-la-tesis)
9. [Pruebas](#9-pruebas)
10. [Usar el código desde Python](#10-usar-el-código-desde-python)
11. [Limitaciones conocidas](#11-limitaciones-conocidas)
12. [Referencias](#12-referencias)

---

## 1. El problema físico

El hamiltoniano del átomo de litio ($Z = 3$, tres electrones) confinado es

$$
\hat H = \sum_{i=1}^{3}\left[-\tfrac12\nabla_i^2 - \frac{Z}{r_i} + V_c(r_i)\right] + \sum_{i<j}\frac{1}{|\mathbf r_i - \mathbf r_j|},
\qquad
V_c(r) = \begin{cases} 0, & r < R_c \\ V_0, & r \ge R_c \end{cases}
$$

Es el modelo de Özmen *et al.* (2023). Como la barrera es finita, los electrones pueden atravesarla (efecto túnel) y, si la cavidad es muy pequeña, el electrón de valencia sale de ella.

Se estudian dos configuraciones, el estado base $1s^2 2s$ y la configuración excitada $1s^2 2p$, con el método de Hartree-Fock de capa abierta en la versión de **energía media de configuración** (Cowan, cap. 6):

$$
E_{av} = 2I_{1s} + I_{nl} + F^0(1s,1s) + 2F^0(1s,nl) - w_l\,G^l(1s,nl), \qquad w_l = \frac{1}{2l+1}
$$

donde $I$ son las energías de un electrón (cinética, núcleo y confinamiento) y $F^k$, $G^k$ las integrales radiales de Slater. De pedir que $E_{av}$ sea mínima salen dos operadores de Fock, uno por subcapa:

$$
\hat F^{(1s)} = \hat h + \hat J_{1s} + \hat J_{nl} - \tfrac12 \tilde K_{nl}, \qquad
\hat F^{(nl)} = \hat h + 2\hat J_{1s} - \tilde K_{1s}
$$

**Qué se calcula** en función de $R_c$ (de 15 a 0.8 u.a.) para $V_0 = 1$ y $5$ Hartree:

- energía total $E_{av}$ y energías orbitales $\varepsilon_{1s}$, $\varepsilon_{nl}$;
- radio medio $\langle r\rangle$ y probabilidad de encontrar a cada electrón fuera de la cavidad, $P_{out}$;
- presión sobre la pared, de dos formas: $P = V_0\,\rho(R_c)$ (teorema de Hellmann-Feynman) y $P = -\frac{1}{4\pi R_c^2}\frac{dE}{dR_c}$ (diferencias finitas);
- entropías de Shannon en el espacio de posiciones, $S_r$, y de momentos, $S_p$, de cada orbital y de la densidad total, y la desigualdad de Białynicki-Birula y Mycielski, $S_r + S_p \ge 3(1+\ln\pi) \approx 6.434$;
- la inversión de los niveles $2s$ y $2p$ (con $V_0 = 5$, $1s^2 2p$ queda por debajo de $1s^2 2s$ para $R_c \lesssim 3$ u.a.).

---

## 2. Estructura del repositorio

```
README.md                         este archivo
requirements.txt                  bibliotecas de Python necesarias
codigo/
    litio_confinado_hf.py         PROGRAMA PRINCIPAL (base de STO): integrales, SCF, optimización, barrido
    pruebas.py                    pruebas rápidas del programa (unos segundos)
    figuras_tesis.py              todas las figuras del Capítulo 4
    tabla_ozmen.py                Tabla 4.2 (malla / STO / Özmen et al.)
    tabla_continuidad.py          Tabla 4.4 (u, u', u'' en la pared)
    tabla_entropias.py            Tabla 4.5 (S_r, S_p y sus sumas)
    comparar_entropias_malla.py   Tabla 4.6 (entropías: STO contra malla)
    malla/
        hf_litio_confinado.py     PROGRAMA DE LA MALLA: operadores, funcional de energía, SCF, pruebas
        barrido.py                barrido en R_c con varios puntos de partida; escribe un JSON
        entropias.py              S_r y S_p de los orbitales de la malla (transformada numérica)
        base_sto.py               mejor energía de la base de la tesis, evaluada en la malla
        et_basis.py               bases grandes con exponentes en progresión geométrica (Tabla 4.3)
        estados_excitados.py      valencia con l arbitraria (1s² 3d, 1s² 4f, ...)
        uhf.py                    Hartree-Fock no restringido (prueba citada en las perspectivas)
datos/
    litio_V0_5.csv, litio_V0_1.csv        resultados finales con la base de STO
    malla/barrido_V0_5.json, barrido_V0_1.json   resultados de la malla (límite de Hartree-Fock)
    sin_pulir_y_logs/                     barridos antes del pulido y bitácoras de las corridas
figuras/                          las figuras de la tesis (PNG), tal como las genera figuras_tesis.py
docs/
    diagrama_flujo.pdf / .png / .tex      diagrama de flujo de los dos programas
```

---

## 3. Instalación

Se necesita Python 3.9 o posterior con NumPy, SciPy y Matplotlib:

```bash
git clone <URL del repositorio>
cd <carpeta del repositorio>
python -m pip install -r requirements.txt
```

No hay nada que compilar ni instalar aparte. El código se probó con Python 3.13, NumPy 2.5, SciPy 1.18 y Matplotlib 3.11; también funciona con NumPy 1.x.

---

## 4. Inicio rápido

Todos los comandos se dan desde la carpeta `codigo/`.

```bash
cd codigo

# 1. Comprobar que el programa funciona (unos segundos)
python pruebas.py

# 2. Volver a hacer las figuras de la tesis a partir de los datos guardados (~30 s)
python figuras_tesis.py

# 3. Un cálculo pequeño: tres radios con V0 = 5, sin búsqueda global (~1 min)
python litio_confinado_hf.py --V0 5 --Rc 6 4 3 --de-maxiter 0 --salida prueba

# 4. El átomo libre (Tabla 4.1; ~40 s sin búsqueda global)
python litio_confinado_hf.py --libre --de-maxiter 0

# 5. Un punto en la malla (límite de Hartree-Fock; ~10 s)
python malla/hf_litio_confinado.py --Rc 2 --V0 5
```

El paso 3 escribe `prueba_V0_5.csv` y cuatro figuras de control (`prueba_V0_5_*.png`).

---

## 5. Programa con base de STO

Archivo: `codigo/litio_confinado_hf.py`.

### 5.1 Qué hace, paso a paso

El diagrama completo está en `docs/diagrama_flujo.pdf`. En resumen:

1. **Base** (`Basis`). Diez funciones $\phi_\mu = N_\mu r^{n-1} e^{-\zeta_\mu r} Y_{lm}$: cinco de tipo $s$ con $n = 1$ y cinco de tipo $p$ con $n = 2$. Los exponentes $\zeta_\mu$ son los parámetros que se optimizan.
2. **Integrales de un electrón** (`one_electron_matrices`). Traslape $\mathbf S$, energía cinética $\mathbf T$, atracción nuclear $\mathbf V^{nuc}$ y confinamiento $\mathbf V^{conf}$; esta última es una función Gamma incompleta. Todas tienen fórmula cerrada (Apéndice B de la tesis).
3. **Integrales de dos electrones** (`radial_eri`). Integrales radiales de Slater $R^k$ en forma cerrada; la parte angular son símbolos 3-$j$ con proyecciones nulas (`three_j_000`, Apéndice A).
4. **Ciclo autoconsistente** (`scf_eav`). Parte de diagonalizar $\mathbf H^{core}$. Para $1s^2 2p$ se diagonaliza $\mathbf F^{(1s)}$ en el bloque $s$ y $\mathbf F^{(nl)}$ en el bloque $p$. Para $1s^2 2s$ los dos orbitales están en el mismo bloque y se usa el operador efectivo de Guest y Saunders. Termina cuando $|\Delta E| < 10^{-10}$ Hartree y el cambio RMS de la matriz de densidad es menor que $10^{-8}$.
5. **Optimización de los exponentes** (`StateOptimizer`, `_best_of`). Para cada radio se conserva la menor energía de tres estrategias, todas terminadas con SLSQP:
   - partir de los exponentes óptimos del radio anterior (cascada);
   - Evolución Diferencial con esos exponentes dentro de la población inicial;
   - para $1s^2 2s$, varios arranques con exponentes en progresión geométrica.

   Hay dos restricciones: $\zeta_i - \zeta_{i+1} \ge 0.4$ y que el menor valor propio de $\mathbf S$ en cada bloque sea $\ge 10^{-7}$.
6. **Barrido** (`run_sweep`). Recorre los radios de mayor a menor y usa los exponentes de cada radio como punto de partida del siguiente.
7. **Pulido** (`polish`). Al terminar, cada radio se vuelve a optimizar partiendo de los exponentes de sus vecinos y se conserva el resultado si la energía baja.
8. **Propiedades** (`build_row`). $\langle r\rangle$ y $P_{out}$ con integrales analíticas; presión de Hellmann-Feynman con la densidad total en la pared; entropías $S_r$ y $S_p$ (`shannon_entropies`); la presión por diferencias finitas se agrega al final (`add_fd_pressure`).

Detalles numéricos que importan:

- El problema generalizado de valores propios se resuelve con $\mathbf S + \eta\mathbf I$, $\eta = 10^{-8}$, solo para estabilizar la diagonalización. Los orbitales ocupados **se normalizan después con la $\mathbf S$ verdadera** (`_renorm`); sin este paso la energía puede quedar por debajo del valor variacional.
- La tolerancia del SCF es $10^{-10}$ porque SLSQP deriva la energía numéricamente respecto a los exponentes; con $10^{-6}$ el ruido arruina esas derivadas.
- La transformada de Fourier de los STO es analítica (`Basis.momentum`), así que $S_p$ solo tiene el error de la regla del trapecio en una malla logarítmica de 8001 puntos.

### 5.2 Opciones de la línea de comandos

```
python litio_confinado_hf.py [opciones]
```

| Opción | Valor por omisión | Qué hace |
|---|---|---|
| `--V0 X` | `5.0` | Altura de la barrera (Hartree). |
| `--Rc R1 R2 ...` | 23 radios de 15 a 0.8 | Radios de la cavidad (u.a.). Se recorren de mayor a menor. |
| `--salida PREFIJO` | `litio` | Prefijo de los archivos de salida: `PREFIJO_V0_<V0>.csv` y figuras de control. |
| `--de-maxiter N` | `40` | Generaciones de Evolución Diferencial por radio. Con `0` solo se usa cascada + SLSQP (mucho más rápido, pero puede quedar en un mínimo local). |
| `--pulir CSV` | — | No hace barrido: pule un CSV existente con los exponentes de los radios vecinos y lo sobrescribe. La altura $V_0$ se lee del CSV. |
| `--semillas CSV ...` | — | Con `--pulir`: otros CSV cuyos exponentes también se prueban como puntos de partida. |
| `--recalcular CSV` | — | No optimiza: recalcula todas las propiedades con los exponentes guardados en el CSV y lo sobrescribe. Sirve para agregar columnas nuevas. |
| `--libre` | — | Átomo libre ($V_0 = 0$): optimiza los exponentes de las dos configuraciones e imprime los valores de la Tabla 4.1. |
| `--prueba-momentos` | — | Pruebas del cálculo en el espacio de momentos (Tabla E.1). |
| `--polarizabilidad` | — | Calcula además la polarizabilidad dipolar por campo finito. Es experimental y **no se usa en la tesis**. |

### 5.3 Cómo se obtuvieron los datos de la tesis

```bash
cd codigo
python litio_confinado_hf.py --V0 5 --salida ../datos/litio      # escribe ../datos/litio_V0_5.csv
python litio_confinado_hf.py --V0 1 --salida ../datos/litio      # escribe ../datos/litio_V0_1.csv
python litio_confinado_hf.py --pulir ../datos/litio_V0_5.csv
python litio_confinado_hf.py --pulir ../datos/litio_V0_1.csv
```

Tiempos aproximados en una computadora de escritorio: 1.5 a 3 minutos por radio en el barrido (unos 45 minutos por barrera) y unos 15 minutos por barrera en el pulido. Las bitácoras de las corridas originales están en `datos/sin_pulir_y_logs/`.

La Evolución Diferencial usa una semilla fija (`seed=0`), así que el barrido es reproducible en una misma versión de SciPy. Con otra versión los exponentes pueden cambiar un poco, pero las energías deben coincidir en $10^{-5}$ Hartree o mejor: la función $E(\boldsymbol\zeta)$ tiene varios mínimos casi con la misma energía.

---

## 6. Programa en la malla radial

Carpeta: `codigo/malla/`. Los scripts se pueden correr desde cualquier carpeta.

### 6.1 Qué hace

Escribe cada orbital como $\psi = \frac{u(r)}{r}Y_{lm}$ y resuelve las ecuaciones radiales de Hartree-Fock (las mismas del programa de STO) directamente en puntos:

- **Malla logarítmica** $x = \ln r$ con paso $h = 0.02$ y $u = r^{1/2}y(x)$; $r_{min} = 10^{-10}$, $r_{max} = \max(50, R_c + 40)$. El radio $R_c$ cae exactamente en un punto.
- **Segunda derivada** con la fórmula de cinco puntos (error $h^4$).
- **Integrales** con la regla del trapecio en $x$, más la corrección de Euler-Maclaurin en los dos lugares donde el integrando tiene un quiebre: en $r' = r$ dentro de los potenciales de Hartree y en la pared.
- **Cada ecuación** se resuelve por iteración inversa. Para $1s^2 2s$ la ortogonalidad entre los orbitales se impone con un sistema bordeado, y en cada ciclo se optimiza el ángulo de rotación $1s$–$2s$.
- Como las ecuaciones no son lineales tienen varias soluciones; `barrido.py` prueba varios puntos de partida en cada radio y conserva la de menor energía con el número de nodos correcto.

El único error es el de discretización: al pasar de $h = 0.02$ a $h = 0.01$ la energía cambia como mucho $7\times10^{-6}$ Hartree.

### 6.2 Scripts y uso

| Script | Uso | Qué hace |
|---|---|---|
| `hf_litio_confinado.py` | `python hf_litio_confinado.py` | Sin argumentos corre las pruebas de verificación (Tabla D.1; ~3 min). |
| | `python hf_litio_confinado.py --Rc 2 --V0 5 [--h 0.01]` | Un radio con escalón finito: imprime $E_{av}$, energías orbitales, $P_{out}$ y $\langle r\rangle$. |
| | `python hf_litio_confinado.py --Rc 2` | Sin `--V0`: pared impenetrable. |
| `barrido.py` | `python barrido.py 5 0.02 ../../datos/malla/barrido_V0_5.json` | Barrido completo (24 radios, las dos configuraciones, ida y vuelta). Argumentos: $V_0$, $h$ y archivo de salida. Tarda cerca de una hora. |
| `entropias.py` | `python entropias.py` | Pruebas de la transformada numérica al espacio de momentos con orbitales STO de transformada conocida. `barrido.py` lo usa para guardar $S_r$ y $S_p$. |
| `base_sto.py` | `python base_sto.py 2s 5 2 0.02` | Mejor energía posible con la base de la tesis, con las integrales hechas en la malla. Argumentos: configuración (`2s` o `2p`), $V_0$, $R_c$ ($\le 0$ = átomo libre) y $h$. Es lento: cerca de una hora por caso (61 min para `2s 5 2 0.02`, que da $E = -5.92996$). |
| `et_basis.py` | `python et_basis.py 5 2 0.02` | Bases de 10, 20 y 30 STO con exponentes en progresión geométrica entre 0.08 y 200 (de las 30, 27 son linealmente independientes). Argumentos: $V_0$, $R_c$ y $h$. |
| `estados_excitados.py` | `python estados_excitados.py 2 5 2` | Configuración $1s^2 nl$ con $l$ arbitraria. Argumentos: $l$, $V_0$ y $R_c$ ($\le 0$ = átomo libre). |
| `uhf.py` | `python uhf.py 5 2` | Hartree-Fock no restringido ($1s\alpha \ne 1s\beta$). Argumentos: $V_0$ y $R_c$ (`0 0` = átomo libre). |

---

## 7. Formato de los datos

### 7.1 `datos/litio_V0_*.csv` (base de STO)

Una fila por radio. El sufijo `_2s` se refiere a la configuración $1s^2 2s$ y `_2p` a $1s^2 2p$; `val` es el orbital de valencia ($2s$ o $2p$).

| Columna | Significado |
|---|---|
| `Rc`, `V0` | Radio de la cavidad y altura de la barrera. |
| `E_2s`, `E_2p` | Energía media de configuración $E_{av}$ (Hartree). |
| `eps1s_2s`, `eps2s`, `eps1s_2p`, `eps2p` | Energías orbitales. |
| `conv_2s`, `conv_2p` | `True` si el ciclo SCF convergió. |
| `Pin_1s_*`, `Pout_1s_*`, `Pin_val_*`, `Pout_val_*` | Probabilidad de encontrar al electrón dentro y fuera de la cavidad. |
| `Pout_tot_*` | Fracción de la carga total fuera de la cavidad, $(2P_{out}^{1s} + P_{out}^{val})/3$. |
| `r_1s_*`, `r_val_*` | Radio medio $\langle r\rangle$ (u.a.). |
| `P_HF_*` | Presión de Hellmann-Feynman, $V_0\,\rho(R_c)$ con la densidad total (Hartree/bohr³; 1 u.a. = 29 421 GPa). |
| `P_HF_val_*` | La misma presión contando solo al electrón de valencia. |
| `P_FD_*` | Presión por diferencias finitas de $E(R_c)$. |
| `S_1s_*`, `S_val_*` | Entropía de Shannon en posiciones, $S_r$, del orbital $1s$ y del de valencia (el $2p$ promediado en $m$). |
| `S_val_2p_pz` | $S_r$ del orbital $2p_z$ ($m = 0$). |
| `Sp_1s_*`, `Sp_val_*`, `Sp_val_2p_pz` | Entropía de Shannon en momentos, $S_p$. |
| `Sr_tot_*`, `Sp_tot_*` | $S_r$ y $S_p$ de la densidad total de la configuración, normalizada a uno. |
| `zeta_2s` | Los cinco exponentes óptimos del bloque $s$ en $1s^2 2s$ (separados por espacios, de mayor a menor). |
| `zeta_2p` | Los diez exponentes óptimos en $1s^2 2p$: primero los cinco $s$ y luego los cinco $p$. |

Con `zeta_2s` y `zeta_2p` se puede reconstruir cualquier orbital sin volver a optimizar (ver la sección 10).

### 7.2 `datos/malla/barrido_V0_*.json` (malla)

Estructura: `{"2s": {"<Rc>": {...}, ...}, "2p": {...}}`. Para cada radio:

| Clave | Significado |
|---|---|
| `E` | Energía media de configuración (límite de Hartree-Fock). |
| `e1s`, `enl` | Energías orbitales del $1s$ y del orbital de valencia. |
| `pout1s`, `poutnl` | Probabilidad fuera de la cavidad. |
| `r1s`, `rnl` | Radio medio. |
| `rho_Rc` | Densidad electrónica total en la pared; la presión es $V_0 \times$ `rho_Rc`. |
| `Sr_1s`, `Sp_1s`, `Sr_nl`, `Sp_nl`, `Sr_tot`, `Sp_tot` | Entropías de Shannon en posiciones y momentos del $1s$, del orbital de valencia y de la densidad total. |
| `normp_1s`, `normp_nl`, `normp_tot` | Norma de la densidad de momentos (control numérico; debe ser 1). |

---

## 8. Cómo reproducir cada tabla y figura de la tesis

Desde `codigo/`. Los scripts de tablas y figuras leen por omisión los archivos de `datos/`.

| Resultado de la tesis | Comando | Notas |
|---|---|---|
| Tabla 4.1 (átomo libre) | `python litio_confinado_hf.py --libre` | El límite de Hartree-Fock de la tabla sale de `python malla/hf_litio_confinado.py`. |
| Tabla 4.2 (malla / STO / Özmen) | `python tabla_ozmen.py` | Imprime la tabla en LaTeX y las diferencias con el límite. |
| Tabla 4.3 (5 STO / 27 STO / malla) | `python malla/et_basis.py V0 Rc 0.02` | Un comando por cada pareja $(V_0, R_c)$; la columna «27 STO» es la línea `N=30`. |
| Tabla 4.4 (continuidad en la pared) | `python tabla_continuidad.py` | |
| Tabla 4.5 (entropías) | `python tabla_entropias.py` | |
| Tabla 4.6 (entropías, STO contra malla) | `python comparar_entropias_malla.py` | |
| Tabla D.1 (verificación de la malla) | `python malla/hf_litio_confinado.py` | |
| Tabla E.1 (entropías exactas del $1s$) | `python litio_confinado_hf.py --prueba-momentos` | |
| Figuras 4.1 a 4.10 | `python figuras_tesis.py` | Escribe los PNG en `figuras/` con los nombres que usa `final.tex`. |
| Prueba de la base en la malla (Sec. 4.2) | `python malla/base_sto.py 2s 5 2 0.02` | Cambiando configuración, $V_0$ y $R_c$. |
| Hartree-Fock no restringido (Cap. 5) | `python malla/uhf.py 0 0` y `python malla/uhf.py 5 2` | Da $-2.4\times10^{-5}$ y $-1.5\times10^{-3}$ Hartree. |

Correspondencia entre figuras y archivos:

| Figura | Archivo | Contenido |
|---|---|---|
| 4.1 | `fig_1_energia_total.png` | Energía del estado base y valores de Özmen *et al.* |
| 4.2 | `fig_6_error_base.png` | Diferencia con el límite de Hartree-Fock |
| 4.3 | `fig_2_energias_orbitales.png` | $\varepsilon_{1s}$ y $\varepsilon_{2s}$ |
| 4.4 | `fig_3_radio_y_pout.png` | $\langle r\rangle$ y $P_{out}$ |
| 4.5 | `fig_4_optimizacion_zetas.png` | Exponentes óptimos del bloque $s$ |
| 4.6 | `fig_5_presion_cuantica.png` | Presión por los dos métodos |
| 4.7 | `fig_1_energia_total_2s_2p.png` | $1s^2 2s$, $1s^2 2p$ y energía de excitación |
| 4.8 | `nubes_electronicas_Rc_10.0.png`, `..._Rc_1.0.png` | Mapas de densidad del electrón de valencia |
| 4.9 | `fig_2_entropia_shannon.png` | $S_r$ de los orbitales |
| 4.10 | `fig_7_entropia_momentos.png` | $S_p$, $S_r + S_p$ y la cota |

---

## 9. Pruebas

```bash
cd codigo
python pruebas.py                         # programa de STO (segundos)
python malla/hf_litio_confinado.py        # programa de la malla (~3 min)
python malla/entropias.py                 # transformada numérica en la malla
```

`pruebas.py` compara con resultados exactos o independientes y termina con código 0 si todo pasa:

1. símbolos 3-$j$: $(0\,1\,1;000) = -1/\sqrt3$, $(1\,1\,2;000) = \sqrt{2/15}$, y los que deben anularse;
2. integrales de un STO $1s$: $T = \zeta^2/2$, $V^{nuc} = -Z\zeta$, y $V^{conf}$ contra integración numérica;
3. integrales de dos electrones contra las fórmulas hidrogenoides: $F^0(1s,1s) = \tfrac58\zeta$, $F^0(1s,2p)$ y $G^1(1s,2p)$;
4. el ion Li²⁺ ($E = -Z^2/2$) y el ion Li⁺ con un solo STO ($E = -(Z - 5/16)^2$);
5. en el espacio de momentos: las entropías exactas del orbital $1s$ hidrogenoide, $S_r + S_p = \ln(32\pi^3) - \tfrac13$, la transformada analítica contra cuadratura y el teorema de Plancherel;
6. que los CSV de `datos/` se reproducen con los exponentes que guardan y que la desigualdad $S_r + S_p \ge 3(1+\ln\pi)$ se cumple en todos los radios.

Las pruebas de la malla comparan con un caso exacto (hidrógeno con pared impenetrable en $R_c = 2$, $E = -1/8$), con el método de disparo para un electrón y con el átomo de litio libre ($E = -7.432727$ Hartree, teorema del virial).

---

## 10. Usar el código desde Python

Los programas también se pueden importar. Por ejemplo, para reconstruir los orbitales de un radio a partir de los exponentes guardados y calcular sus propiedades (desde `codigo/`):

```python
import numpy as np
import litio_confinado_hf as hf

filas = hf.load_csv("../datos/litio_V0_5.csv")
fila = next(f for f in filas if f["Rc"] == 2.0)

zeta_s = [float(z) for z in fila["zeta_2s"].split()]
base = hf.Basis(zeta_s, [3.0, 2.0, 1.2, 0.7, 0.3])     # el bloque p no interviene en 1s² 2s
res = hf.scf_eav(base, Rc=2.0, V0=5.0, config="2s")

print(res["E"], res["eps1"], res["eps2"])              # energía y energías orbitales
r = np.linspace(0.01, 6, 600)
R_2s = base.radial(r) @ res["c2"]                      # parte radial del orbital 2s
F_2s = base.momentum(np.array([0.5, 1.0])) @ res["c2"] # parte radial en el espacio de momentos
print(hf.orbital_observables(base, res["c2"], 2.0))    # P_in, P_out, <r>, S_radial
print(hf.shannon_entropies(base, res["c1"], res["c2"]))
```

Funciones principales de `litio_confinado_hf.py`:

| Función o clase | Para qué sirve |
|---|---|
| `Basis(zeta_s, zeta_p)` | Define la base. `radial(r)` y `momentum(p)` evalúan las funciones en posiciones y en momentos. |
| `one_electron_matrices(B, Rc, V0)` | Devuelve $\mathbf S$, $\mathbf T$, $\mathbf V^{nuc}$, $\mathbf V^{conf}$ y la matriz del dipolo. |
| `radial_eri(B, kmax)` | Integrales radiales $R^k(\mu\nu;\lambda\sigma)$. |
| `scf_eav(B, Rc, V0, config)` | Ciclo SCF. Devuelve un diccionario con `E`, los coeficientes `c1` ($1s$) y `c2` (valencia), `eps1`, `eps2` y `converged`. |
| `StateOptimizer(config, V0)` | Optimización de exponentes: `energy(x, Rc)`, `local_refine(Rc, x0)`, `global_search(Rc, semilla)`. |
| `run_sweep(V0, lista_Rc)` y `polish(filas, V0)` | Barrido y pulido. |
| `free_atom()` | Átomo libre. |
| `orbital_observables`, `shannon_entropies`, `check_bbm` | Propiedades de los orbitales y verificación de la desigualdad entrópica. |

En la malla, `hf_litio_confinado.py` define la clase `Malla(Z, Rc, V0, h)` y las funciones `scf(m)` y `scf_2p(m)`, que devuelven los orbitales en la malla, la energía y el número de ciclos.

---

## 11. Limitaciones conocidas

- **Error de la base con la barrera alta.** Las funciones de la base son suaves en todo el espacio y no pueden reproducir el salto que la pared produce en la segunda derivada de la función de onda, $[u''] = 2V_0\,u(R_c)$. Con $V_0 = 1$ el error respecto al límite de Hartree-Fock es de 1 a 9 mHartree, pero con $V_0 = 5$ llega a 0.1 Hartree para $1s^2 2s$ (en $R_c = 2$ u.a.). Para energías de referencia hay que usar la malla.
- **Solo el litio en $1s^2 nl$.** El programa de STO está escrito para tres electrones en las configuraciones $1s^2 2s$ y $1s^2 2p$; no es un programa general de Hartree-Fock. El bloque $p$ solo tiene funciones con $n = 2$.
- **Sin correlación electrónica.** Es Hartree-Fock: en el átomo libre faltan unos 45 mHartree respecto a la energía exacta no relativista.
- **Núcleo fijo en el centro** de la cavidad y potencial esférico.
- **Varios mínimos en los exponentes.** Las curvas de los exponentes óptimos contra $R_c$ no son suaves; las cantidades físicas sí lo son.

---

## 12. Referencias

- A. Özmen, B. Çakır, C. Demir y Y. Yakar, «Investigation of electronic structure of a lithium atom confined by a finite spherical cavity», *Physica B* **656**, 414775 (2023). Modelo y valores de comparación.
- R. D. Cowan, *The Theory of Atomic Structure and Spectra*, University of California Press (1981). Energía media de configuración (cap. 6), símbolos 3-$j$ (cap. 5), ecuaciones radiales (cap. 7).
- A. Szabo y N. S. Ostlund, *Modern Quantum Chemistry*, Dover (1996). Método de Hartree-Fock y ecuaciones de Roothaan-Hall.
- M. F. Guest y V. R. Saunders, *Mol. Phys.* **28**, 819 (1974). Operador efectivo de capa abierta.
- C. Froese Fischer, *The Hartree-Fock Method for Atoms*, Wiley (1977). Hartree-Fock numérico.
- I. Białynicki-Birula y J. Mycielski, *Commun. Math. Phys.* **44**, 129 (1975). Desigualdad entrópica.
- C. Martínez-Flores y R. Cabrera-Trujillo, *Matter Radiat. Extremes* **5**, 024401 (2020). Litio en una cavidad impenetrable.

La lista completa está en la bibliografía de la tesis.

---

## Autor

Andrés Valenzuela Leyva — Licenciatura en Física, Universidad de Sonora.

Si usas este código, cita la tesis:

> A. Valenzuela Leyva, *Estudio de la Estructura Electrónica y Propiedades del Átomo de Litio Confinado mediante el Método de Hartree-Fock*, Tesis de Licenciatura en Física, Universidad de Sonora, Hermosillo, México (2026).
