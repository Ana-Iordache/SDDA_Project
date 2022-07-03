import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
import matplotlib.pyplot as plt


def nan_replace(t):
    assert isinstance(t, pd.DataFrame)
    nume_variabile = list(t.columns)
    for v in nume_variabile:
        if any(t[v].isna()):
            if is_numeric_dtype(t[v]):
                t[v].fillna(t[v].mean(), inplace=True)
            else:
                modulul = t[v].mode()[0]
                t[v].fillna(modulul, inplace=True)


def tabelare_varianta(alpha, etichete):
    procent_varianta = alpha * 100 / sum(alpha)
    tabel_varianta = pd.DataFrame(data={
        "Varianta": alpha,
        "Varianta cumulata": np.cumsum(alpha),
        "Procent varianta": procent_varianta,
        "Procent cumulat": np.cumsum(procent_varianta)},
        index=etichete
    )
    return tabel_varianta


def tabelare_matrice(x, nume_linii=None, nume_coloane=None, out=None):
    t = pd.DataFrame(x, nume_linii, nume_coloane)
    if out is not None:
        t.to_csv(out)
    return t


def harta(shp, camp_legatura, t, titlu="Harta scoruri"):
    variabile_harta = list(t)
    shp1 = pd.merge(shp, t, left_on=camp_legatura, right_index=True)
    for v in variabile_harta:
        f = plt.figure(titlu + "-" + v, figsize=(10, 7))
        ax = f.add_subplot(1, 1, 1)
        ax.set_title(titlu + "-" + v)
        shp1.plot(v, cmap="Reds", ax=ax, legend=True)
