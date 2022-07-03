# Proiect DSAD Iordache Ana-Maria, grupa 1086
import factor_analyzer as fact
import pandas as pd
import functii as f
import numpy as np
import grafice as g
from geopandas import GeoDataFrame

tabel = pd.read_csv("Miscarea_naturala_a_populatiei.csv", index_col=1)

f.nan_replace(tabel)

variabile_observate = list(tabel)[1:]

valori = tabel[variabile_observate].values
n, m = np.shape(valori)

# Testul Bartlett
bartlett_test = fact.calculate_bartlett_sphericity(valori)
# Validare model
if bartlett_test[1] > 0.001:
    print("Nu exista factori!")
    exit(0)
print(bartlett_test)

# Indecsi KMO
kmo = fact.calculate_kmo(valori)

kmo_t = pd.DataFrame(
    data={
        "Index KMO": np.append(kmo[0], kmo[1])
    }, index=variabile_observate + ["Total"]
)
# Corelograma indecsi KMO
g.corelograma(kmo_t, vmin=0, titlu="Corelograma indecsi KMO")
kmo_t.to_csv("Indecsi_KMO.csv")

if all(kmo_t["Index KMO"] < 0.6):
    print("Nu exista factori!")
    exit(0)

# Construire model FARA ROTATIE
rotatie = ""
model_fact = fact.FactorAnalyzer(n_factors=m, rotation=None)
model_fact.fit(valori)

# Varianta factori
alpha = model_fact.get_factor_variance()[0]
etichete_factori = ["F" + str(i + 1) for i in range(m)]
tabel_varianta = f.tabelare_varianta(alpha, etichete_factori)
tabel_varianta.to_csv("Varianta_F_" + rotatie + ".csv")

# Corelatii intre variabilele observate si factori
corelatii = model_fact.loadings_
corelatii_tabel = f.tabelare_matrice(corelatii, variabile_observate, etichete_factori, "Corelatii_" + rotatie + ".csv")
# Corelograma corelatii factoriale
g.corelograma(corelatii_tabel)
# Plot corelatii factoriale
g.plot_componente(corelatii_tabel, "F1", "F2", "Plot corelatii factoriale", aspect=1)

# Scoruri factoriale
scoruri = model_fact.transform(valori)
scoruri_tabel = f.tabelare_matrice(scoruri, tabel.index, etichete_factori, "ScoruriFactoriale_" + rotatie + ".csv")
# Plot scoruri factoriale
g.plot_componente(scoruri_tabel, "F1", "F2", "Plot scoruri factoriale", aspect=1)

# Harta scoruri factoriale
shp = GeoDataFrame.from_file("RO_NUTS2/Ro.shp")
g.harta(shp, scoruri[:, :4], "sj", tabel.index)

# Construire model CU ROTATIE VARIMAX
rotatie = "varimax"
model_fact = fact.FactorAnalyzer(n_factors=m, rotation=rotatie)
model_fact.fit(valori)

# Varianta factori
alpha = model_fact.get_factor_variance()[0]
etichete_factori = ["F" + str(i + 1) for i in range(m)]
tabel_varianta = f.tabelare_varianta(alpha, etichete_factori)
tabel_varianta.to_csv("Varianta_F_" + rotatie + ".csv")

# Corelatii intre variabilele observate si factori
corelatii = model_fact.loadings_
corelatii_tabel = f.tabelare_matrice(corelatii, variabile_observate, etichete_factori, "Corelatii_" + rotatie + ".csv")
# Corelograma corelatii factori
g.corelograma(corelatii_tabel, titlu="Corelograma corelatii factoriale - rotatie " + rotatie)
# Plot corelatii factoriale
g.plot_componente(corelatii_tabel, "F1", "F2", "Plot corelatii factoriale - rotatie " + rotatie, aspect=1)

# Scoruri factoriale
scoruri = model_fact.transform(valori)
scoruri_tabel = f.tabelare_matrice(scoruri, tabel.index, etichete_factori, "ScoruriFactoriale_" + rotatie + ".csv")
# Plot scoruri factoriale
g.plot_componente(scoruri_tabel, "F1", "F2", "Plot scoruri factoriale - rotatie " + rotatie, aspect=1)

# Comunalitati
comunalitati = model_fact.get_communalities()
comunalitati_tabel = pd.DataFrame(
    data={"Comunalitati": comunalitati}, index=variabile_observate
)
# Corelograma comunalitati
g.corelograma(comunalitati_tabel, vmin=0, titlu="Comunalitati")
comunalitati_tabel.to_csv("Comunalitati.csv")

g.show()
