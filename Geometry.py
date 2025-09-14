import pandas as pd
import geopandas as gpd
from shapely import wkt
from urllib.parse import quote

# ==========================
# 1. Cargar datos y población
# ==========================
def cargar_datos():
    URL = "https://raw.githubusercontent.com/JuancarRG1/TFM/main/"
    Archivos = [f"Datos_España_{i}.csv" for i in range(1, 9)]
    Datos = []
    for Archivo in Archivos:
        URL_f = URL + quote(Archivo)
        Data = pd.read_csv(URL_f, encoding="utf-8")
        Datos.append(Data)
    return pd.concat(Datos, ignore_index=True)

def muertes_por_100k(Datos_Totales):
    Population = pd.read_csv(
        "https://raw.githubusercontent.com/JuancarRG1/TFM/main/Population.csv",
        sep=";", encoding="latin1"
    )
    Datos_Totales["Periodo"] = Datos_Totales["Periodo"].astype(int)
    Population["Periodo"] = Population["Periodo"].astype(int)

    Datos = pd.merge(
        Datos_Totales, Population, how="left",
        on=["Sexo", "Edad", "Periodo", "Comunidades y Ciudades Autónomas"],
        suffixes=('', '_Population')
    )

    Datos = Datos.dropna(subset=["Total_Population"])
    Datos["Total"] = Datos["Total"].astype(float)
    Datos["Total_Population"] = Datos["Total_Population"].astype(float)
    Datos["Muertes por 100000 habitantes"] = (
        Datos["Total"] / Datos["Total_Population"] * 100000
    )

    return Datos


# ==========================
# 2. Cargar geometría y merge
# ==========================
def Geometry():
    # Cargar datos y métrica
    datos_totales = cargar_datos()
    datos_metricas = muertes_por_100k(datos_totales)

    # Filtrar "Nacional"
    datos_metricas = datos_metricas[
        datos_metricas["Comunidades y Ciudades Autónomas"] != "Nacional"
    ]

    # Cargar geometría base
    url_geo = "https://raw.githubusercontent.com/codeforgermany/click_that_hood/main/public/data/spain-communities.geojson"
    gdf_geo = gpd.read_file(url_geo)

    # Normalizar nombres
    gdf_geo['name'] = gdf_geo['name'].replace({
        "Baleares": "04 Balears, Illes",
        "Pais Vasco": "16 País Vasco",
        "Valencia": "10 Comunitat Valenciana",
        "Madrid": "13 Madrid, Comunidad de",
        "Murcia": "14 Murcia, Región de",
        "La Rioja": "17 Rioja, La",
        "Canary Islands": "05 Canarias",
        "Castilla-Leon": "07 Castilla y León",
        "Castilla-La Mancha": "08 Castilla-La Mancha",
        "Asturias": "03 Asturias, Principado de",
        "Navarra": "15 Navarra, Comunidad Foral de",
        "Extremadura": "11 Extremadura",
        "Catalonia": "09 Cataluña",
        "Andalucia": "01 Andalucía",
        "Galicia": "12 Galicia",
        "Cantabria": "06 Cantabria",
        "Aragón": "02 Aragón",
        "Ceuta": "18 Ceuta",
        "Melilla": "19 Melilla"
    })

    # Simplificar geometrías
    gdf_geo["geometry"] = gdf_geo["geometry"].simplify(tolerance=0.01, preserve_topology=True)

    # Merge para añadir geometría
    df_merged = datos_metricas.merge(
        gdf_geo[["name", "geometry"]],
        left_on="Comunidades y Ciudades Autónomas",
        right_on="name",
        how="left"
    ).drop(columns=["name"])

    # Convertir geometría a WKT para guardar en CSV
    df_merged["geometry"] = df_merged["geometry"].apply(lambda x: x.wkt if x else None)

    # Guardar CSV listo para usar en Streamlit
    df_merged.to_csv("Geometry.csv", index=False, encoding="utf-8")

    return df_merged

if __name__ == "__main__":
    Geometry()
    print("Archivo 'Geometry.csv' generado con éxito.")