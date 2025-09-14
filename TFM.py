# CÓDIGO DE LA HERRAMIENTA DE ANÁLISIS DE MORTALIDAD EN ESPAÑA

# Se importan los paquetes necesarios.

import pandas as pd
import numpy as np
from urllib.parse import quote
import streamlit as st
import altair as alt
import plotly.express as px
# import geopandas as gpd
# from datetime import datetime
# import json
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# Se hace que se ocupe la página entera.

st.set_page_config(layout="wide")

st.markdown(
    """
    <style>
    .block-container {
        padding-left: 0.5rem;
        padding-right: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Se establecen las URLs de las que se recuperan los datos.

# Se crea una función con resultados que se guardan en la memoria caché.

@st.cache_data
def cargar_datos():
    URL = "https://raw.githubusercontent.com/JuancarRG1/TFM/main/"
    Archivos = [f"Datos_España_{i}.csv" for i in range(1, 9)] # Se leen por el nombre que les he dado.
    Datos = []

    for Archivo in Archivos:
        URL_f = URL + quote(Archivo) # Se toma cada URL de las que se desea exportar.
        Data = pd.read_csv(URL_f, encoding="utf-8") # Se lee la URL como un CSV.
        Datos.append(Data) # Se combinan todos los conjuntos de datos.

    return pd.concat(Datos, ignore_index=True)

# Se cargan los datos.

Datos_Totales = cargar_datos()

   


# Se convierte el número de muertos a muertes por 100.000 habitantes.

@st.cache_data
def Muertes_100000_Habitantes(Datos_Totales):

    Population = pd.read_csv("https://raw.githubusercontent.com/JuancarRG1/TFM/main/Population.csv", sep = ";", encoding = "latin1")    
    Datos_Totales["Periodo"] = Datos_Totales["Periodo"].astype(int)
    Population["Periodo"] = Population["Periodo"].astype(int)

    Datos = pd.merge(Datos_Totales,Population,how="left",
        on=["Sexo", "Edad", "Periodo", "Comunidades y Ciudades Autónomas"],
        suffixes=('', '_Population'))
    
    Datos = Datos.dropna(subset=["Total_Population"])
    Datos["Total"] = Datos["Total"].astype(float)
    Datos["Total_Population"] = Datos["Total_Population"].astype(float)
    Datos["Muertes por 100000 habitantes"] = (Datos["Total"] / Datos["Total_Population"].replace(0, np.nan)) * 100000

    
    return Datos

Datos = Muertes_100000_Habitantes(Datos_Totales)


# Se comienza a crear la aplicación.

# Se establece un título.

st.title("Causas de mortalidad en España")

# Se configura el filtro por causa de muerte.

st.subheader("Selecciona una o varias causas de muerte")
Causa = Datos['Causa de muerte'].unique()
Causa_Filtro = st.multiselect("Causa(s) de muerte:", Causa)

# Se configura el filtro por edad.

st.subheader("Selecciona un intervalo de edades")
Edades = Datos['Edad'].unique()
Edades_Filtro = st.multiselect("Selecciona intervalos de edad:", sorted(Edades))

# Se configura el filtro por comunidad autónoma.

st.subheader("Selecciona las comunidades autónomas")
Comunidades = Datos['Comunidades y Ciudades Autónomas'].unique()
Comunidades_Filtro = st.multiselect("Selecciona comunidades:", sorted(Comunidades))

# Se configura el filtro por sexo.

st.subheader("Selecciona el sexo")
Sexo = Datos['Sexo'].unique()
Sexo_Filtro = st.multiselect("Selecciona sexo", ["Hombres", "Mujeres", "Total"])

# Se configura el filtro por años.

st.subheader("Selecciona los años")
Min_Año = int(Datos['Periodo'].min())
Max_Año = int(Datos['Periodo'].max())
Rango_Años = st.slider("Selecciona el rango temporal", min_value=Min_Año, max_value=Max_Año, value=(Min_Año, Max_Año))


# Se filtra el dataset.

Datos_Filtro = Datos.copy() # Ya no necesito ver la tabla

if Causa_Filtro:
    Datos_Filtro = Datos_Filtro[Datos_Filtro['Causa de muerte'].isin(Causa_Filtro)]

if Edades_Filtro:
    Datos_Filtro = Datos_Filtro[Datos_Filtro['Edad'].isin(Edades_Filtro)]

if Comunidades_Filtro:
    Datos_Filtro = Datos_Filtro[Datos_Filtro['Comunidades y Ciudades Autónomas'].isin(Comunidades_Filtro)]

if Sexo_Filtro:
    Datos_Filtro = Datos_Filtro[Datos_Filtro['Sexo'].isin(Sexo_Filtro)]
    
if Rango_Años:
    Datos_Filtro = Datos_Filtro[(Datos_Filtro['Periodo'] >= Rango_Años[0]) & (Datos_Filtro['Periodo'] <= Rango_Años[1])]
    
# st.dataframe(Datos_Filtro) # Ya no necesito ver la tabla para comprobaciones

st.subheader("Visualización de las Causas de Muerte")

# Crear pestañas
tab1, tab2, tab3 = st.tabs(["📊 Gráfica de líneas", "📊 Gráfico de áreas", "📊 Gráfico de flechas"])

# Se muestra una gráfica de evolución temporal.

with tab1:
    st.subheader("Evolución temporal de muertes por 100.000 habitantes")

# st.subheader("Evolución temporal de muertes por 100.000 habitantes")

    if not Datos_Filtro.empty:
        Datos_Agrupados = Datos_Filtro.groupby(['Periodo', 'Causa de muerte'], as_index=False)['Muertes por 100000 habitantes'].mean()

        linea = alt.Chart(Datos_Agrupados).mark_line().encode(
            x=alt.X('Periodo:Q', title='Año', axis=alt.Axis(format='.0f')),
            y=alt.Y('Muertes por 100000 habitantes:Q', title='Muertes por 100.000 habitantes'),
            color=alt.Color('Causa de muerte:N'))

        puntos = alt.Chart(Datos_Agrupados).mark_point(size=60).encode(
            x='Periodo:Q',
            y='Muertes por 100000 habitantes:Q',
            color=alt.Color('Causa de muerte:N',
                legend=alt.Legend(
                    title="Causa de muerte",
                    labelLimit=1000,  # Caracteres por línea
                    symbolLimit=500,  # Espacio disponible
                    orient="bottom",
                    columns=1, # La leyenda se escribe en una columna (por lo largos que son los elementos)
                    labelFontSize=10.8,   # Tamaño del texto de las etiquetas
                    titleFontSize=16    # Tamaño del título de la leyenda
                                 )
                           ),
            tooltip=['Periodo', 'Causa de muerte', 'Muertes por 100000 habitantes'])

        grafico = (linea + puntos).properties(
            width=900,
            height=1200
        ).interactive()

        st.altair_chart(grafico, use_container_width=True)
    
    else:
        st.warning("No hay datos para mostrar la gráfica. Revisa los filtros aplicados.")
    
with tab2:
    st.subheader("Distribución de muertes por causa")

    if not Datos_Filtro.empty:
        Treemap_Datos = Datos_Filtro.groupby('Causa de muerte', as_index=False)['Muertes por 100000 habitantes'].mean()

        # Etiqueta solo para mostrar en hover
        Treemap_Datos['Etiqueta'] = Treemap_Datos['Causa de muerte'] + '<br>' + Treemap_Datos['Muertes por 100000 habitantes'].round(0).astype(int).astype(str) + ' por 100.000'

        fig = px.treemap(
            Treemap_Datos,
            path=['Causa de muerte'],
            values='Muertes por 100000 habitantes',
            color='Muertes por 100000 habitantes',
            color_continuous_scale='Reds',
            title="Proporción relativa de muertes por causa"
        )

        # Ocultar info extraña del tooltip y dejarlo claro y limpio
        fig.update_traces(
            hovertemplate='<b>%{label}</b><br>Muertes por 100.000 habitantes: %{value:.0f}<extra></extra>',
            textinfo='label+value',
            textfont_size=12
        )

        fig.update_layout(
            margin=dict(t=50, l=25, r=25, b=25)
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No hay datos suficientes para generar el gráfico de áreas.")
    
# with tab3:
#     st.subheader("Comparación por causa de muerte")
# 
#     if not Datos_Filtro.empty:
#         # Agrupamos los datos por causa y calculamos la media
#         Barras_Datos = Datos_Filtro.groupby('Causa de muerte', as_index=False)['Muertes por 100000 habitantes'].mean()

#         barras = alt.Chart(Barras_Datos).mark_bar().encode(
#             x=alt.X('Muertes por 100000 habitantes:Q', title='Muertes por 100.000 habitantes'),
#             y=alt.Y('Causa de muerte:N', sort='-x', title='Causa de muerte'),
#             tooltip=['Causa de muerte', 'Muertes por 100000 habitantes']
#         ).properties(
#             width=800,
#             height=600
#         )
# 
#         st.altair_chart(barras, use_container_width=True)
#     else:
#         st.warning("No hay datos para mostrar la gráfica de barras.")

with tab3:
    st.subheader("Selecciona los dos intervalos temporales a comparar:")

    if not Datos_Filtro.empty:

        Años_Disponibles = sorted(Datos_Filtro['Periodo'].unique())
        col1, col2 = st.columns(2)

        with col1:
            Rango_1 = st.slider("Selecciona el primer intervalo de años", min_value=min(Años_Disponibles), max_value=max(Años_Disponibles), value=(Años_Disponibles[0], Años_Disponibles[len(Años_Disponibles)//3]),key="slider_1")

        with col2:
            Rango_2 = st.slider("Selecciona el segundo intervalo de años", min_value=min(Años_Disponibles), max_value=max(Años_Disponibles), value=(Años_Disponibles[0], Años_Disponibles[len(Años_Disponibles)*2//3]),key="slider_2")

        # Filtrar los datos para cada intervalo
        Intervalo_1 = Datos_Filtro[(Datos_Filtro['Periodo'] >= Rango_1[0]) & (Datos_Filtro['Periodo'] <= Rango_1[1])]
        Intervalo_2 = Datos_Filtro[(Datos_Filtro['Periodo'] >= Rango_2[0]) & (Datos_Filtro['Periodo'] <= Rango_2[1])]

        # Promedio de muertes por 100.000 por causa
        Media_1 = Intervalo_1.groupby('Causa de muerte', as_index=False)['Muertes por 100000 habitantes'].mean()
        Media_2 = Intervalo_2.groupby('Causa de muerte', as_index=False)['Muertes por 100000 habitantes'].mean()

        # Añadir etiquetas para identificar de qué intervalo vienen
        Media_1['Intervalo'] = f"{Rango_1[0]}–{Rango_1[1]}"
        Media_2['Intervalo'] = f"{Rango_2[0]}–{Rango_2[1]}"

        # Unir ambos datasets
        Comparacion = pd.concat([Media_1, Media_2])

        # Ranking inverso (1 = más muertes)
        Comparacion['Rank'] = Comparacion.groupby('Intervalo')['Muertes por 100000 habitantes'].rank(ascending=False, method='first')

        st.subheader("Ranking comparativo de muertes por 100.000 habitantes")

        if not Comparacion.empty and len(Comparacion['Intervalo'].unique()) == 2:

            # Opcional: añadir clasificación por grupo si quieres colorear por categoría
            # Comparacion['Grupo'] = Comparacion['Causa de muerte'].map(dict_grupo)

            # Asegurar orden de Intervalo en eje X
            Comparacion['Intervalo'] = pd.Categorical(Comparacion['Intervalo'], categories=sorted(Comparacion['Intervalo'].unique()), ordered=True)
            
            # Escala del Rango
            
            x_scale = alt.Scale(range=[700, 1200])
            
            # Líneas que conectan las posiciones
            lines = alt.Chart(Comparacion).mark_line().encode(
                x=alt.X('Intervalo:N', title=None, scale = x_scale),
                y=alt.Y('Rank:Q', title='Ranking', scale=alt.Scale(domain=(Comparacion['Rank'].max() + 1, 0)), axis= None),
                color=alt.Color('Causa de muerte:N', legend=None),
                detail='Causa de muerte:N'
            )

            # Etiquetas a la izquierda (primer intervalo)
            left_labels = alt.Chart(Comparacion[Comparacion['Intervalo'] == Comparacion['Intervalo'].min()]).mark_text(
                align='right',
                baseline='middle',
                dx=-5,
                fontSize=13
            ).encode(
                x=alt.X('Intervalo:N', scale = x_scale),
                y=alt.Y('Rank:Q'),
                text='Causa de muerte:N',
                color=alt.Color('Causa de muerte:N', legend=None)
            )

            # Etiquetas a la derecha (segundo intervalo)
            right_labels = alt.Chart(Comparacion[Comparacion['Intervalo'] == Comparacion['Intervalo'].max()]).mark_text(
                align='left',
                baseline='middle',
                dx=5,
                fontSize=13
            ).encode(
                x=alt.X('Intervalo:N', scale = x_scale),
                y=alt.Y('Rank:Q'),
                text='Causa de muerte:N',
                color=alt.Color('Causa de muerte:N', legend=None)
            )

            # Combinar líneas y etiquetas
            chart = (lines + left_labels + right_labels).properties(width=1200, height=1000).interactive().configure_view(continuousWidth=1600,continuousHeight=1000)


            st.altair_chart(chart, use_container_width=True)

        else:
            st.warning("No hay suficientes datos para mostrar la comparación de intervalos.")
       
# Esto concluye las representaciones de las causas de muerte.

# Ahora, se representan las muertes por comunidad autónoma.

st.subheader("Visualización de las Muertes por Comunidad Autónoma")

tab1, tab2, tab3 = st.tabs(["📊 Mapa de Muertes", "📊 Gráfico de líneas", "📊 Gráfico de flechas"])

with tab1:
    st.subheader("Mapa de Muertes en España")

    if not Datos_Filtro.empty:
        
       URL_mapa = "https://public.tableau.com/views/DeathsinSpainper100_000inhabitants/DeathinSpain?:language=es-ES&:embed=true"
       st.components.v1.iframe(URL_mapa, height = 800, scrolling = True)

with tab2:
    st.subheader("Evolución temporal de las muertes por 100.000 habitantes")

    if not Datos_Filtro.empty:
        Datos_Filtro['Periodo'] = Datos_Filtro['Periodo'].astype(int)

        Datos_Agrupados = Datos_Filtro.groupby(
            ['Periodo', 'Comunidades y Ciudades Autónomas'],
            as_index=False
            )['Muertes por 100000 habitantes'].mean()
        
        Datos_Agrupados = Datos_Agrupados.dropna(subset=['Muertes por 100000 habitantes'])
        
        # Calcular el mínimo y máximo de forma segura
        minimo = Datos_Agrupados['Muertes por 100000 habitantes'].min()
        maximo = Datos_Agrupados['Muertes por 100000 habitantes'].max()
        
        margen = 10
        y_min = max(minimo - margen, 0)
        
        chart = alt.Chart(Datos_Agrupados).mark_line(point=True).encode(
            x=alt.X('Periodo:O', title='Año'),
            y=alt.Y('Muertes por 100000 habitantes:Q',
                    title='Muertes por 100.000 habitantes',
                    scale=alt.Scale(domain=[y_min, maximo])  # Corregido: dominio válido y ordenado
                    ),
            color=alt.Color('Comunidades y Ciudades Autónomas:N',
                            title='Comunidad/Ciudad Autónoma',
                            scale=alt.Scale(scheme='category20'),
                            legend=alt.Legend(
                                title="Comunidad o ciudad autónoma",
                                labelLimit=0,
                                titleLimit=0,
                                orient="bottom",
                                columns=1,
                                labelFontSize=11,
                                titleFontSize=16
                                )
                            ),
            tooltip=['Periodo', 'Comunidades y Ciudades Autónomas', 'Muertes por 100000 habitantes']
            ).properties(
                width=900,
                height=1200
                ).interactive()
                
        st.altair_chart(chart, use_container_width=True)

with tab3:
    st.subheader("Selecciona los dos intervalos temporales a comparar:")

    if not Datos_Filtro.empty:

        Años_Disponibles = sorted(Datos_Filtro['Periodo'].unique())
        col1, col2 = st.columns(2)

        with col1:
            Rango_3 = st.slider("Selecciona el primer intervalo de años", min_value=min(Años_Disponibles), max_value=max(Años_Disponibles), value=(Años_Disponibles[0], Años_Disponibles[len(Años_Disponibles)//3]),key="slider_3")

        with col2:
            Rango_4 = st.slider("Selecciona el segundo intervalo de años", min_value=min(Años_Disponibles), max_value=max(Años_Disponibles), value=(Años_Disponibles[0], Años_Disponibles[len(Años_Disponibles)*2//3]),key="slider_4")

        # Filtrar los datos para cada intervalo
        Intervalo_3 = Datos_Filtro[(Datos_Filtro['Periodo'] >= Rango_3[0]) & (Datos_Filtro['Periodo'] <= Rango_3[1])]
        Intervalo_4 = Datos_Filtro[(Datos_Filtro['Periodo'] >= Rango_4[0]) & (Datos_Filtro['Periodo'] <= Rango_4[1])]

        # Promedio de muertes por 100.000 por causa
        Media_3 = Intervalo_3.groupby('Comunidades y Ciudades Autónomas', as_index=False)['Muertes por 100000 habitantes'].mean()
        Media_4 = Intervalo_4.groupby('Comunidades y Ciudades Autónomas', as_index=False)['Muertes por 100000 habitantes'].mean()

        # Añadir etiquetas para identificar de qué intervalo vienen
        Media_3['Intervalo'] = f"{Rango_3[0]}–{Rango_3[1]}"
        Media_4['Intervalo'] = f"{Rango_4[0]}–{Rango_4[1]}"

        # Unir ambos datasets
        Comparacion = pd.concat([Media_3, Media_4])

        # Ranking inverso (1 = más muertes)
        Comparacion['Rank'] = Comparacion.groupby('Intervalo')['Muertes por 100000 habitantes'].rank(ascending=False, method='first')

        st.subheader("Ranking comparativo de muertes por 100.000 habitantes")

        if not Comparacion.empty and len(Comparacion['Intervalo'].unique()) == 2:

            # Opcional: añadir clasificación por grupo si quieres colorear por categoría
            # Comparacion['Grupo'] = Comparacion['Causa de muerte'].map(dict_grupo)

            # Asegurar orden de Intervalo en eje X
            Comparacion['Intervalo'] = pd.Categorical(Comparacion['Intervalo'], categories=sorted(Comparacion['Intervalo'].unique()), ordered=True)
            
            # Escala del Rango
            
            x_scale = alt.Scale(range=[1050, 1450])
            
            # Líneas que conectan las posiciones
            lines = alt.Chart(Comparacion).mark_line().encode(
                x=alt.X('Intervalo:N', title=None, scale = x_scale),
                y=alt.Y('Rank:Q', title='Ranking', scale=alt.Scale(domain=(Comparacion['Rank'].max() + 1, 0)), axis= None),
                color=alt.Color('Comunidades y Ciudades Autónomas:N', legend=None),
                detail='Comunidades y Ciudades Autónomas:N'
            )

            # Etiquetas a la izquierda (primer intervalo)
            left_labels = alt.Chart(Comparacion[Comparacion['Intervalo'] == Comparacion['Intervalo'].min()]).mark_text(
                align='right',
                baseline='middle',
                dx=-5,
                fontSize=18
            ).encode(
                x=alt.X('Intervalo:N', scale = x_scale),
                y=alt.Y('Rank:Q'),
                text='Comunidades y Ciudades Autónomas:N',
                color=alt.Color('Comunidades y Ciudades Autónomas:N', legend=None)
            )

            # Etiquetas a la derecha (segundo intervalo)
            right_labels = alt.Chart(Comparacion[Comparacion['Intervalo'] == Comparacion['Intervalo'].max()]).mark_text(
                align='left',
                baseline='middle',
                dx=5,
                fontSize=18
            ).encode(
                x=alt.X('Intervalo:N', scale = x_scale),
                y=alt.Y('Rank:Q'),
                text='Comunidades y Ciudades Autónomas:N',
                color=alt.Color('Comunidades y Ciudades Autónomas:N', legend=None)
            )

            # Combinar líneas y etiquetas
            chart = (lines + left_labels + right_labels).properties(width=1200, height=1000).interactive().configure_view(continuousWidth=1600,continuousHeight=1000)


            st.altair_chart(chart, use_container_width=True)

        else:
            st.warning("No hay suficientes datos para mostrar la comparación de intervalos.")
            
# Esto concluye las representaciones de las comunidades autónomas.

# Ahora, se representan las muertes por sexo.

st.subheader("Visualización de las Muertes por Sexo")

tab1, tab2, tab3 = st.tabs(["📊 Gráfio de líneas", "📊 Gráfico de barras (causas)", "📊 Gráfico de barras (autonomías)"])

with tab1:  
    st.subheader("Evolución temporal de las muertes por 100.000 habitantes según sexo")

    if not Datos.empty:
        # Promediar o sumar según lo que quieras comparar (normalmente promedio por sexo/año)
        Evolucion = Datos_Filtro.groupby(["Periodo", "Sexo"], as_index=False)["Muertes por 100000 habitantes"].mean()

        chart = alt.Chart(Evolucion).mark_line(point=True).encode(
            x=alt.X("Periodo:O", title="Año"),
            y=alt.Y("Muertes por 100000 habitantes:Q", title="Muertes por 100.000 habitantes"),
            color=alt.Color("Sexo:N", title="Sexo"),
            tooltip=["Periodo", "Sexo", alt.Tooltip("Muertes por 100000 habitantes:Q", format=".2f")]
        ).properties(
            width=900,
            height=500
        ).interactive()

        st.altair_chart(chart, use_container_width=True)

    else:
        st.warning("No hay datos suficientes para mostrar la evolución temporal.")

with tab2:  
    st.subheader("Distribución de las causas de muerte por sexo")

    if not Datos_Filtro.empty:
        
        # Se agrega por causa y sexo 
        Barras = Datos_Filtro.groupby(["Causa de muerte", "Sexo"], as_index=False)["Muertes por 100000 habitantes"].mean()

        # Ajuste para compensar que los datos se dan por hombres y mujeres
        Barras["Muertes por 100000 habitantes/2"] = Barras["Muertes por 100000 habitantes"] / 2


        chart = alt.Chart(Barras).mark_bar().encode(
            x=alt.X("sum(Muertes por 100000 habitantes/2):Q", title="Muertes por 100.000 habitantes"),
            y=alt.Y("Causa de muerte:N", sort="-x", title="Causa de muerte"),
            color=alt.Color("Sexo:N", title="Sexo", scale=alt.Scale(scheme="set1")),
            tooltip=[
                alt.Tooltip("Causa de muerte:N", title="Causa"),
                alt.Tooltip("Sexo:N", title="Sexo"),
                alt.Tooltip("sum(Muertes por 100000 habitantes):Q", format=".2f", title="Muertes por 100.000")
            ]
        ).properties(
            width=900,
            height=600
        )

        st.altair_chart(chart, use_container_width=True)
    else:
        st.warning("No hay datos suficientes para generar la gráfica de barras apiladas.")

with tab3:  
    st.subheader("Distribución de muertes por autonomía y sexo")

    if not Datos_Filtro.empty:
        
        # Se agrega por causa y sexo
        Barras = Datos_Filtro.groupby(["Comunidades y Ciudades Autónomas", "Sexo"], as_index=False)["Muertes por 100000 habitantes"].mean()
        
        # Ajuste para compensar que los datos se dan por hombres y mujeres
        Barras["Muertes por 100000 habitantes/2"] = Barras["Muertes por 100000 habitantes"] / 2


        chart = alt.Chart(Barras).mark_bar().encode(
            x=alt.X("sum(Muertes por 100000 habitantes/2):Q", title="Muertes por 100.000 habitantes"),
            y=alt.Y("Comunidades y Ciudades Autónomas:N", sort="-x", title="Comunidades y Ciudades Autónomas"),
            color=alt.Color("Sexo:N", title="Sexo", scale=alt.Scale(scheme="set1")),
            tooltip=[
                alt.Tooltip("Comunidades y Ciudades Autónomas:N", title="Causa"),
                alt.Tooltip("Sexo:N", title="Sexo"),
                alt.Tooltip("sum(Muertes por 100000 habitantes):Q", format=".2f", title="Muertes por 100.000")
            ]
        ).properties(
            width=900,
            height=600
        )

        st.altair_chart(chart, use_container_width=True)
    else:
        st.warning("No hay datos suficientes para generar la gráfica de barras apiladas.")

# Esto concluye las representaciones relacionadas con el sexo.

# Ahora, se hacen predicciones.

st.subheader("Predicciones del avance de la mortalidad")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Método estadístico (Holt Winters)", "📊 Método estadístico (ARIMA)", "📊 Random Forest", "📊 Red neuronal (Perceptrón multicapa)"])





with tab1:  
    st.subheader("Predicción futura con Exponential Smoothing (Holt-Winters)")

    if not Datos.empty:
        # Selección de causa y sexo

        Causa = st.selectbox("Selecciona la causa de muerte", Datos_Filtro["Causa de muerte"].unique())
        Sexo = st.selectbox("Selecciona el sexo", Datos_Filtro["Sexo"].unique())
        Comunidad = st.selectbox("Selecciona la autonomía", Datos_Filtro["Comunidades y Ciudades Autónomas"].unique())
        Edad = st.selectbox("Selecciona el intervalo de edades", Datos_Filtro["Edad"].unique())
        Rango_Años = st.slider("Selecciona un rango temporal", min_value=Min_Año, max_value=Max_Año, value=(Min_Año, Max_Año))
        
        # Filtrar
        serie = Datos_Filtro[
            (Datos_Filtro["Causa de muerte"] == Causa) & 
            (Datos_Filtro["Sexo"] == Sexo) &
            (Datos_Filtro["Comunidades y Ciudades Autónomas"] == Comunidad) &
            (Datos_Filtro["Edad"] == Edad) &
            (Datos_Filtro["Periodo"].between(Rango_Años[0], Rango_Años[1]))
            ].groupby("Periodo", as_index=True)["Muertes por 100000 habitantes"].mean()

        if len(serie) > 5:  # Una cantidad de datos suficiente
        
            # Ajuste con un modelo Holt-Winters
            
            modelo = ExponentialSmoothing(serie, trend="mul", seasonal="mul", seasonal_periods=20)
            ajustado = modelo.fit()

            # Predicción de futuro
            pasos = st.slider("Selecciona años a predecir", 1, 20, 20)
            pred = ajustado.forecast(pasos)
            
            # Ajustar el índice de la predicción a años futuros
            ultimo_año = serie.index.max()
            pred.index = range(ultimo_año + 1, ultimo_año + pasos + 1)
            
            # Gráfico con Plotly
            
            # Crear figura
            fig = go.Figure()

            # Datos reales
            fig.add_trace(go.Scatter(
                x=serie.index, 
                y=serie.values,
                mode="lines+markers",
                name="Datos reales"
                ))

            # Predicción
            fig.add_trace(go.Scatter(
                x=pred.index,
                y=pred.values,
                mode="lines+markers",
                name="Predicción",
                line=dict(dash="dash")
                ))

            # Personalización
            fig.update_layout(
                title="Predicción de muertes por 100.000 habitantes por un método de exponential smoothing",
                xaxis_title="Año",
                yaxis_title="Muertes por 100.000 habitantes",
                hovermode="x unified",
                template="plotly_white",
                width=900,
                height=500
                )

            # Mostrar en Streamlit
            st.plotly_chart(fig, use_container_width=True)

            # Preparación de datos para Altair
            #df_real = pd.DataFrame({
            #    "Año": serie.index,
            #    "Muertes por 100000 habitantes": serie.values,
            #    "Tipo": "Datos reales"
            #})

            #df_pred = pd.DataFrame({
            #    "Año": pred.index,
            #    "Muertes por 100000 habitantes": pred.values,
            #    "Tipo": "Predicción"
            #})

            #df_all = pd.concat([df_real, df_pred])

            # Gráfico con Altair
            #chart = alt.Chart(df_all).mark_line(point=True).encode(
            #    x=alt.X("Año:O", title="Año"),
            #    y=alt.Y("Muertes por 100000 habitantes:Q", title="Muertes por 100.000 habitantes"),
            #    color=alt.Color("Tipo:N", scale=alt.Scale(domain=["Datos reales","Predicción"],range=["steelblue","red"])),
            #    strokeDash=alt.StrokeDash("Tipo:N", title="Serie",
            #                              scale=alt.Scale(domain=["Datos reales", "Predicción"],
            #                                              range=[[1], [5,5]]))  # sólida y discontinua
            #).properties(
            #    title="Predicción de muertes por 100.000 habitantes con Exponential Smoothing",
            #    width=900,
            #    height=500
            #).interactive()

            # Mostrar en Streamlit
            #st.altair_chart(chart, use_container_width=True)

        else:
            st.warning("No hay suficientes datos históricos para entrenar el modelo.")
    else:
        st.warning("No hay datos cargados para hacer la predicción.")
        
        

with tab2:  
    st.subheader("Predicción futura con ARIMA/SARIMA")

    if not Datos.empty:
        # Selección de filtros
        Causa = st.selectbox("Selecciona la causa de muerte (ARIMA)", Datos_Filtro["Causa de muerte"].unique())
        Sexo = st.selectbox("Selecciona el sexo (ARIMA)", Datos_Filtro["Sexo"].unique())
        Comunidad = st.selectbox("Selecciona la autonomía (ARIMA)", Datos_Filtro["Comunidades y Ciudades Autónomas"].unique())
        Edad = st.selectbox("Selecciona el intervalo de edades (ARIMA)", Datos_Filtro["Edad"].unique())
        Rango_Años = st.slider("Selecciona un rango temporal (ARIMA)", min_value=Min_Año, max_value=Max_Año, value=(Min_Año, Max_Año))
        
        # Filtrar
        serie = Datos_Filtro[
            (Datos_Filtro["Causa de muerte"] == Causa) & 
            (Datos_Filtro["Sexo"] == Sexo) &
            (Datos_Filtro["Comunidades y Ciudades Autónomas"] == Comunidad) &
            (Datos_Filtro["Edad"] == Edad) &
            (Datos_Filtro["Periodo"].between(Rango_Años[0], Rango_Años[1]))
            ].groupby("Periodo", as_index=True)["Muertes por 100000 habitantes"].mean()

        if len(serie) > 5:
            try:
                # Ajustar ARIMA manualmente (ejemplo: ARIMA(1,1,1))
                modelo = sm.tsa.ARIMA(serie, order=(1,1,1))
                ajustado = modelo.fit()

                # Predicción futura
                pasos = st.slider("Selecciona años a predecir (ARIMA)", 1, 20, 10)
                pred = ajustado.get_forecast(steps=pasos)
                pred_mean = pred.predicted_mean
                conf_int = pred.conf_int()

                # Ajustar índices
                ultimo_año = serie.index.max()
                pred_mean.index = range(ultimo_año + 1, ultimo_año + pasos + 1)
                conf_int.index = pred_mean.index

                # DataFrames para Altair
                df_real = pd.DataFrame({
                    "Año": serie.index,
                    "Muertes por 100000 habitantes": serie.values,
                    "Tipo": "Datos reales"
                })

                df_pred = pd.DataFrame({
                    "Año": pred_mean.index,
                    "Muertes por 100000 habitantes": pred_mean.values,
                    "Tipo": "Predicción"
                })

                df_conf = pd.DataFrame({
                    "Año": conf_int.index,
                    "lower": conf_int.iloc[:,0],
                    "upper": conf_int.iloc[:,1]
                })

                df_all = pd.concat([df_real, df_pred])

                # Gráfico
                chart = alt.Chart(df_all).mark_line(point=True).encode(
                    x=alt.X("Año:O", title="Año"),
                    y=alt.Y("Muertes por 100000 habitantes:Q", title="Muertes por 100.000 habitantes"),
                    color=alt.Color(
                        "Tipo:N",
                        scale=alt.Scale(domain=["Datos reales", "Predicción"], range=["steelblue", "red"])
                    ),
                    strokeDash=alt.StrokeDash(
                        "Tipo:N",
                        scale=alt.Scale(domain=["Datos reales", "Predicción"], range=[[1], [5,5]])
                    )
                )

                band = alt.Chart(df_conf).mark_area(opacity=0.2, color="red").encode(
                    x="Año:O",
                    y="lower:Q",
                    y2="upper:Q"
                )

                final_chart = (band + chart).properties(
                    title="Predicción de muertes por 100.000 habitantes con ARIMA simple",
                    width=900,
                    height=500
                ).interactive()

                st.altair_chart(final_chart, use_container_width=True)

            except Exception as e:
                st.error(f"Error al ajustar el modelo ARIMA simple: {e}")
        else:
            st.warning("No hay suficientes datos históricos para entrenar el modelo ARIMA simple.")
            
            
with tab3:
    st.subheader("Predicción futura con Random Forest")

    if not Datos.empty:
        # Selección de filtros
        Causa = st.selectbox("Selecciona la causa de muerte", Datos_Filtro["Causa de muerte"].unique(), key = "Filtro11")
        Sexo = st.selectbox("Selecciona el sexo", Datos_Filtro["Sexo"].unique(), key = "Filtro22")
        Comunidad = st.selectbox("Selecciona la autonomía", Datos_Filtro["Comunidades y Ciudades Autónomas"].unique(), key = "Filtro33")
        Edad = st.selectbox("Selecciona el intervalo de edades", Datos_Filtro["Edad"].unique(), key = "Filtro44")
        Rango_Años = st.slider("Selecciona un rango temporal", min_value=Min_Año, max_value=Max_Año, value=(Min_Año, Max_Año), key = "Filtro55")

        # Filtrar datos
        serie = Datos_Filtro[
            (Datos_Filtro["Causa de muerte"] == Causa) &
            (Datos_Filtro["Sexo"] == Sexo) &
            (Datos_Filtro["Comunidades y Ciudades Autónomas"] == Comunidad) &
            (Datos_Filtro["Edad"] == Edad) &
            (Datos_Filtro["Periodo"].between(Rango_Años[0], Rango_Años[1]))
        ].groupby("Periodo", as_index=True)["Muertes por 100000 habitantes"].mean()

        if len(serie) > 10:  # necesitamos datos suficientes

            # Crear dataset supervisado: usar lags como predictores
            df = pd.DataFrame({"y": serie})
            for i in range(1, 16):  # 15 lags
                df[f"lag{i}"] = df["y"].shift(i)
            df = df.dropna()

            X = df[["lag1", "lag2", "lag3", "lag4", "lag5", "lag6", "lag7", "lag8", "lag9", "lag10", "lag11", "lag12", "lag13", "lag14", "lag15"]]
            y = df["y"]

            # Entrenar modelo Random Forest
            model = RandomForestRegressor(n_estimators=500, random_state=42)
            model.fit(X, y)

            # Predicción futura
            pasos = st.slider("Selecciona años a predecir", 1, 20, 10)

            ultimos = df.iloc[-1][["lag1", "lag2", "lag3", "lag4", "lag5", "lag6", "lag7", "lag8", "lag9", "lag10", "lag11", "lag12", "lag13", "lag14", "lag15"]].values.tolist()
            predicciones = []

            for _ in range(pasos):
                pred = model.predict([ultimos])[0]
                predicciones.append(pred)
                ultimos = [pred] + ultimos[:-1]  # shift lags

            # Crear dataframe de resultados
            ultimo_año = serie.index.max()
            pred_index = list(range(ultimo_año + 1, ultimo_año + pasos + 1))

            # Crear figura Plotly
            fig = go.Figure()

            # Datos reales
            fig.add_trace(go.Scatter(
                x=serie.index,
                y=serie.values,
                mode="lines+markers",
                name="Datos reales",
                line=dict(color="steelblue")
            ))

            # Predicción
            fig.add_trace(go.Scatter(
                x=pred_index,
                y=predicciones,
                mode="lines+markers",
                name="Predicción",
                line=dict(color="red", dash="dash")
            ))

            # Layout
            fig.update_layout(
                title="Predicción de muertes por 100.000 habitantes con Random Forest",
                xaxis_title="Año",
                yaxis_title="Muertes por 100.000 habitantes",
                hovermode="x",
                template="plotly_white",
                width=900,
                height=500
            )

            # Mostrar en Streamlit
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("No hay suficientes datos históricos para entrenar el modelo de Random Forest.")
    else:
        st.warning("No hay datos cargados para hacer la predicción.")
        
        
with tab4:
    st.subheader("Predicción futura con Red Neuronal MLP")

    if not Datos.empty:
        # Selección de filtros
        Causa = st.selectbox("Selecciona la causa de muerte", Datos_Filtro["Causa de muerte"].unique(), key = "Filtro1")
        Sexo = st.selectbox("Selecciona el sexo", Datos_Filtro["Sexo"].unique(), key = "Filtro2")
        Comunidad = st.selectbox("Selecciona la autonomía", Datos_Filtro["Comunidades y Ciudades Autónomas"].unique(), key = "Filtro3")
        Edad = st.selectbox("Selecciona el intervalo de edades", Datos_Filtro["Edad"].unique(), key = "Filtro4")
        Rango_Años = st.slider("Selecciona un rango temporal", min_value=Min_Año, max_value=Max_Año, value=(Min_Año, Max_Año), key = "Filtro5")

        # Filtrar datos
        serie = Datos_Filtro[
            (Datos_Filtro["Causa de muerte"] == Causa) &
            (Datos_Filtro["Sexo"] == Sexo) &
            (Datos_Filtro["Comunidades y Ciudades Autónomas"] == Comunidad) &
            (Datos_Filtro["Edad"] == Edad) &
            (Datos_Filtro["Periodo"].between(Rango_Años[0], Rango_Años[1]))
        ].groupby("Periodo", as_index=True)["Muertes por 100000 habitantes"].mean()

        if len(serie) > 10:

            # Normalización
            scaler = MinMaxScaler()
            serie_scaled = scaler.fit_transform(serie.values.reshape(-1,1))

            # Crear dataset supervisado
            def crear_dataset(data, lag=15):
                X, y = [], []
                for i in range(lag, len(data)):
                    X.append(data[i-lag:i, 0])
                    y.append(data[i, 0])
                return np.array(X), np.array(y)

            X, y = crear_dataset(serie_scaled, lag=15)

            # Modelo MLP
            model = Sequential()
            model.add(Dense(64, activation="relu", input_shape=(X.shape[1],)))
            model.add(Dense(32, activation="relu"))
            model.add(Dense(1))
            model.compile(optimizer="adam", loss="mse")
            model.fit(X, y, epochs=200, verbose=0)

            # Predicción futura
            pasos = st.slider("Selecciona años a predecir", 1, 20, 10, key = "Predict")
            predicciones = []
            entrada = X[-1]

            for _ in range(pasos):
                pred = model.predict(entrada.reshape(1, -1))[0,0]
                predicciones.append(pred)
                entrada = np.append(entrada[1:], pred)

            # Desnormalizar
            predicciones = scaler.inverse_transform(np.array(predicciones).reshape(-1,1)).ravel()

            # Índices futuros
            ultimo_año = serie.index.max()
            pred_index = list(range(ultimo_año + 1, ultimo_año + pasos + 1))

            # Plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=serie.index, y=serie.values, mode="lines+markers", name="Datos reales", line=dict(color="steelblue")))
            fig.add_trace(go.Scatter(x=pred_index, y=predicciones, mode="lines+markers", name="Predicción", line=dict(color="green", dash="dash")))
            fig.update_layout(title="Predicción de muertes por 100.000 habitantes con Red Neuronal MLP", hovermode = "x unified",
                              xaxis_title="Año", yaxis_title="Muertes por 100.000 habitantes",
                              template="plotly_white", width=900, height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No hay suficientes datos históricos para entrenar la red neuronal MLP.")
    else:
        st.warning("No hay datos cargados para hacer la predicción.")