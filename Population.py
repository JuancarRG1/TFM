import pandas as pd
import numpy as np

# Cargar el DataFrame (ajusta según tu archivo si lo estás leyendo)
df = pd.read_csv(r"C:\\Users\\ESDPC\\Desktop\\Confidencial\\Máster\\TFM\\56940.csv", sep=";", thousands=".", encoding="utf-8")

# Eliminar las filas según el prefijo de la columna "Periodo"
df_filtrado = df[~df["Periodo"].str.startswith(("1 de abril", "1 de julio", "1 de octubre"))]

# Lista de años a eliminar (como texto, porque aparecen como parte de un string)
anios_excluir = tuple(str(año) for año in range(1971, 1981))  # del 1971 al 1980 inclusive

# Filtrar filas que NO terminan en esos años
df_filtrado = df_filtrado[~df_filtrado["Periodo"].str.endswith(anios_excluir)]

def categorizar_edad(edad_str):
   
    try:
        edad = int(edad_str.split()[0])  # extrae número de "1 año", "45 años", etc.
    except:
        return "Todas las edades"


    if edad < 1:
        return "Menos de 1 año"
    elif edad <= 14:
        return "De 1 a 14 años"
    elif edad <= 29:
        return "De 15 a 29 años"
    elif edad <= 39:
        return "De 30 a 39 años"
    elif edad <= 44:
        return "De 40 a 44 años"
    elif edad <= 49:
        return "De 45 a 49 años"
    elif edad <= 54:
        return "De 50 a 54 años"
    elif edad <= 59:
        return "De 55 a 59 años"
    elif edad <= 64:
        return "De 60 a 64 años"
    elif edad <= 69:
        return "De 65 a 69 años"
    elif edad <= 74:
        return "De 70 a 74 años"
    elif edad <= 79:
        return "De 75 a 79 años"
    elif edad <= 84:
        return "De 80 a 84 años"
    elif edad <= 89:
        return "De 85 a 89 años"
    elif edad <= 94:
        return "De 90 a 94 años"
    else:
        return "95 y más años"

# Aplicar la función
df_filtrado["Intervalo de edad"] = df_filtrado["Edad simple"].apply(categorizar_edad)

# Agrupar por intervalo de edad
df_agrupado = df_filtrado.groupby(["Sexo", "Comunidades y ciudades autónomas", "Periodo", "Intervalo de edad"],as_index=False)["Total"].sum().reset_index()


# Opcional: resetear índice si lo deseas
df_filtrado.reset_index(drop=True, inplace=True)

# Mostrar o guardar
print(df_filtrado.head())
df_agrupado.to_csv("Population.csv", index=False, encoding="latin1", sep=";")