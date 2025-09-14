import pandas as pd
import numpy as np

# Leer el archivo original
df = pd.read_csv("C:/Users/ESDPC/Desktop/Confidencial/Máster/TFM/Datos_España.csv",encoding="latin-1",sep="\t", thousands=".")  # Reemplaza con tu archivo

# Calcular el número de filas por parte
partes = np.array_split(df, 8)

# Guardar cada parte como un nuevo archivo CSV
for i, parte in enumerate(partes, start=1):
    parte.to_csv(f"parte_{i}.csv", index=False)