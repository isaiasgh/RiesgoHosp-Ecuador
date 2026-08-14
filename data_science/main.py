import pandas as pd

ruta_original = 'inec_egresos_hospitalarios_2024.csv'

# separador y encoding
df = pd.read_csv(ruta_original, sep=';', encoding='utf-8-sig')

columna_objetivo = 'con_egrpa' 

# muestra balanceada (100 vivos & 100 muertos)
muestra_reducida = df.groupby(columna_objetivo).sample(n=100, random_state=42)

# mezclar filas aleatoriamente
muestra_reducida = muestra_reducida.sample(frac=1, random_state=42).reset_index(drop=True)

# guardamos el CSV reducido
muestra_reducida.to_csv('muestra_para_ia.csv', index=False, sep=';')

print("Muestra de 200 registros creada con éxito!")