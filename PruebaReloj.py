import os
import time
import pandas as pd

# Función para descargar carpetas de Google Drive usando gdown
def descargar_carpeta(folder_id, output_folder):
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    comando = f"gdown --folder {url} -O {output_folder} --fuzzy"
    os.system(comando)

# Procesamiento de archivos CSV de pasos
def procesar_pasos(carpeta):
    for archivo in os.listdir(carpeta):
        if archivo.endswith('.csv'):
            ruta_archivo = os.path.join(carpeta, archivo)
            print(f"\nProcesando archivo de pasos: {ruta_archivo}")

            try:
                df = pd.read_csv(ruta_archivo)

                # Asegúrate de que el nombre de la columna sea correcto
                if 'Pasos' in df.columns:
                    for pasos in df['Pasos']:
                        if pasos > 2000:
                            print(f"Vas bien ✅ (Pasos: {pasos})")
                        else:
                            print(f"Tienes que caminar más 🚶‍♂️ (Pasos: {pasos})")
                else:
                    print("No se encontró la columna 'Pasos' en este archivo.")
            except Exception as e:
                print(f"Error leyendo {archivo}: {e}")

# Procesamiento de archivos CSV de frecuencia cardiaca
def procesar_frecuencia_cardiaca(carpeta):
    for archivo in os.listdir(carpeta):
        if archivo.endswith('.csv'):
            ruta_archivo = os.path.join(carpeta, archivo)
            print(f"\nProcesando archivo de frecuencia cardiaca: {ruta_archivo}")

            try:
                df = pd.read_csv(ruta_archivo)

                # Asegúrate de que el nombre de la columna sea correcto
                if 'Frecuencia cardiaca' in df.columns:
                    for frecuencia in df['Frecuencia cardiaca']:
                        if frecuencia >= 60:
                            print(f"OK ✅ (Frecuencia: {frecuencia})")
                        else:
                            print(f"UY ⚠️ (Frecuencia: {frecuencia})")
                else:
                    print("No se encontró la columna 'Frecuencia_Cardiaca' en este archivo.")
            except Exception as e:
                print(f"Error leyendo {archivo}: {e}")

# IDs de las carpetas de Google Drive (ya están listos para ti)
carpeta_pasos_id = '1qcLDPE9GO7wmEPSNFW-cl7UIoHPR7mkv'  # ID de la carpeta de pasos
carpeta_frecuencia_id = '1vopjlkuoCNjvwzngp6DsX7id5ZGgjE-m'   #ID de la carpeta de frecuencia cardiaca

# Carpetas locales donde se descargarán los archivos
carpeta_pasos_output = 'datos_pasos'
carpeta_frecuencia_output = 'datos_frecuencia'

while True:
    print("\nDescargando datos de pasos...")
    descargar_carpeta(carpeta_pasos_id, carpeta_pasos_output)

    print("\nDescargando datos de frecuencia cardiaca...")
    descargar_carpeta(carpeta_frecuencia_id, carpeta_frecuencia_output)

    print("\nProcesando datos de pasos...")
    procesar_pasos(carpeta_pasos_output)

    print("\nProcesando datos de frecuencia cardiaca...")
    procesar_frecuencia_cardiaca(carpeta_frecuencia_output)

    print("\nEsperando 5 minutos...\n")
    time.sleep(500)  # Espera de 5 minutos