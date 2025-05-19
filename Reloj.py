import requests

API_KEY = "1g7722ebax5mxnkicfmqo094e"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Accept": "application/json"
}

# Paso 1: Obtener actividades
activities_url = "https://intervals.icu/api/v1/activities"
response = requests.get(activities_url, headers=HEADERS)

if response.status_code != 200:
    print("Error al obtener actividades:", response.text)
    exit()

activities = response.json()
if not activities:
    print("No hay actividades disponibles.")
    exit()

# Paso 2: Tomar la última actividad
last_activity_id = activities[0]["id"]

# Paso 3: Obtener detalles de esa actividad
activity_url = f"https://intervals.icu/api/v1/activities/{last_activity_id}"
response = requests.get(activity_url, headers=HEADERS)

if response.status_code != 200:
    print("Error al obtener datos de la actividad:", response.text)
    exit()

activity = response.json()
hr_data = activity.get("hr", [])

if not hr_data:
    print("No hay datos de frecuencia cardíaca en esta actividad.")
    exit()

# Paso 4: Calcular HR promedio
hr_values = [point["v"] for point in hr_data]
avg_hr = sum(hr_values) / len(hr_values)

# Paso 5: Evaluar estado
if avg_hr <= 80:
    print(f"Promedio de FC: {avg_hr:.1f} bpm → ESTÁS BIEN")
else:
    print(f"Promedio de FC: {avg_hr:.1f} bpm → ESTÁS MAL")