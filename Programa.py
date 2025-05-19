import os
import cv2
import numpy as np
import mtcnn
import joblib
import json
import tkinter as tk
from tkinter import messagebox
import speech_recognition as sr
from google.cloud import speech
import base64
from google.cloud import firestore
from cryptography.fernet import Fernet
import pyttsx3
from architecture import *
from train_v2 import normalize, l2_normalizer
from scipy.spatial.distance import cosine
from tensorflow.keras.layers import Conv2D
import pickle
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore
import uuid
from playsound import playsound

##Credenciales Firestone
cred = credentials.Certificate("Base.json") 
firebase_admin.initialize_app(cred)

key = Fernet.generate_key()
cipher_suite = Fernet(key)

db = firestore.client()

#Credenciales Google
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\redesneuronales-451900-bd460539cbb2.json"
health_data_dir = './health_data'
os.makedirs(health_data_dir, exist_ok=True)
#Modelos IA
model = joblib.load('IAhipertension.pkl')
scaler = joblib.load('IAscaler.pkl')

confidence_t = 0.99
recognition_t = 0.5
required_size = (160, 160)

collection_name = "usuarios"

def encrypt_name(name):
    encrypted = cipher_suite.encrypt(name.encode())
    return base64.b64encode(encrypted).decode()

def decrypt_name(encrypted_name):
    decrypted = cipher_suite.decrypt(base64.b64decode(encrypted_name))
    return decrypted.decode()

def save_encrypted_user_data(data):
    # Cifrar solo el nombre
    data["name"] = encrypt_name(data["name"])
    random_id = str(uuid.uuid4()) 

    # Guardar en Firestore
    doc_ref = db.collection(collection_name).document(random_id)
    doc_ref.set(data)
    print(f"Datos del paciente guardados con ID anónimo: {random_id}")

def get_user_data(user_id):
    doc_ref = db.collection(collection_name).document(user_id)
    doc = doc_ref.get()
    if doc.exists:
        data = doc.to_dict()
        data["name"] = decrypt_name(data["name"])
        print(f"Datos descifrados de {user_id}: {data}")
        return data
    else:
        print("Usuario no encontrado.")
        return None
    
collected_data_users = {}

def get_face(img, box):
    x1, y1, width, height = box
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = img[y1:y2, x1:x2]
    return face, (x1, y1), (x2, y2)

def get_encode(face_encoder, face, size):
    face = normalize(face)
    face = cv2.resize(face, size)
    encode = face_encoder.predict(np.expand_dims(face, axis=0))[0]
    return encode

def load_pickle(path):
    with open(path, 'rb') as f:
        encoding_dict = pickle.load(f)
    return encoding_dict

def detect(img, detector, encoder, encoding_dict):
    global detection_active
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(img_rgb)
    for res in results:
        if res['confidence'] < confidence_t:
            continue
        face, pt_1, pt_2 = get_face(img_rgb, res['box'])
        encode = get_encode(encoder, face, required_size)
        encode = l2_normalizer.transform(encode.reshape(1, -1))[0]
        name = 'unknown'

        distance = float("inf")
        for db_name, db_encode in encoding_dict.items():
            dist = cosine(db_encode, encode)
            if dist < recognition_t and dist < distance:
                name = db_name
                distance = dist

        if name != 'unknown':
            if name not in collected_data_users:
                collected_data_users[name] = {'count': 1} 
                save_data(name)  # Recolectar datos por primera vez
            else:
                collected_data_users[name]['count'] += 1

            # Segundo conteo
            if collected_data_users[name]['count'] == 2:
                st_app.speak(f"Hola {name}, ¿quieres actualizar tus datos?")
                response = st_app.transcribe_from_mic()
                if response and 'sí' in response.lower():
                    save_data(name)
                else:
                    st_app.speak("¿Cuál es tu tensión arterial actual?")
                    tension = st_app.transcribe_from_mic()
                    data_path = os.path.join(health_data_dir, f'{name}_health.json')
                    if os.path.exists(data_path):
                        with open(data_path, 'r') as f:
                            data = json.load(f)
                    else:
                        data = {}

                    data["heartRate"] = tension
                    data["heartRate"] = float(data["heartRate"])
                    with open(data_path, 'w') as f:
                        json.dump(data, f, indent=4)
                    
            #Tercer Conteo    
            elif collected_data_users[name]['count'] == 3:
                data["heartRate"] = tension
                st_app.speak(f"{name}, recuerda que tu última tension fue {tension}. ¿Quieres actualizar tus datos?")
                response = st_app.transcribe_from_mic()
                if response and 'sí' in response.lower():
                    save_data(name)
                else:
                    st_app.speak("¿Cuál es tu tensión arterial actual?")
                    tension = st_app.transcribe_from_mic()
                    data_path = os.path.join(health_data_dir, f'{name}_health.json')
                    if os.path.exists(data_path):
                        with open(data_path, 'r') as f:
                            data = json.load(f)
                    else:
                        data = {}

                    data["heartRate"] = tension
                    data["heartRate"] = float(data["heartRate"])
                    with open(data_path, 'w') as f:
                        json.dump(data, f, indent=4)

            cv2.rectangle(img, pt_1, pt_2, (0, 255, 0), 2)
            cv2.putText(img, name + f'__{distance:.2f}', (pt_1[0], pt_1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 200, 200), 2)
        else:
            cv2.rectangle(img, pt_1, pt_2, (0, 0, 255), 2)
            cv2.putText(img, name, pt_1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
    detection_active = False
    return img

class SpeechToTextApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Conversión de Voz a Texto")
        self.root.geometry("400x300")
        self.root.configure(bg="#F0F8FF") 

        self.engine = pyttsx3.init()

    def transcribe_from_mic(self):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("Escuchando...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)
            print("Audio capturado, procesando...")

        try:
            client = speech.SpeechClient()
            audio_data = audio.get_wav_data()
            recognition_audio = speech.RecognitionAudio(content=audio_data)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=44100,
                language_code="es-CO"
            )
            response = client.recognize(config=config, audio=recognition_audio)

            if response.results:
                transcription = ''
                for result in response.results:
                    transcription += result.alternatives[0].transcript + ' '
                return transcription.strip()
            else:
                return ''
        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un error al transcribir: {str(e)}")
            print(f"Error al transcribir: {e}")
            return ''

    def speak(self, text):
        self.engine.say(text)
        self.engine.runAndWait()

def save_data(name):
    data = {}
    data["name"] = name
    st_app.speak(f"Hola {name}, por favor responde las siguientes preguntas.")
    st_app.speak(f"¿Eres hombre?")
    data["male"] = st_app.transcribe_from_mic()
    data["male"] = 1 if data["male"] and 'sí' in data["male"].lower() else 0
    st_app.speak(f"¿Cuál es tu edad?")
    data["age"] = (st_app.transcribe_from_mic())
    st_app.speak(f"¿Eres fumador?")
    data["currentSmoker"] = st_app.transcribe_from_mic()
    data["currentSmoker"] = 1 if data["currentSmoker"] and 'sí' in data["currentSmoker"].lower() else 0
    st_app.speak(f"¿Cuántos cigarrillos fumas por día?")
    data["cigsPerDay"] = (st_app.transcribe_from_mic())
    st_app.speak(f"¿Estás tomando medicamentos para la presión arterial?")
    data["BPMeds"] = st_app.transcribe_from_mic()
    data["BPMeds"] = 1 if data["BPMeds"] and 'sí' in data["BPMeds"].lower() else 0
    st_app.speak(f"¿Tienes diabetes?")
    data["diabetes"] = st_app.transcribe_from_mic()
    data["diabetes"] = 1 if data["diabetes"] and 'sí' in data["diabetes"].lower() else 0
    st_app.speak(f"¿Cuál es tu nivel de colesterol total?")
    data["totChol"] = (st_app.transcribe_from_mic())
    st_app.speak(f"¿Cuál es tu presión arterial sistólica?")
    data["sysBP"] = (st_app.transcribe_from_mic())
    st_app.speak(f"¿Cuál es tu presión arterial diastólica?")
    data["diaBP"] = (st_app.transcribe_from_mic())
    st_app.speak(f"¿Cuál es tu índice de masa corporal?")
    data["BMI"] = (st_app.transcribe_from_mic())
    st_app.speak(f"¿Cuál es tu frecuencia cardíaca?")
    data["heartRate"] = (st_app.transcribe_from_mic())
    st_app.speak(f"¿Cuál es tu nivel de glucosa?")
    data["glucose"] = (st_app.transcribe_from_mic())

    nombres = {
    "age": "edad",
    "cigsPerDay": "cigarrillos por día",
    "totChol": "colesterol total",
    "sysBP": "presión sistólica",
    "diaBP": "presión diastólica",
    "BMI": "índice de masa corporal",
    "heartRate": "frecuencia cardíaca",
    "glucose": "nivel de glucosa"
}
    
    for key in ["age", "cigsPerDay", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]:
        try:
            data[key] = float(data[key])
        except ValueError:
            nombre_voz = nombres.get(key, key)  
            st_app.speak(f"No fue posible identificar el valor ingresado de {nombre_voz}")
            st_app.speak(f"Por favor, vuelve a repetir el valor de {nombre_voz}")
            data[key] = st_app.transcribe_from_mic()
            data[key] = float(data[key])

    data_path = os.path.join(health_data_dir, f'{name}_health.json')
    with open(data_path, 'w') as f:
        json.dump(data, f)

    #Cifrar Datos
    save_encrypted_user_data(data)

    xn = np.array([data["male"], data["age"] , data["currentSmoker"],data["cigsPerDay"], data["BPMeds"], data["diabetes"],
                    data["totChol"], data["sysBP"], data["diaBP"], data["BMI"], data["heartRate"], data["glucose"]])
    Xn_std = scaler.transform(xn.reshape(1, -1))  # Escalado y redimensionado

       #Predicción
    resul = model.predict(Xn_std)
    if resul[0] == 0:
        st_app.speak(f"{name}, segun la recolección de sus datos medicos, usted no tiene peligro de padecer hipertension.")
    else:
        st_app.speak(f"{name}, segun la recolección de sus datos medicos, usted podria padecer de hipertension")

def esperar_comando_activacion():
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("Esperando comando de activación ('oye hiro' o 'hiro')...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)

        try:
            client = speech.SpeechClient()
            audio_data = audio.get_wav_data()
            recognition_audio = speech.RecognitionAudio(content=audio_data)
            config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=44100,
            language_code="es-CO"
            )
            response = client.recognize(config=config, audio=recognition_audio)

            if response.results:
                transcription = ''
                for result in response.results:
                    transcription += result.alternatives[0].transcript + ' '
                transcription = transcription.strip().lower()
                print("Transcripción:", transcription)

                if "oye hero" in transcription or transcription == "hero" or transcription == "giro" or transcription == "oye giro" :
                    print("¡Activación detectada!")
                    return True
                else:
                    print("No se detectó la palabra clave.")
                    return False
            else:
                return False
        except Exception as e:
            print(f"Error durante el reconocimiento de activación: {e}")
            return False

if __name__ == "__main__":
    face_encoder = InceptionResNetV2()
    face_encoder.load_weights("facenet_keras_weights.h5")
    face_detector = mtcnn.MTCNN()
    encoding_dict = load_pickle('encodings/encodings.pkl')
    
    root = tk.Tk()
    st_app = SpeechToTextApp(root)
    
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('camera', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if esperar_comando_activacion():
            playsound("Start.mp3")
            st_app.speak("Hola, Soy Hiro")
            detection_active = True
            detect(frame, face_detector, face_encoder, encoding_dict)

    cap.release()
    cv2.destroyAllWindows()
