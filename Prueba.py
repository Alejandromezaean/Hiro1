import os
import cv2
import numpy as np
import mtcnn
import joblib
import json
import tkinter as tk
from tkinter import messagebox, simpledialog
import speech_recognition as sr
from google.cloud import speech
import pyttsx3
from architecture import *
from train_v2 import normalize, l2_normalizer
from scipy.spatial.distance import cosine
from tensorflow.keras.models import load_model
import pickle
from datetime import datetime

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\Aleja\Documents\archivoia.json"
health_data_dir = './health_data'
os.makedirs(health_data_dir, exist_ok=True)
model = joblib.load('IAhipertension2.pkl')
scaler = joblib.load('IAscaler2.pkl')

confidence_t = 0.99
recognition_t = 0.5
required_size = (160, 160)

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

def detect(img, detector, encoder, encoding_dict, voice=True):
    data = {}   
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
                st_app.speak(f"Hola {name}, ¿Como prefieres responder las preguntas?") 
                ask_response_format(name)  # Nuevo: Preguntar por voz o texto
                save_data(name)
            else:
                collected_data_users[name]['count'] += 1
                st_app.speak(f"Hola {name}, ¿Como prefieres responder las preguntas?")
                ask_response_format(name)
                if user_input_method == 'voice':
                    st_app.speak(f"Hola {name}, ¿quieres actualizar tus datos?")
                    response = st_app.transcribe_from_mic()
                    if response and 'sí' in response.lower():
                        save_data(name)
                    else:
                        st_app.speak("¿Cuál es tu frecuencia cardiaca?")
                        data_path = os.path.join(health_data_dir, f'{name}_health.json')
                        with open(data_path, 'r') as f:
                            data = json.load(f)
                        try:
                            data["heartRate"] = float(st_app.transcribe_from_mic())
                        except ValueError:
                            st_app.speak("El valor proporcionado no es un número válido.")
                            data_path = os.path.join(health_data_dir, f'{name}_health.json')
                            with open(data_path, 'w') as f:
                                json.dump(data, f)
                        
                else:
                    response = simpledialog.askstring("Pregunta", "¿quieres actualizar los datos?")
                    if response and 'si' in response.lower():
                        save_data(name)
                    else:
                        st_app.speak("¿Cuál es tu frecuencia cardiaca?")
                        data_path = os.path.join(health_data_dir, f'{name}_health.json')
                        with open(data_path, 'r') as f:
                            data = json.load(f)
                        data["heartRate"] = float(simpledialog.askstring("Pregunta", "¿Cuál es tu frecuencia cardiaca?"))
                        data_path = os.path.join(health_data_dir, f'{name}_health.json')
                        with open(data_path, 'w') as f:
                            json.dump(data, f)

            cv2.rectangle(img, pt_1, pt_2, (0, 255, 0), 2)
            cv2.putText(img, name + f'__{distance:.2f}', (pt_1[0], pt_1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 200, 200), 2)
        else:
            cv2.rectangle(img, pt_1, pt_2, (0, 0, 255), 2)
            cv2.putText(img, name, pt_1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
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

def ask_response_format(name):
    global user_input_method
    def select_voice():
        global user_input_method
        user_input_method = 'voice'
        choice_window.destroy()

    def select_text():
        global user_input_method
        user_input_method = 'text'
        choice_window.destroy()

    choice_window = tk.Tk()
    choice_window.title("Método de Entrada")
    choice_window.geometry("300x150")
    tk.Label(choice_window, text="¿Cómo prefieres responder las preguntas?").pack(pady=10)

    tk.Button(choice_window, text="Voz", command=select_voice).pack(side=tk.LEFT, padx=20, pady=20)
    tk.Button(choice_window, text="Texto", command=select_text).pack(side=tk.RIGHT, padx=20, pady=20)

    choice_window.mainloop()

def save_data(name):
    data = {}
    data["name"] = name

    if user_input_method == 'voice':
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
    else:
        data["male"] = simpledialog.askstring("Pregunta", "¿Eres hombre? (sí/no)")
        data["male"] = 1 if data["male"] and 'si' in data["male"].lower() else 0
        data["age"] = simpledialog.askfloat("Pregunta", "¿Cuál es tu edad?")
        data["currentSmoker"] = simpledialog.askstring("Pregunta", "¿Eres fumador? (Responde 'sí' o 'no')")
        data["currentSmoker"] = 1 if data["currentSmoker"] and 'si' in data["currentSmoker"].lower() else 0
        data["cigsPerDay"] = simpledialog.askstring("Pregunta", "¿Cuántos cigarrillos fumas por día?")
        data["BPMeds"] = simpledialog.askstring("Pregunta", "¿Estás tomando medicamentos para la presión arterial? (Responde 'sí' o 'no')")
        data["BPMeds"] = 1 if data["BPMeds"] and 'si' in data["BPMeds"].lower() else 0
        data["diabetes"] = simpledialog.askstring("Pregunta", "¿Tienes diabetes? (Responde 'sí' o 'no')")
        data["diabetes"] = 1 if data["diabetes"] and 'si' in data["diabetes"].lower() else 0
        data["totChol"] = simpledialog.askstring("Pregunta", "¿Cuál es tu nivel de colesterol total?")
        data["sysBP"] = simpledialog.askstring("Pregunta", "¿Cuál es tu presión arterial sistólica?")
        data["diaBP"] = simpledialog.askstring("Pregunta", "¿Cuál es tu presión arterial diastólica?")
        data["BMI"] = simpledialog.askstring("Pregunta", "¿Cuál es tu índice de masa corporal?")
        data["heartRate"] = simpledialog.askstring("Pregunta", "¿Cuál es tu frecuencia cardíaca?")
        data["glucose"] = simpledialog.askstring("Pregunta", "¿Cuál es tu nivel de glucosa?")


    # Procesamiento numérico y guardado de datos se mantiene igual...
    for key in ["age", "cigsPerDay", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]:
        try:
            data[key] = float(data[key])
        except ValueError:
            messagebox.showerror("Error", f"El valor ingresado en {key} no es numérico.")

    data_path = os.path.join(health_data_dir, f'{name}_health.json')
    with open(data_path, 'w') as f:
        json.dump(data, f)

    xn = np.array([data["male"], data["age"] , data["currentSmoker"],data["cigsPerDay"], data["BPMeds"], data["diabetes"], data["totChol"], data["sysBP"], data["diaBP"], data["BMI"], data["heartRate"], data["glucose"]])
    Xn_std = scaler.transform(xn.reshape(1, -1))

    resul = model.predict(Xn_std)
    if resul[0] == 0:
        st_app.speak(f"Felicidades, no cuentas con hipertension en este momento")
        message = "Recuerda hacerte chequeos anualmente para prevenir esta enfermedad."
    else:
        st_app.speak(f"Podrias padecer de hipertension, asegurate de consultar con tu medico")
        message = "Consulta con tu medico y recuerda tener una vida saludable."
    messagebox.showinfo("Resultado", message)

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
        frame = detect(frame, face_detector, face_encoder, encoding_dict)
        cv2.imshow('camera', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
