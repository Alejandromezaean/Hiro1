import os
import wave
import pyaudio
from google.cloud import speech
import tkinter as tk
from tkinter import messagebox

# Ruta del archivo JSON con las credenciales
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:\\Users\\SALA\\Documents\\archivoia.json"

class SpeechToTextApp:
    def _init_(self, root):
        self.root = root
        self.root.title("Speech to Text")

        self.record_button = tk.Button(root, text="Grabar Audio", command=self.record_audio)
        self.record_button.pack(pady=20)

        self.transcribe_button = tk.Button(root, text="Transcribir Audio", command=self.transcribe_audio)
        self.transcribe_button.pack(pady=20)

        self.output_text = tk.Text(root, wrap=tk.WORD, width=50, height=10)
        self.output_text.pack(pady=20)

        self.audio_file_path = "C:\\Users\\SALA\\Documents\\miAudioo.wav"  # Ruta del archivo de audio

    def record_audio(self):
        # Configuración de PyAudio
        chunk = 1024
        sample_format = pyaudio.paInt16  # Formato de audio
        channels = 1
        fs = 48000  # Frecuencia de muestreo
        seconds = 5  # Duración de la grabación

        p = pyaudio.PyAudio()

        print("Grabando...")
        stream = p.open(format=sample_format, channels=channels,
                        rate=fs, input=True,
                        frames_per_buffer=chunk)

        frames = []

        for _ in range(0, int(fs / chunk * seconds)):
            data = stream.read(chunk)
            frames.append(data)

        print("Grabación terminada.")

        stream.stop_stream()
        stream.close()
        p.terminate()

        # Guardar el archivo de audio
        with wave.open(self.audio_file_path, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(p.get_sample_size(sample_format))
            wf.setframerate(fs)
            wf.writeframes(b''.join(frames))

        messagebox.showinfo("Información", "Audio grabado con éxito.")

    def transcribe_audio(self):
        client = speech.SpeechClient()
        print(f"Leyendo archivo de audio: {self.audio_file_path}")

        with open(self.audio_file_path, "rb") as audio_file:
            content = audio_file.read()

        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=48000,
            language_code="es-CO"
        )

        print("Enviando solicitud a Google Speech-to-Text...")
        response = client.recognize(config=config, audio=audio)

        self.output_text.delete(1.0, tk.END)  # Limpiar el área de texto

        if response.results:
            for result in response.results:
                self.output_text.insert(tk.END, f'Transcripción: {result.alternatives[0].transcript}\n')
        else:
            self.output_text.insert(tk.END, "No se recibió ninguna transcripción.")

if __name__ == "_main_":
    root = tk.Tk()
    app = SpeechToTextApp(root)
    root.mainloop()