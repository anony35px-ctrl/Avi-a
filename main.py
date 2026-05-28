import speech_recognition as sr
import os

recognizer = sr.Recognizer()

def reconocer_audio(archivo):

    with sr.AudioFile(archivo) as source:
        audio = recognizer.record(source)

    try:
        texto = recognizer.recognize_google(audio, language='es-ES')
        texto = texto.lower()

        print("Comando detectado:", texto)

        ejecutar_comando(texto)

    except Exception as e:
        print("No se pudo reconocer el audio")
        print(e)

def ejecutar_comando(comando):

    if "abrir word" in comando:
        print("Abriendo Word...")
        os.system("start winword")

    elif "chrome" in comando:
        print("Abriendo Chrome...")
        os.system("start chrome")

    elif "apagar equipo" in comando:
        print("Apagando equipo...")
        os.system("shutdown /s /t 5")

    else:
        print("Comando no reconocido")

# -------- MENU --------

while True:

    print("\n--- ASISTENTE VIRTUAL ---")
    print("1. Reproducir audio1")
    print("2. Reproducir audio2")
    print("3. Reproducir audio3")
    print("4. Salir")

    opcion = input("Selecciona opción: ")

    if opcion == "1":
        reconocer_audio("audio1.wav")

    elif opcion == "2":
        reconocer_audio("audio2.wav")

    elif opcion == "3":
        reconocer_audio("audio3.wav")

    elif opcion == "4":
        break

    else:
        print("Opción inválida")