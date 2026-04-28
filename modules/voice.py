import pyttsx3

class Voice:
    def __init__(self):
        pass

    def speak(self, text):
        try:
            engine = pyttsx3.init()   
            engine.setProperty('rate', 170)
            engine.setProperty('volume', 1.0)

            engine.say(text)
            engine.runAndWait()

            engine.stop()  

        except Exception as e:
            print("Voice Error:", e)