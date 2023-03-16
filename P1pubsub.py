# pubsub_hilos.py
import redis
import random
import time
import json
import threading
import numpy as np

i = 0

class Subscriber(threading.Thread):
    def __init__(self, r1, channels):
        threading.Thread.__init__(self)
        self.redis, self.init = r1, 0
        self.pubsub = self.redis.pubsub()
        print('Inicializado Subscriber...')
        try:
            self.pubsub.subscribe(channels)
        except Exception as e:
            print(e)

    def work(self, item):
        global data
        # datos1 = 0
        # global definir aquí variables a comunicar entre hilos
        try:
            data = json.loads(item.decode('utf8'))
            print(data)
            #print(type(data))
            if type(data) == dict:
                guardarDatoX(data["id"])
                guardarDatoY(data["valor"])
            # Implementar lógica de recolección de datos
            # Organizar los datos
        except Exception as e:
            print(e)

    def run(self):
        while True:
            try:
                message = self.pubsub.get_message()
                if message:
                    self.work(message['data'])
                time.sleep(0.01)
            except ConnectionError:  # Para reconectar a redis server :)
                print('[lost connection]')
                while True:
                    print('trying to reconnect...')
                    try:
                        self.redis.ping()
                    except ConnectionError:
                        time.sleep(10)
                    else:
                        self.pubsub.subscribe(['canal1-1'])
                        break
                time.sleep(0.01)  # Esto es para darle tiempo al sistema :)

class Publisher(threading.Thread):
    def __init__(self, r2):
        threading.Thread.__init__(self)
        global json_datos  ## Variable json a publicar
        json_datos = json.dumps({"id": "inicio"})
        time.sleep(0.1)  ### un poco de tiempo
        try:
            r2.publish("canal1-0", json_datos)  # publicar dato en canal-1
        except Exception as e:
            print(e)

    def run(self):
        while True:
            time.sleep(0.01)  # Esto es para darle tiempo al sistema :)

def crearArchivoX():
    datosNumpy = np.array(datosAlmacenadosX)
    try:
        np.savetxt("data1-X.csv",datosNumpy, delimiter=",")
        print("SE GUARDARON LOS DATOS")
    except:
        print("NO SE PUDO GUARDAR LOS DATOS")
        
def crearArchivoY():
    datosNumpy = np.array(datosAlmacenadosY)
    try:
        np.savetxt("data1-Y.csv",datosNumpy, delimiter=",")
        print("SE GUARDARON LOS DATOS")
    except:
        print("NO SE PUDO GUARDAR LOS DATOS")

datosAlmacenadosX = []
def guardarDatoX(dato):
    datosAlmacenadosX.append(dato)
    crearArchivoX()

datosAlmacenadosY = []
def guardarDatoY(dato):
    datosAlmacenadosY.append(dato)
    crearArchivoY()


if __name__ == "__main__":
    global r
    try:
        r = redis.Redis(host='192.168.68.105', port=6379, db=0)
    except Exception as e:
        print(e)
    sub = Subscriber(r, ['canal1-1'])
    sub.start()
    pub = Publisher(r)
    pub.start()
    print(i)
