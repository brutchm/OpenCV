import numpy as np
import cv2 as cv
import face_recognition as fr
import os



def reconocimientoVehiculos():
    captura = cv.VideoCapture('vid2.mp4')

    ret, cuadro = captura.read()

    x, y, width, height = 95, 360, 100, 80
    auto = (x, y, width, height)

    areaInteres = cuadro[y:y + height, x: x + width]
    hsvAI = cv.cvtColor(areaInteres, cv.COLOR_BGR2HSV)
    mascara = cv.inRange(hsvAI, np.array((0., 60., 32.)), np.array((180., 255., 255)))
    roi_hist = cv.calcHist([hsvAI], [0], mascara, [180], [0, 180])
    cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)

    term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)
    cv.imshow('areaInteres', areaInteres)
    while (1):
        ret, cuadro = captura.read()
        if ret == True:

            hsv = cv.cvtColor(cuadro, cv.COLOR_BGR2HSV)
            destino = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

            ret, auto = cv.meanShift(destino, auto, term_crit)

            x, y, w, h = auto
            imagenFinal = cv.rectangle(cuadro, (x, y), (x + w, y + h), 255, 3)

            cv.imshow('destino', destino)
            cv.imshow('imagenFinal', imagenFinal)
            k = cv.waitKey(30) & 0xff
            if k == 27:
                break
        else:
            break


def getCarasCodificadas():
    codificadas = {}

    for ruta, dnombres, fnombres in os.walk("./caras"):
        for c in fnombres:
            if c.endswith(".jpg") or c.endswith(".png"):
                caras = fr.load_image_file("caras/" + c)
                encoding = fr.face_encodings(caras)[0]
                codificadas[c.split(".")[0]] = encoding

    return codificadas


def imagenCodificadaDesconocida(img):
    cara = fr.load_image_file("Caras/" + img)
    codificada = fr.face_encodings(cara)[0]

    return codificada


def clasificarCara(im):
    caras = getCarasCodificadas()
    imagenesCodificadas = list(caras.values())
    nombresConocidos = list(caras.keys())

    img = cv.imread(im, 1)
    # img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    # img = img[:,:,::-1]

    locacionImagen = fr.face_locations(img)
    imagenesCodificadasDesconocidas = fr.face_encodings(img, locacionImagen)

    nombresCaras = []
    for caraCodificada in imagenesCodificadasDesconocidas:
        matches = fr.compare_faces(imagenesCodificadas, caraCodificada)
        nombre = "Desconocido"

        distanciaRostro = fr.face_distance(imagenesCodificadas, caraCodificada)
        indiceMatched = np.argmin(distanciaRostro)
        if matches[indiceMatched]:
            nombre = nombresConocidos[indiceMatched]

        nombresCaras.append(nombre)

        for (top, right, bottom, left), nombre in zip(locacionImagen, nombresCaras):
            cv.rectangle(img, (left - 20, top - 20), (right + 20, bottom + 20), (255, 0, 0), 2)
            cv.rectangle(img, (left - 20, bottom - 15), (right + 20, bottom + 20), (255, 0, 0), cv.FILLED)
            font = cv.FONT_HERSHEY_DUPLEX
            cv.putText(img, nombre, (left - 20, bottom + 15), font, 1.0, (255, 255, 255), 2)

    while True:
        cv.imshow('Video', img)
        if cv.waitKey(1) & 0xFF == ord('q'):
            return nombresCaras


def reconocimientoFacial():
    print(clasificarCara("test2.jpg"))
    k = cv.waitKey(30) & 0xff
    if k == 27:
        return


def ejecutarOpcion(opcion):
    if (int(opcion) == 1):
        reconocimientoVehiculos()
    if (int(opcion) == 2):
        reconocimientoFacial()


if __name__ == '__main__':
    opcion = -1
    while (int(opcion) != 0):
        print("Menu")
        print("1. Reconocimiento de vehiculos")
        print("2. Reconocimiento de rostros")
        print("0. Salir")
        opcion = input()
        ejecutarOpcion(opcion)
    print("Gracias por usar el programa")
