import numpy as np
import cv2 as cv
import face_recognition as fr
import os

def getCarasCodificadas():
    codificadas = {}

    for ruta, dnombres, fnombres in os.walk("./DirCaras"):
        for c in fnombres:
            if c.endswith(".jpg") or c.endswith(".png"):
                caras = fr.load_image_file("DirCaras/" + c)
                encoding = fr.face_encodings(caras)[0]
                codificadas[c.split(".")[0]] = encoding

    return codificadas


def imagenCodificadaDesconocida(img):
    cara = fr.load_image_file("DirCaras/" + img)
    codificada = fr.face_encodings(cara)[0]

    return codificada


def clasificarCara(im):
    caras = getCarasCodificadas()
    imagenesCodificadas = list(caras.values())
    nombresConocidos = list(caras.keys())

    img = cv.imread(im, 1)
    # img = cv.resize(img, (0, 0), fx=0.5, fy=0.5)
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
    print(clasificarCara("DirTest/test2.jpg"))
    k = cv.waitKey(30) & 0xff
    if k == 27:
        return


if __name__ == '__main__':
    reconocimientoFacial()