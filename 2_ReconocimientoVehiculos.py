import numpy as np
import cv2 as cv

def reconocimientoVehiculos():
    captura = cv.VideoCapture('DirAutos/vid2.mp4')

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




if __name__ == '__main__':
    reconocimientoVehiculos()
