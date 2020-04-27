import cv2
import numpy as np

def stackImagenes(scale, imgArray):
    filas = len(imgArray)
    columnas = len(imgArray[0])
    filasDisponibles = isinstance(imgArray[0], list)
    ancho = imgArray[0][0].shape[1]
    alto = imgArray[0][0].shape[0]
    if filasDisponibles:
        for dato in range ( 0, filas):
            for y in range(0, columnas):
                if imgArray[dato][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[dato][y] = cv2.resize(imgArray[dato][y], (0, 0), None, scale, scale)
                else:
                    imgArray[dato][y] = cv2.resize(imgArray[dato][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[dato][y].shape) == 2: imgArray[dato][y]= cv2.cvtColor( imgArray[dato][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((alto, ancho, 3), np.uint8)
        hor = [imageBlank]*filas
        hor_con = [imageBlank]*filas
        for dato in range(0, filas):
            hor[dato] = np.hstack(imgArray[dato])
        ver = np.vstack(hor)
    else:
        for dato in range(0, filas):
            if imgArray[dato].shape[:2] == imgArray[0].shape[:2]:
                imgArray[dato] = cv2.resize(imgArray[dato], (0, 0), None, scale, scale)
            else:
                imgArray[dato] = cv2.resize(imgArray[dato], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[dato].shape) == 2: imgArray[dato] = cv2.cvtColor(imgArray[dato], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

def obtenerContornos(img, contornoImagen):
    contorno,jerarquia = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contorno:
        area = cv2.contourArea(cnt)
        print(area)
        if area>500:
            cv2.drawContours(contornoImagen, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt,True)
            #print(peri)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)
            print(len(approx))
            objCor = len(approx)
            x, y, w, h = cv2.boundingRect(approx)

            if objCor ==3: objectType ="Triangulo"
            elif objCor == 4:
                aspRatio = w/float(h)
                if aspRatio >0.98 and aspRatio <1.03: objectType= "Cuadrado"
                else:objectType="Rectangulo"
            elif objCor>4: objectType= "Circulo"
            else:objectType="Desconocida"



            cv2.rectangle(contornoImagen, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(contornoImagen, objectType,
                        (x+(w//2)-10,y+(h//2)-10), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                        (0,0,0), 2)

def reconocimientoFiguras():
    ruta = 'DirFiguras/shapes.png'
    img = cv2.imread(ruta)
    contornoImagen = img.copy()

    imgGris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGris, (7, 7), 1)
    imgCanny = cv2.Canny(imgBlur,50,50)
    obtenerContornos(imgCanny, contornoImagen)

    imgBlanca = np.zeros_like(img)
    imgStack = stackImagenes(0.8, ([img, imgGris, imgBlur],
                                   [imgCanny, contornoImagen, imgBlanca]))

    cv2.imshow("Imagenes", imgStack)

    cv2.waitKey(0)


if __name__ == '__main__':
    reconocimientoFiguras()