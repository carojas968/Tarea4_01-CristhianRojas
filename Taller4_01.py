# ========================= TALLER 4 CRISTHIAN ALEAJNDRO ROJAS MARTINEZ ============================
import numpy as np
import random
# import matplotlib.pyplot as plt
import cv2
import os
import sys
from enum import Enum


# -----------  PRIMERA PARTE Generador de cuadriláteros
class Quadrilateral:
    N = 500

    def __init__(self, N):
        if N % 2 == 0:
            self.N = N
        else:
            print(" Por favor digite un numero par")

    def generate(self):
        N = self.N
        x1 = random.randint(0, N / 2)
        y1 = random.randint(0, N / 2)
        x2 = random.randint(N / 2, N)
        y2 = random.randint(0, N / 2)
        x3 = random.randint(N / 2, N)
        y3 = random.randint(N / 2, N)
        x4 = random.randint(0, N / 2)
        y4 = random.randint(N / 2, N)
        array_created = np.zeros([N, N, 3], dtype=np.uint8)
        # Reading an image in default mode
        image = array_created

        # Window name in which image is displayed
        window_name = 'Image'

        # Start coordinate, here (0, 0)
        # represents the top left corner of image
        start_point1 = (x1, y1)

        # End coordinate, here (250, 250)
        # represents the bottom right corner of image
        end_point2 = (x2, y2)

        start_point3 = (x3, y3)
        end_point4 = (x4, y4)

        # Green color in BGR
        color = (255, 0, 255)

        # Line thickness of 9 px
        thickness = 1

        # Using cv2.line() method
        # Dibujar las  4 lineas
        image = cv2.line(image, start_point1, end_point2, color, thickness)
        image = cv2.line(image, end_point2, start_point3, color, thickness)
        image = cv2.line(image, start_point3, end_point4, color, thickness)
        image = cv2.line(image, end_point4, start_point1, color, thickness)
        # Nombre de archivo
        filename = 'savedImage.jpg'

        # usar cv2.imwrite() method
        # guardar la imagen
        cv2.imshow('Cuadrilatero', image)
        cv2.waitKey(0)
        cv2.imwrite(filename, image)


# Elementos de otras clases necesarios para desarrollar el ejercicio
class Methods(Enum):
    Standard = 1
    Direct = 2


def gradient_map(gray):
    # Image derivatives
    scale = 1
    delta = 0
    depth = cv2.CV_16S  # to avoid overflow

    grad_x = cv2.Sobel(gray, depth, 1, 0, ksize=3, scale=scale, delta=delta)
    grad_y = cv2.Sobel(gray, depth, 0, 1, ksize=3, scale=scale, delta=delta)

    grad_x = np.float32(grad_x)
    grad_x = grad_x * (1 / 512)
    grad_y = np.float32(grad_y)
    grad_y = grad_y * (1 / 512)

    # Gradient and smoothing
    grad_x2 = cv2.multiply(grad_x, grad_x)
    grad_y2 = cv2.multiply(grad_y, grad_y)

    # Magnitude of the gradient
    Mag = np.sqrt(grad_x2 + grad_y2)

    # Orientation of the gradient
    theta = np.arctan(cv2.divide(grad_y, grad_x + np.finfo(float).eps))

    return theta, Mag


def orientation_map(gray, n):
    # Image derivatives
    scale = 1
    delta = 0
    depth = cv2.CV_16S  # to avoid overflow

    grad_x = cv2.Sobel(gray, depth, 1, 0, ksize=3, scale=scale, delta=delta)
    grad_y = cv2.Sobel(gray, depth, 0, 1, ksize=3, scale=scale, delta=delta)

    grad_x = np.float32(grad_x)
    grad_x = grad_x * (1 / 512)
    grad_y = np.float32(grad_y)
    grad_y = grad_y * (1 / 512)

    # Gradient and smoothing
    grad_x2 = cv2.multiply(grad_x, grad_x)
    grad_y2 = cv2.multiply(grad_y, grad_y)
    grad_xy = cv2.multiply(grad_x, grad_y)
    g_x2 = cv2.blur(grad_x2, (n, n))
    g_y2 = cv2.blur(grad_y2, (n, n))
    g_xy = cv2.blur(grad_xy, (n, n))

    # Magnitude of the gradient
    Mag = np.sqrt(grad_x2 + grad_y2)
    M = cv2.blur(Mag, (n, n))

    # Gradient local aggregation
    vx = 2 * g_xy
    vy = g_x2 - g_y2
    fi = cv2.divide(vx, vy + np.finfo(float).eps)

    case1 = vy >= 0
    case2 = np.logical_and(vy < 0, vx >= 0)
    values1 = 0.5 * np.arctan(fi)
    values2 = 0.5 * (np.arctan(fi) + np.pi)
    values3 = 0.5 * (np.arctan(fi) - np.pi)
    theta = np.copy(values3)
    theta[case1] = values1[case1]
    theta[case2] = values2[case2]

    return theta, M


class Hough:
    def __init__(self, bw_edges):
        [self.rows, self.cols] = bw_edges.shape[:2]
        self.center_x = self.cols // 2
        self.center_y = self.rows // 2
        self.theta = np.arange(0, 360, 0.5)
        self.bw_edges = bw_edges

    def standard_transform(self):

        rmax = int(round(0.5 * np.sqrt(self.rows ** 2 + self.cols ** 2)))
        y, x = np.where(self.bw_edges >= 1)

        accumulator = np.zeros((rmax, len(self.theta)))

        for idx, th in enumerate(self.theta):
            r = np.around(
                (x - self.center_x) * np.cos((th * np.pi) / 180) + (y - self.center_y) * np.sin((th * np.pi) / 180))
            r = r.astype(int)
            r_idx = np.where(np.logical_and(r >= 0, r < rmax))
            np.add.at(accumulator[:, idx], r[r_idx[0]], 1)
        return accumulator

    def direct_transform(self, theta_data):

        rmax = int(round(0.5 * np.sqrt(self.rows ** 2 + self.cols ** 2)))
        y, x = np.where(self.bw_edges >= 1)

        x_ = x - self.center_x
        y_ = y - self.center_y

        th = theta_data[y, x] + np.pi / 2

        hist_val, bin_edges = np.histogram(th, bins=32)
        print('Histogram', hist_val)

        print(np.amin(th), np.amax(th))
        th[y_ < 0] = th[y_ < 0] + np.pi
        print(np.amin(th), np.amax(th))
        accumulator = np.zeros((rmax, len(self.theta)))

        r = np.around(x_ * np.cos(th) + y_ * np.sin(th))
        r = r.astype(int)
        th = np.around(360 * th / np.pi)
        th = th.astype(int)
        th[th == 720] = 0
        print(np.amin(th), np.amax(th))
        r_idx = np.where(np.logical_and(r >= 0, r < rmax))
        np.add.at(accumulator, (r[r_idx[0]], th[r_idx[0]]), 1)
        return accumulator

    @staticmethod
    def find_peaks(accumulator, nhood, accumulator_threshold, N_peaks):
        done = False
        acc_copy = accumulator
        nhood_center = [(nhood[0] - 1) / 2, (nhood[1] - 1) / 2]
        peaks = []
        while not done:
            [p, q] = np.unravel_index(acc_copy.argmax(), acc_copy.shape)
            if acc_copy[p, q] >= accumulator_threshold:
                peaks.append([p, q])
                p1 = p - nhood_center[0]
                p2 = p + nhood_center[0]
                q1 = q - nhood_center[1]
                q2 = q + nhood_center[1]

                [qq, pp] = np.meshgrid(np.arange(np.max([q1, 0]), np.min([q2, acc_copy.shape[1] - 1]) + 1, 1), \
                                       np.arange(np.max([p1, 0]), np.min([p2, acc_copy.shape[0] - 1]) + 1, 1))
                pp = np.array(pp.flatten(), dtype=np.intp)
                qq = np.array(qq.flatten(), dtype=np.intp)

                acc_copy[pp, qq] = 0
                done = np.array(peaks).shape[0] == N_peaks
            else:
                done = True

        return peaks


# --------------- SEGUNDA PARTE Detector de esquinas


def DetectCorners(image):
    method = Methods.Standard
    high_thresh = 300
    bw_edges = cv2.Canny(image, high_thresh * 0.3, high_thresh, L2gradient=True)  # aca esta el resultado de canny
    hough = Hough(bw_edges)
    if method == Methods.Standard:
        accumulator = hough.standard_transform()
    elif method == Methods.Direct:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        theta, _ = gradient_map(image_gray)
        accumulator = hough.direct_transform(theta)
    else:
        sys.exit()

    acc_thresh = 50
    N_peaks = 4
    nhood = [25, 9]
    peaks = hough.find_peaks(accumulator, nhood, acc_thresh, N_peaks)

    _, cols = image.shape[:2]
    image_draw = np.copy(image)

    matriz = []
    for peak in peaks:
        a = []
        rho = peak[0]
        theta_ = hough.theta[peak[1]]

        theta_pi = np.pi * theta_ / 180
        theta_ = theta_ - 180
        a = np.cos(theta_pi)
        b = np.sin(theta_pi)
        x0 = a * rho + hough.center_x
        y0 = b * rho + hough.center_y
        c = -rho
        x1 = int(round(x0 + cols * (-b)))
        y1 = int(round(y0 + cols * a))
        x2 = int(round(x0 - cols * (-b)))
        y2 = int(round(y0 - cols * a))
        m = (y2 - y1) / (x2 - x1)
        a = [x1, y1, x2, y2, m]
        matriz.append(a)

        image_draw = cv2.line(image_draw, (x1, y1), (x2, y2), [0, 255, 255],
                              thickness=2)  # imagen con las lineas identificadas

    # Proceso para identificar las intersecciones de un cuadrilatero
    puntos = []
    for linea in matriz:
        if np.absolute(linea[4]) > 1:
            k = 1
            for linea2 in matriz:

                if np.absolute(linea2[4]) < 1:
                    ec1 = linea[4] * (k - linea[0]) + linea[1]
                    ec2 = linea2[4] * (k - linea2[0]) + linea2[1]

                    k = (linea[4] * linea[0] - linea2[4] * linea2[0] + linea2[1] - linea[1]) / (linea[4] - linea2[4])
                    x = round(k)
                    y = round(linea[4] * (x - linea[0]) + linea[1])
                    puntos.append([x, y])
    # Proceso para ubicar circulos en cada esquina
    for puntoss in puntos:
        image_corner = cv2.circle(image, (puntoss[0], puntoss[1]), 3, [0, 255, 255], 3)
        # CREACION DE LA IMAGEN CON LOS PUNTOS
    arrayPuntosEsquineros = np.array(puntos)  # ARRAY DE ESQUINEROS
    cv2.imshow("corners", image_corner)
    cv2.waitKey(0)
    cv2.destroyWindow("corners")


# ==================================================================================
#### RESOLUCION DE LA TAREA CON EJEMPLOS
# =====================================================================================

# Gnerador de Cuadrilateros Aleatoreos al final este permite guardar la imagen generada en un archivo de nombre savedImage.jpg
ImagenCuadrilatero = Quadrilateral(1000)
ImagenCuadrilatero.generate()

# Detector de esquinas
# apartir de la imagen generada anteriormente se reciben dos parametros la ruta donde esta alojada la imagen.jpg y el nombre del archivo
path = sys.argv[1]
image_name = sys.argv[2]
path_file = os.path.join(path, image_name)
image = cv2.imread(path_file)
cv2.imshow('Imagen Con Esquinas', image )
cv2.waitKey(0)

DetectCorners(image)
