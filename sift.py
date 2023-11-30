import numpy as np
import cv2
import functools

float_tolerance = 1e-7

'''Generar una imagen base a partir de la imagen de entrada aumentando el muestreo en 2 en ambas direcciones y desenfocándola'''
def BaseImage(image, sigma, assumed_blur):
    image = cv2.resize(image, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    return cv2.GaussianBlur(image, (0, 0), sigmaX=np.sqrt(max((sigma ** 2) - ((2 * assumed_blur) ** 2), 0.01)), sigmaY=np.sqrt(max((sigma ** 2) - ((2 * assumed_blur) ** 2), 0.01)))
  
'''Calcular el número de octavas en la pirámide de imágenes en función de la forma de la imagen base (valor predeterminado de OpenCV)'''
def NumberOfOctaves(image_shape):
    return int(np.round(np.log(min(image_shape)) / np.log(2) - 1))

'''Generar una lista de núcleos gaussianos en los que desenfocar la imagen de entrada.'''
def GaussianKernels(sigma, num_intervals):
    k = 2 ** (1. / num_intervals)
    return [sigma * np.sqrt(k ** (2 * i) - k ** (2 * (i - 1))) for i in range(num_intervals + 3)]

'''Generar pirámide espacial de escala de imágenes gaussianas'''
def GaussianImages(image, num_octaves, gaussian_kernels):
    gaussian_images = []
    for octave_index in range(num_octaves):
        octave_images = [image]
        for kernel in gaussian_kernels[1:]:
            image = cv2.GaussianBlur(image, (0, 0), sigmaX=kernel, sigmaY=kernel)
            octave_images.append(image)
        gaussian_images.append(octave_images)
        image = cv2.resize(octave_images[-3], (octave_images[-3].shape[1] // 2, octave_images[-3].shape[0] // 2), interpolation=cv2.INTER_NEAREST)
    return np.array(gaussian_images, dtype=object)

'''Generar pirámide de imágenes de diferencia de gaussianas'''
def DoGImages(gaussian_images):
    return np.array([[cv2.subtract(second, first) for first, second in zip(gaussian_octave, gaussian_octave[1:])] for gaussian_octave in gaussian_images], dtype=object)
