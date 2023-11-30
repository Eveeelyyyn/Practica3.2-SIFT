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

'''Encontrar posiciones de píxeles de todos los extremos del espacio de escala en la pirámide de imágenes'''
def ScaleSpaceExtrema(gaussian_images, dog_images, num_intervals, sigma, image_border_width, contrast_threshold=0.04):
    threshold = np.floor(0.5 * contrast_threshold / num_intervals * 255)  
    keypoints = []

    for octave_index, dog_images_in_octave in enumerate(dog_images):
        for image_index, (first_image, second_image, third_image) in enumerate(zip(dog_images_in_octave, dog_images_in_octave[1:], dog_images_in_octave[2:])):
            for i in range(image_border_width, first_image.shape[0] - image_border_width):
                for j in range(image_border_width, first_image.shape[1] - image_border_width):
                    if PixelAnExtremum(first_image[i-1:i+2, j-1:j+2], second_image[i-1:i+2, j-1:j+2], third_image[i-1:i+2, j-1:j+2], threshold):
                        localization_result = ExtremumViaQuadraticFit(i, j, image_index + 1, octave_index, num_intervals, dog_images_in_octave, sigma, contrast_threshold, image_border_width)
                        if localization_result is not None:
                            keypoint, localized_image_index = localization_result
                            keypoints_with_orientations = KeypointsWithOrientations(keypoint, octave_index, gaussian_images[octave_index][localized_image_index])
                            for keypoint_with_orientation in keypoints_with_orientations:
                                keypoints.append(keypoint_with_orientation)
    return keypoints

'''Devuelve Verdadero si el elemento central de la matriz de entrada de 3x3x3 es estrictamente mayor o menor que todos sus vecinos; Falso en caso contrario.'''
def PixelAnExtremum(first_subimage, second_subimage, third_subimage, threshold):
    center_pixel_value = second_subimage[1, 1]
    if abs(center_pixel_value) <= threshold:
        return False
    if center_pixel_value > 0:
        return np.all(center_pixel_value >= first_subimage) and np.all(center_pixel_value >= third_subimage) and np.all(center_pixel_value >= second_subimage[0, :]) and np.all(center_pixel_value >= second_subimage[2, :]) and center_pixel_value >= second_subimage[1, 0] and center_pixel_value >= second_subimage[1, 2]
    else:
        return np.all(center_pixel_value <= first_subimage) and np.all(center_pixel_value <= third_subimage) and np.all(center_pixel_value <= second_subimage[0, :]) and np.all(center_pixel_value <= second_subimage[2, :]) and center_pixel_value <= second_subimage[1, 0] and center_pixel_value <= second_subimage[1, 2]


'''Refinar iterativamente las posiciones de los píxeles de los extremos del espacio de escala mediante un ajuste cuadrático alrededor de los vecinos de cada extremo'''
def ExtremumViaQuadraticFit(i, j, image_index, octave_index, num_intervals, dog_images_in_octave, sigma, contrast_threshold, image_border_width, eigenvalue_ratio=10, num_attempts_until_convergence=5):
    extremum_is_outside_image = False
    image_shape = dog_images_in_octave[0].shape
    for attempt_index in range(num_attempts_until_convergence):
        first_image, second_image, third_image = dog_images_in_octave[image_index-1:image_index+2]
        pixel_cube = np.stack([first_image[i-1:i+2, j-1:j+2],
                            second_image[i-1:i+2, j-1:j+2],
                            third_image[i-1:i+2, j-1:j+2]]).astype('float32') / 255.
        gradient = GradientAtCenterPixel(pixel_cube)
        hessian = HessianAtCenterPixel(pixel_cube)
        extremum_update = -np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        if abs(extremum_update[0]) < 0.5 and abs(extremum_update[1]) < 0.5 and abs(extremum_update[2]) < 0.5:
            break
        j += int(np.round(extremum_update[0]))
        i += int(np.round(extremum_update[1]))
        image_index += int(np.round(extremum_update[2]))
        if i < image_border_width or i >= image_shape[0] - image_border_width or j < image_border_width or j >= image_shape[1] - image_border_width or image_index < 1 or image_index > num_intervals:
            extremum_is_outside_image = True
            break
    if extremum_is_outside_image:
        return None
    if attempt_index >= num_attempts_until_convergence - 1:
        return None
    functionValueAtUpdatedExtremum = pixel_cube[1, 1, 1] + 0.5 * np.dot(gradient, extremum_update)
    if abs(functionValueAtUpdatedExtremum) * num_intervals >= contrast_threshold:
        xy_hessian = hessian[:2, :2]
        xy_hessian_trace = np.trace(xy_hessian)
        xy_hessian_det = np.linalg.det(xy_hessian)
        if xy_hessian_det > 0 and eigenvalue_ratio * (xy_hessian_trace ** 2) < ((eigenvalue_ratio + 1) ** 2) * xy_hessian_det:
            keypoint = cv2.KeyPoint()
            keypoint.pt = ((j + extremum_update[0]) * (2 ** octave_index), (i + extremum_update[1]) * (2 ** octave_index))
            keypoint.octave = octave_index + image_index * (2 ** 8) + int(np.round((extremum_update[2] + 0.5) * 255)) * (2 ** 16)
            keypoint.size = sigma * (2 ** ((image_index + extremum_update[2]) / np.float32(num_intervals))) * (2 ** (octave_index + 1))  
            keypoint.response = abs(functionValueAtUpdatedExtremum)
            return keypoint, image_index
    return None

'''Gradiente aproximado en el píxel central [1, 1, 1] de una matriz de 3x3x3 utilizando la fórmula de diferencia central de orden O(h^2), donde h es el tamaño del paso'''
def GradientAtCenterPixel(pixel_array):
    return np.array([0.5 * (pixel_array[1, 1, 2] - pixel_array[1, 1, 0]), 0.5 * (pixel_array[1, 2, 1] - pixel_array[1, 0, 1]), 0.5 * (pixel_array[2, 1, 1] - pixel_array[0, 1, 1])])


'''Hessiano aproximado en el píxel central [1, 1, 1] de una matriz de 3x3x3 utilizando una fórmula de diferencia central de orden O(h^2), donde h es el tamaño del paso'''
def HessianAtCenterPixel(pixel_array):
    dxx = pixel_array[1, 1, 2] - 2 * pixel_array[1, 1, 1] + pixel_array[1, 1, 0]
    dyy = pixel_array[1, 2, 1] - 2 * pixel_array[1, 1, 1] + pixel_array[1, 0, 1]
    dss = pixel_array[2, 1, 1] - 2 * pixel_array[1, 1, 1] + pixel_array[0, 1, 1]
    dxy = 0.25 * (pixel_array[1, 2, 2] - pixel_array[1, 2, 0] - pixel_array[1, 0, 2] + pixel_array[1, 0, 0])
    dxs = 0.25 * (pixel_array[2, 1, 2] - pixel_array[2, 1, 0] - pixel_array[0, 1, 2] + pixel_array[0, 1, 0])
    dys = 0.25 * (pixel_array[2, 2, 1] - pixel_array[2, 0, 1] - pixel_array[0, 2, 1] + pixel_array[0, 0, 1])
    return np.array([[dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]])
