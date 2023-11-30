import numpy as np
import cv2
import sift
from matplotlib import pyplot as plt

if __name__ == "__main__":
    img1 = cv2.imread('04803880_11119539065.jpg', 0)  # queryImage
    img2 = cv2.imread('00883281_9633489441.jpg', 0)  # trainImage

    sigma=1.6
    num_intervals=3
    assumed_blur=0.5
    image_border_width=5

    image = img1.astype('float32')

    '''Espacio de escala'''
    base_image = sift.BaseImage(image, sigma, assumed_blur)
    num_octaves = sift.NumberOfOctaves(base_image.shape)
    gaussian_kernels = sift.GaussianKernels(sigma, num_intervals)
    gaussian_images = sift.GaussianImages(base_image, num_octaves, gaussian_kernels)

    # Mostrar 4 octavas con 5 imagenes cada una
    plt.figure(figsize=(20, 10))
    for octave_index, octave in enumerate(gaussian_images[:4]):
        for image_index, image in enumerate(octave[:5]):
            plt.subplot(4, 5, octave_index * 5 + image_index + 1)
            plt.imshow(image, cmap='gray')
            plt.title(f"Octave {octave_index+1}, Image {image_index+1}")
            plt.axis('on')

    plt.tight_layout()
    plt.show()

    '''diferencia de Gaussianas'''
    dog_images = sift.DoGImages(gaussian_images)

    # Mostrar la diferencia de gaussianas
    plt.figure(figsize=(20, 10))
    for octave_index, dog_images_in_octave in enumerate(dog_images[:4]):
        for dog_image_index, dog_image in enumerate(dog_images_in_octave):
            plt.subplot(4, len(dog_images_in_octave), octave_index * len(dog_images_in_octave) + dog_image_index + 1)
            plt.imshow(dog_image, cmap='gray')
            plt.title(f"Octave {octave_index+1}, DoG {dog_image_index+1}")
            plt.axis('off')

    plt.tight_layout()
    plt.show()

    '''Histograma de orientacion'''
    keypoints = sift.ScaleSpaceExtrema(gaussian_images, dog_images, num_intervals, sigma, image_border_width)

    keypoints_with_orientations = []
    for keypoint in keypoints:
        octave_index = (keypoint.octave & 255)  # Extract octave index from keypoint
        localized_image_index = (keypoint.octave >> 8) & 255  # Extract image index from keypoint
        gaussian_image = gaussian_images[octave_index][localized_image_index]
        oriented_keypoints = sift.KeypointsWithOrientations(keypoint, octave_index, gaussian_image)
        keypoints_with_orientations.extend(oriented_keypoints)


    orientations = [keypoint.angle for keypoint in keypoints_with_orientations]

    # Calcula el histograma de orientaciones
    hist, bins = np.histogram(orientations, bins=36, range=(0, 360))  # Puedes ajustar el número de bins según tu preferencia
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Crea el gráfico del histograma de orientaciones
    plt.figure(figsize=(8, 6))
    plt.bar(bin_centers, hist, width=10, align='center', color='blue')
    plt.title('Histograma de Orientaciones de Keypoints')
    plt.xlabel('Orientación (grados)')
    plt.ylabel('Frecuencia')
    plt.xlim(0, 360)
    plt.xticks(np.arange(0, 361, 45))  # Personaliza los ticks del eje x si es necesario
    plt.grid(True)

    # Muestra el gráfico
    plt.show()

    '''Identificacion de puntos clave'''
    keypoints = sift.removeDuplicateKeypoints(keypoints)
    keypoints = sift.convertKeypointsToInputImageSize(keypoints)

    keypoint_image = cv2.drawKeypoints(img1, keypoints, None, color=(255,0,0))

    cv2.imshow('Keypoints', keypoint_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    '''Pareo de puntos clave'''
    descriptors = sift.generateDescriptors(keypoints, gaussian_images)

    #Para la imagen 2
    image2 = img2.astype('float32')
    base_image2 = sift.BaseImage(image2, sigma, assumed_blur)
    num_octaves2 = sift.NumberOfOctaves(base_image2.shape)
    gaussian_kernels2 = sift.GaussianKernels(sigma, num_intervals)
    gaussian_images2 = sift.GaussianImages(base_image2, num_octaves, gaussian_kernels)
    dog_images2 = sift.DoGImages(gaussian_images2)
    keypoints2 = sift.ScaleSpaceExtrema(gaussian_images2, dog_images2, num_intervals, sigma, image_border_width)
    keypoints2 = sift.removeDuplicateKeypoints(keypoints2)
    keypoints2 = sift.convertKeypointsToInputImageSize(keypoints2)
    descriptors2 = sift.generateDescriptors(keypoints2, gaussian_images2)

    MIN_MATCH_COUNT = 10
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors, descriptors2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        # Estimar la homografía entre plantilla y escena.
        src_pts = np.float32([ keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([ keypoints2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)[0]

        # Dibujar plantilla detectada en la imagen de la escena
        h, w = img1.shape
        pts = np.float32([[0, 0],
                        [0, h - 1],
                        [w - 1, h - 1],
                        [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

        h1, w1 = img1.shape
        h2, w2 = img2.shape
        nWidth = w1 + w2
        nHeight = max(h1, h2)
        hdif = int((h2 - h1) / 2)
        newimg = np.zeros((nHeight, nWidth, 3), np.uint8)

        for i in range(3):
            newimg[hdif:hdif + h1, :w1, i] = img1
            newimg[:h2, w1:w1 + w2, i] = img2

        # Dibujar coincidencias de puntos clave SIFT
        for m in good:
            pt1 = (int(keypoints[m.queryIdx].pt[0]), int(keypoints[m.queryIdx].pt[1] + hdif))
            pt2 = (int(keypoints2[m.trainIdx].pt[0] + w1), int(keypoints2[m.trainIdx].pt[1]))
            cv2.line(newimg, pt1, pt2, (255, 0, 0))

        plt.imshow(newimg)
        plt.show()
    else:
        print("No se encuentran suficientes coincidencias - %d/%d" % (len(good), MIN_MATCH_COUNT))
