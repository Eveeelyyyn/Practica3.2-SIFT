# Algoritmo SIFT (Scale-Invariant Feature Transform) 📸
El algoritmo SIFT (Scale-Invariant Feature Transform) es una técnica ampliamente utilizada para la detección y descripción de características en imágenes. Su capacidad para identificar puntos clave invariantes a la escala y la orientación lo hace valioso en tareas como el emparejamiento de características y reconocimiento de objetos. 

En este trabajo de práctica se aplican y se desarrolla el método SIFT para la extracción de descriptores y también para la fase de pareo de puntos clave entre dos imágenes.


## Resultados

### Localización de Puntos Clave

En las Figuras 1 y 2, se muestran las imágenes que resaltan los puntos clave detectados por el algoritmo en relación con la imagen original. Estos puntos clave, respaldados por descriptores de 128 valores, encapsulan las características distintivas de las regiones identificadas por el robusto algoritmo SIFT.

<table>
  <tr>
    <td align="center">
      <img src="/images/Puntos Clave (Primera imagen).jpeg" alt="Resultado 1" width="400"/>
    </td>
    <td align="center">
      <img src="/images/Puntos Clave (Segunda imagen).jpeg" alt="Resultado 2" width="400"/>
    </td>
  </tr>
  <tr>
    <td align="center">
      Localización de puntos clave en la Imagen 1
    </td>
    <td align="center">
      Localización de puntos clave en la Imagen 2
    </td>
  </tr>
</table>


### Visualización de Espacio de Escala

Las Figuras 3 y 4 presentan imágenes que muestran una cuadrícula de imágenes dispuestas en filas y columnas, representando el espacio de escala. La visualización varía según la imagen de entrada y los parámetros utilizados en la generación del espacio de escala.

<table>
  <tr>
    <td align="center">
      <img src="/images/Espacio de escala (imagen 1).jpeg" alt="Resultado 1" width="400"/>
    </td>
    <td align="center">
      <img src="/images/Espacio de escala.jpeg" alt="Resultado 2" width="400"/>
    </td>
  </tr>
  <tr>
    <td align="center">
      Espacio de escala (Primera imagen)
    </td>
    <td align="center">
      Espacio de escala (Segunda imagen)
    </td>
  </tr>
</table>


### Diferencia de Gaussianos

En las Figuras 5 y 6, se presenta la diferencia de Gaussianas en las imágenes correspondientes, resaltando las características a diferentes escalas. La visualización está organizada en filas para cada octava y columnas para cada escala dentro de la octava.

<table>
  <tr>
    <td align="center">
      <img src="/images/Diferencia de Gaussiano (imagen 1).jpeg" alt="Resultado 1" width="400"/>
    </td>
    <td align="center">
      <img src="/images/Diferencia de Gausiano.jpeg" alt="Resultado 2" width="400"/>
    </td>
  </tr>
  <tr>
    <td align="center">
      Diferencia de Gaussiano (Primera imagen)
    </td>
    <td align="center">
      Diferencia de Gaussiano (Segunda imagen)
    </td>
  </tr>
</table>

### Histograma de Orientación

Las Figuras 7 y 8 representan los histogramas de orientación para la magnitud de los gradientes en cada imagen trabajada. Estos histogramas ofrecen una perspectiva clara sobre cómo se distribuyen las orientaciones de los gradientes en la imagen.

<table>
  <tr>
    <td align="center">
      <img src="/images/Histograma de orientacion  (imagen 1).jpeg" alt="Resultado 1" width="400"/>
    </td>
    <td align="center">
      <img src="/images/Histograma de orientacion.jpeg" alt="Resultado 2" width="400"/>
    </td>
  </tr>
  <tr>
    <td align="center">
      Histograma de orientación (Primera imagen)
    </td>
    <td align="center">
      Histograma de orientación (Segunda imagen)
    </td>
  </tr>
</table>

### Pareo de Puntos Clave

Finalmente, en las Figuras 9 y 10, se visualiza el pareo de puntos clave entre las dos imágenes trabajadas utilizando el algoritmo SIFT y el emparejador de fuerza bruta (BFMatcher). Las líneas conectan puntos clave coincidentes entre las dos imágenes.


<table>
  <tr>
    <td align="center">
      <img src="/images/Pareo de puntos clave (imagen 1).jpeg" alt="Resultado 1" width="400"/>
    </td>
    <td align="center">
      <img src="/images/Pareo de puntos clave.jpeg" alt="Resultado 2" width="400"/>
    </td>
  </tr>
  <tr>
    <td align="center">
      Pareo de puntos clave respecto a la primera imagen
    </td>
    <td align="center">
     Pareo de puntos clave respecto a la segunda imagen
    </td>
  </tr>
</table>


## Cómo Usar el Programa

Aquí te proporcionamos instrucciones sobre cómo utilizar nuestro programa:
1. Clona este repositorio en tu máquina local.
2. Asegúrate de tener Python y las bibliotecas necesarias instaladas.
3. Ejecuta el programa y proporciona una imagen en escala de grises como entrada.
4. El programa aplicará las técnicas de segmentación y mostrará los resultados utilizando Matplotlib.

## Autores

Este proyecto fue realizado por un equipo de estudiantes:

| [<img src="https://avatars.githubusercontent.com/u/113084234?v=4" width=115><br><sub>Aranza Michelle Gutierrez Jimenez</sub>](https://github.com/AranzaMich) |  [<img src="https://avatars.githubusercontent.com/u/113297618?v=4" width=115><br><sub>Evelyn Solano Portillo</sub>](https://github.com/Eveeelyyyn) |  [<img src="https://avatars.githubusercontent.com/u/112792541?v=4" width=115><br><sub>Marco Castelan Rosete</sub>](https://github.com/marco2220x) | [<img src="https://avatars.githubusercontent.com/u/113079687?v=4" width=115><br><sub>Daniel Vega Rodríguez</sub>](https://github.com/DanVer2002) |
| :---: | :---: | :---: | :---: |
