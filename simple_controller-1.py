"""camera_pid controller."""

from controller import Display, Keyboard, Robot, Camera
from vehicle import Car, Driver
import numpy as np
import cv2
from datetime import datetime
import math
import os

#Getting image from /Applications/Webots/webots --version):
def get_image(camera):
    raw_image = camera.getImage()  
    image = np.frombuffer(raw_image, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4)
    )
    return image

#Boder detection using Canny Edge Detection
def canny_edge_detection(image):
    """
    Aplica el algoritmo de detección de bordes de Canny a la imagen.
    
    Args:
        image: Imagen de entrada.
    
    Returns:
        Imagen procesada con Canny Edge Detection.
    """
    # Convertir la imagen a escala de grises
    img_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Aplicar el filtro de Canny
    img_canny = cv2.Canny(img_grey, 50, 150)

    return img_canny

#regiion of interest
def apply_region_of_interest(img_grey, img_canny):
    """
    Aplica una región de interés (ROI) a la imagen procesada.
    
    Args:
        img_grey: Imagen en escala de grises.
        img_canny: Imagen procesada con Canny Edge Detection.
    
    Returns:
        Imagen con la máscara aplicada.
    """
    # Obtener las dimensiones de la imagen
    height, width = img_grey.shape

    # Definir los vértices del trapezoide dinámico
    vertices = np.array([[
        (width * 0.1, height),          # Esquina inferior izquierda
        (width * 0.4, height * 0.6),   # Punto superior izquierdo
        (width * 0.6, height * 0.6),   # Punto superior derecho
        (width * 0.9, height)          # Esquina inferior derecha
    ]], dtype=np.int32)

    # Crear una máscara negra del mismo tamaño que la imagen
    img_roi = np.zeros_like(img_grey)

    # Rellenar el polígono definido por los vértices con blanco
    cv2.fillPoly(img_roi, vertices, 255)

    # Aplicar la máscara a la imagen procesada con Canny
    img_mask = cv2.bitwise_and(img_canny, img_roi)

    return img_mask

# Transformada de Hough
def hough_transform(img_mask):
    """
    Aplica la transformada de Hough para detectar líneas en la imagen.
    """
    # Distancia en píxeles entre líneas paralelas detectadas
    rho = 1  # Se elige 1 para una resolución precisa en la detección de líneas.

    # Resolución angular en radianes
    theta = np.pi / 180  # Se usa 1 grado (en radianes) para un equilibrio entre precisión y rendimiento.

    # Umbral mínimo de votos para considerar una línea
    threshold = 10  # Se elige un valor bajo para detectar líneas incluso con pocos votos.

    # Longitud mínima de una línea para ser aceptada
    min_line_len = 10  # Se establece un valor bajo para detectar líneas cortas.

    # Máxima distancia entre segmentos de línea para unirlos
    max_line_gap = 20  # Se permite una separación moderada para unir segmentos cercanos.

    # Aplicar la transformada de Hough probabilística
    lines = cv2.HoughLinesP(
        img_mask, 
        rho, 
        theta, 
        threshold, 
        np.array([]), 
        minLineLength=min_line_len, 
        maxLineGap=max_line_gap
    )

    # Verificar si se detectaron líneas
    if lines is None:
        print("No se detectaron líneas en la imagen.")
        return None

    # Imprimir la cantidad de líneas detectadas
    print(f"Líneas detectadas: {len(lines)}")
    return lines

#Angulo de direccion Aplicado al vehiculo
def calculate_steering_angle(lines, image_width, image_height):
    """
    Calcula el ángulo de dirección basado en las líneas detectadas.
    """
    if lines is None:
        print("No se detectaron líneas. Conduciendo recto.")
        return 0.0  # Ángulo por omisión (recto)

    left_lines = []
    right_lines = []

    # Clasificar líneas en izquierda y derecha
    for line in lines:
        x1, y1, x2, y2 = line[0]
        slope = (y2 - y1) / (x2 - x1)  # Pendiente de la línea
        intercept = y1 - slope * x1  # Intersección con el eje y

        # Filtrar pendientes extremas (líneas casi verticales u horizontales)
        if abs(slope) < 0.5 or abs(slope) > 2.0:
            # Razón: Ignorar líneas con pendientes muy pequeñas (casi horizontales)
            # o muy grandes (casi verticales) ya que no son útiles para la detección de carriles.
            continue

        if slope < 0:  # Línea izquierda
            left_lines.append((slope, intercept))
        else:  # Línea derecha
            right_lines.append((slope, intercept))

    # Promediar las líneas izquierda y derecha
    left_lane = np.mean(left_lines, axis=0) if left_lines else None  # Promedio de líneas izquierdas
    right_lane = np.mean(right_lines, axis=0) if right_lines else None  # Promedio de líneas derechas

    # Calcular el punto medio entre las líneas
    if left_lane is not None and right_lane is not None:
        # Si ambas líneas están presentes, calcular el punto medio entre ellas
        left_x = (image_height - left_lane[1]) / left_lane[0]  # Punto x donde la línea izquierda cruza la base
        right_x = (image_height - right_lane[1]) / right_lane[0]  # Punto x donde la línea derecha cruza la base
        mid_x = (left_x + right_x) / 2  # Punto medio entre las dos líneas
    elif left_lane is not None:
        # Si solo está presente la línea izquierda, usarla para calcular el punto medio
        mid_x = (image_height - left_lane[1]) / left_lane[0]
    elif right_lane is not None:
        # Si solo está presente la línea derecha, usarla para calcular el punto medio
        mid_x = (image_height - right_lane[1]) / right_lane[0]
    else:
        print("No se detectaron líneas válidas. Conduciendo recto.")
        return 0.0  # Conducir recto si no se detectan líneas válidas

    # Calcular el ángulo de dirección
    center_x = image_width / 2  # Centro de la imagen (posición ideal del vehículo)
    offset = mid_x - center_x  # Desplazamiento del punto medio respecto al centro
    angle = math.atan2(offset, image_height)  # Ángulo basado en el desplazamiento

    # Limitar el rango del ángulo
    max_angle = 0.5  # Rango máximo del ángulo en radianes
    # Razón: Limitar el ángulo para evitar giros bruscos que puedan desestabilizar el vehículo.
    angle = max(-max_angle, min(max_angle, angle))

    print(f"Ángulo calculado: {angle:.2f} rad")
    return angle


def draw_lines(image, lines):
    """
    Dibuja las líneas detectadas en la imagen y las combina con la imagen original.

    Args:
        image: Imagen original en la que se dibujarán las líneas.
        lines: Líneas detectadas por la transformada de Hough.

    Returns:
        Imagen combinada con las líneas dibujadas.
    """
    # Crear una imagen negra del mismo tamaño que la original
    img_lines = np.zeros_like(image)

    # Dibujar las líneas detectadas en la imagen negra
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Dibujar cada línea detectada con color verde y grosor de 5 píxeles
            # Razón: El color verde (0, 255, 0) es fácilmente visible sobre la mayoría de los fondos,
            # y un grosor de 5 asegura que las líneas sean claramente distinguibles.
            cv2.line(img_lines, (x1, y1), (x2, y2), (0, 255, 0), 5)

    # Combinar la imagen original con las líneas detectadas
    alpha = 0.8  # Peso de la imagen original
    # Razón: Un peso de 0.8 asegura que la imagen original sea claramente visible,
    # mientras que las líneas detectadas no la opacan completamente.

    beta = 1.0   # Peso de las líneas
    # Razón: Un peso de 1.0 para las líneas garantiza que sean completamente visibles
    # y no se vean atenuadas al combinarse con la imagen original.

    gamma = 0.0  # Valor escalar adicional
    # Razón: Un valor escalar de 0.0 asegura que no se añada brillo adicional a la imagen combinada.

    # Combinar las imágenes usando los pesos definidos
    img_lane_lines = cv2.addWeighted(image, alpha, img_lines, beta, gamma)

    return img_lane_lines

#Image processing
def greyscale_cv2(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_img

#Display image 
def display_image(display, image):
    # Image to display
    image_rgb = np.dstack((image, image,image,))
    # Display image
    image_ref = display.imageNew(
        image_rgb.tobytes(),
        Display.RGB,
        width=image_rgb.shape[1],
        height=image_rgb.shape[0],
    )
    display.imagePaste(image_ref, 0, 0, False)

#initial angle and speed 
manual_steering = 0
steering_angle = 0
angle = 0.0
speed = 15.0

# set target speed
def set_speed(kmh):
    global speed            #robot.step(50)
#update steering angle
def set_steering_angle(wheel_angle):
    global angle, steering_angle
    # Check limits of steering
    if (wheel_angle - steering_angle) > 0.1:
        wheel_angle = steering_angle + 0.1
    if (wheel_angle - steering_angle) < -0.1:
        wheel_angle = steering_angle - 0.1
    steering_angle = wheel_angle
  
    # limit range of the steering angle
    if wheel_angle > 0.5:
        wheel_angle = 0.5
    elif wheel_angle < -0.5:
        wheel_angle = -0.5
    # update steering angle
    angle = wheel_angle

#validate increment of steering angle
def change_steer_angle(inc):
    global manual_steering
    # Apply increment
    new_manual_steering = manual_steering + inc
    # Validate interval 
    if new_manual_steering <= 25.0 and new_manual_steering >= -25.0: 
        manual_steering = new_manual_steering
        set_steering_angle(manual_steering * 0.02)
    # Debugging
    if manual_steering == 0:
        print("going straight")
    else:
        turn = "left" if steering_angle < 0 else "right"
        print("turning {} rad {}".format(str(steering_angle),turn))

# main
def main():
    # Create the Robot instance.
    robot = Car()
    driver = Driver()

    # Get the time step of the current world.
    timestep = int(robot.getBasicTimeStep())

    # Create camera instance
    camera = robot.getDevice("camera")
    camera.enable(timestep)  # timestep

    # processing display
    display_img = Display("display_image")

    #create keyboard instance
    keyboard=Keyboard()
    keyboard.enable(timestep)

    while robot.step() != -1:
        # Get image from camera
        image = get_image(camera)

        # Convierte la imagen a escala de grises
        grey_image = greyscale_cv2(image)

        # Process image
        img_canny = canny_edge_detection(image)

        # Aplica la región de interés
        img_mask = apply_region_of_interest(grey_image, img_canny)
        
        # Aplica la transformada de Hough
        lines = hough_transform(img_mask)

        # Dibuja las líneas detectadas en la imagen original
        image_with_lines = draw_lines(image.copy(), lines)

        # Calcula el ángulo de dirección
        angle = calculate_steering_angle(lines, camera.getWidth(), camera.getHeight())

        # Muestra la imagen con las líneas detectadas en el display
        display_image(display_img, cv2.cvtColor(image_with_lines, cv2.COLOR_BGR2GRAY))

        # Read keyboard
        key=keyboard.getKey()
        if key == keyboard.UP: #up
            set_speed(speed + 5.0)
            print("up")
        elif key == keyboard.DOWN: #down
            set_speed(speed - 5.0)
            print("down")
        elif key == keyboard.RIGHT: #right
            change_steer_angle(+1)
            print("right")
        elif key == keyboard.LEFT: #left
            change_steer_angle(-1)
            print("left")
        elif key == ord('A'):
            #filename with timestamp and saved in current directory
            current_datetime = str(datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
            file_name = current_datetime + ".png"
            print("Image taken")
            camera.saveImage(os.getcwd() + "/" + file_name, 1)
            
        #update angle and speed
        driver.setSteeringAngle(angle)
        driver.setCruisingSpeed(speed)


if __name__ == "__main__":
    main()