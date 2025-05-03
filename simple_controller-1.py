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
speed = 15

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

# Región de Interés (ROI)
def region_of_interest(image):
    # Obtener las dimensiones de la imagen
    height, width = image.shape
    # Definir un triángulo como la región de interés (ROI)
    polygons = np.array([
        [(0, height), (width, height), (width // 2, height // 2)]  # Coordenadas del triángulo
    ])
    # Crear una máscara del mismo tamaño que la imagen, inicializada en ceros
    mask = np.zeros_like(image)
    # Rellenar el triángulo en la máscara con el valor 255 (blanco)
    cv2.fillPoly(mask, polygons, 255)
    # Aplicar la máscara a la imagen original usando una operación AND bit a bit
    masked_image = cv2.bitwise_and(image, mask)
    # Devolver la imagen enmascarada
    return masked_image

# Detección de líneas usando la Transformada de Hough
def detect_lines(image):
    # Aplicar la Transformada de Hough Probabilística para detectar líneas
    lines = cv2.HoughLinesP(
        image,               # Imagen de entrada (debe ser en escala de grises)
        rho=1,               # Resolución de la acumulación en píxeles
        theta=np.pi / 180,   # Resolución angular en radianes
        threshold=50,        # Umbral mínimo de votos para considerar una línea
        minLineLength=50,    # Longitud mínima de una línea para ser detectada
        maxLineGap=10        # Máxima distancia entre segmentos para unirlos en una línea
    )
    # Devolver las líneas detectadas
    return lines

# Calcular el ángulo promedio de las líneas detectadas
def calculate_steering_angle(lines, image_width):
    # Si no se detectan líneas, devolver un ángulo de dirección de 0 (recto)
    if lines is None:
        return 0.0

    # Lista para almacenar los ángulos de las líneas detectadas
    angles = []
    for line in lines:
        # Extraer las coordenadas de los puntos de la línea
        x1, y1, x2, y2 = line[0]
        # Calcular el ángulo de la línea usando la función atan2
        angle = math.atan2(y2 - y1, x2 - x1)
        # Agregar el ángulo a la lista
        angles.append(angle)

    # Calcular el promedio de los ángulos detectados
    avg_angle = np.mean(angles)
    # Escalar el ángulo promedio al rango adecuado para el vehículo [-0.5, 0.5]
    steering_angle = avg_angle * (0.5 / (np.pi / 2))
    # Devolver el ángulo de dirección calculado
    return steering_angle

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

        # Process and display image 
        grey_image = greyscale_cv2(image)

        # Definir la Región de Interés
        roi_image = region_of_interest(grey_image)

        # Detectar líneas en la imagen
        lines = detect_lines(roi_image)

        # Calcular el ángulo de dirección basado en las líneas detectadas
        steering_angle = calculate_steering_angle(lines, grey_image.shape[1])

        # Mantener velocidad constante
        driver.setCruisingSpeed(15)

        display_image(display_img, grey_image)
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