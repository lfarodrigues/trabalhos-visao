import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob

# Calcular o lado do quadrado na imagem
def calculate_lateral_size(image):
    # Definir parâmetros
    low_threshold = 0
    pixel_value = [113,113,113]
    high_threshold = int(0.299 * pixel_value[0] + 0.587 * pixel_value[1] + 0.114 * pixel_value[2])
    
    # Aplicar a limiarização
    mask = cv2.inRange(image, low_threshold, high_threshold)
    # Encontrar contornos
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtrar contornos por área
    min_area = 20000
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    # MASK VIZUALITION
    # Criar imagem preta para preenchimento
    #filled_image = np.zeros_like(image)

    # Desenhar contornos na imagem original e preencher os contornos
    #cv2.drawContours(image, filtered_contours, -1, (0, 255, 0), 2)
    #cv2.drawContours(filled_image, filtered_contours, -1, (255, 255, 255), thickness=cv2.FILLED)

    print("Number of contours detected: " + str(len(filtered_contours)))
    areas = []
    # Calcular e exibir as áreas
    for contour in filtered_contours:
        area = cv2.contourArea(contour)
        areas.append(area)
    
    minArea = min(areas) # por que em algumas imagem o quadrado externo também é detectado

    #calculate the square lateral size in pixels
    return np.sqrt(minArea)

# Lado do triangulo no mundo real
Lr = 15.5

#//// CAMERA CALIBRATION /////
a = 9
b = 6
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((a*b,3), np.float32)
objp[:,:2] = np.mgrid[0:a,0:b].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('imgs/calibration/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (a,b), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

# Get the camera parameters matrix
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

#//// SQUARE SEGMENTATION /////
images = glob.glob('imgs/drone/*.jpg')
heights = np.array([], dtype=float)
x = np.arange(len(images))
Hi = mtx[1,2]   # from camera matrix parameters
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    Li = calculate_lateral_size(gray)
    print("Square lateral size: " + str(Li))
    # Triangle similarity
    k = Li/Lr
    Hr = Hi/k

    heights = np.append(heights, Hr)

print(len(images))

# Plotar os dados
plt.plot(heights)
plt.title("Gráfico de Alturas")
plt.xlabel("Indice")
plt.ylabel("Altura")

plt.show()

cv2.destroyAllWindows()