# -*- coding: utf-8 -*-
"""example_fundamental.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/10A2s9wI9PHOSlWpkcJUHbSskpR5Qz8V0

# Examplo de calculo da matriz fundamental a partir de pontos de correspondência entre duas imagens
"""

# imports

from google.colab.patches import cv2_imshow
import numpy as np
import cv2
from matplotlib import pyplot as plt

#
#  Rotinas para achar e plotar linhas epipolares
#

def findEpipolarLines(F, pts):
	#
	#  Acha linhas epipolares F*pts
	#

  # Cria matriz com pontos (em coordenadas homogeneas e faz multiplicacao matricial)
	xx = np.concatenate([pts1.T, np.ones( (1,len(pts1)))], axis = 0)
	lines = F@xx
	return lines

def desenhaLinhasEpipolares(img1, img2, lines, pts1, pts2):
    #
    #  Desenha pontos pts1 na imagem img1 e pontos correspondetes pts2
    #  na imagem img2, juntamente com as lihas epipolares
    #
    c = img1.shape[1] # numero de colunas na imagem

  	# Converte pra grayscale
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)

    #
    #  Varre pontos em correspondencia e plota linhas epipolares
    #
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0,255,3).tolist()) # cria cor aleatoria para mostrar ponto
        #
        #  Coordenadas inicial e final do ponto de cada linha epipolar (na primeira e ultima coluna)
        #
        x0, y0 = map(int, [0, -r[2]/r[1] ])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        #
        # Plota a linha epipolar na primeira imagem
        #
#        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img2 = cv2.line(img2, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1, img2

#
# Carrega arquivo com pares de imagens (da base Middlebury)
#
!wget https://www.inf.ufrgs.br/~crjung/fvc/pratica2/stereo_images.zip
!unzip stereo_images
#
#  Carrega a imagem de um plot 2D com eixos coordenados
#
img1 = cv2.imread('stereo_images//tsukuba1.ppm',0)
img2 = cv2.imread('stereo_images//tsukuba5.ppm',0)

print('Imagem 1')
cv2_imshow(img1)
print('Imagem 2')
cv2_imshow(img2)

"""Vamos primeiramente achar pontos de correspondência, e depois estimar a matriz fundamental"""

#
#  Realiza casamento de pontos usando o descritor ORB
#


# inicializa descritor (ORB) para achar correspondencias
features = cv2.ORB_create(nfeatures=1000)

# Acha os keypoints e os descritores em cada imagem
kp1, des1 = features.detectAndCompute(img1, None)
kp2, des2 = features.detectAndCompute(img2, None)

# Casamento dos descritores por forca bruta
matcher = cv2.BFMatcher()
matches = list(matcher.match(des1, des2))

#
print('Pontos casados: %d' % len(matches))

#  ordena casamento por distancia em ordem crescente
matches.sort(key = lambda matches: matches.distance)


#
#  Pega os num_match melhores casamentos - voce pode alterar esse parametro
#
num_match = 100
pts1 = []
pts2 = []
for m in matches[0:num_match]:
  pts2.append(kp2[m.trainIdx].pt)
  pts1.append(kp1[m.queryIdx].pt)

# Converte para array
pts1 = np.array(pts1)
pts2 = np.array(pts2)

#
#  Calcula matriz fundamental com RANSAC (você pode implementar sua propria versao do 8-PA)
#
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

# Seleciona apenas inliers usados pelo algoritmo
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]

"""Finalmente, acharemos as linhas epipolares nos pontos de correspondencia considerados inliers e plotaremos na segunda imagem."""

#
#  Acha linhas epipolares para os inliers
#


lines = findEpipolarLines(F, pts1).T

# Converte para inteiro para plotar nas imagens
pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
out1, out2 = desenhaLinhasEpipolares(img2, img1, lines, pts2, pts1)

#
#  Mostra os resultados, lado a lado
#

out = np.hstack([out1, out2])
print('Vista de referencia (esquerda) e segunda vista com linhas epipolares')
cv2_imshow(out)

"""Observe que o par de imagens já é retificado, e se espera ter lihas epipolares horizontais e colineares.  De fato, vamos olhar a matriz fundamental com mais cuidado."""

print(F/np.max(F)) #lembre que a matriz fundamental não é afetada por escalares, fazemos a normalização pelo máximo para facilitar a análise