import numpy as np
import matplotlib.pyplot as plt
import cv2



#Sitraka

def griser(filename):
    image = cv2.imread(filename)
    b,v,r = cv2.split(image) #recuperation 3 matrice R V et B

    y = 0.299*r + 0.587*v + 0.114*b #utilisation du formule de luminance Y

    image_grise = y.astype(np.uint8)

    return image_grise

def somme(histo,debut,fin):
        somme = 0
        for i in range(debut,fin):
            somme = somme + histo[i]
        return somme

def somme2(histo,debut,fin):
        somme = 0
        for i in range(debut,fin):
            for j in range(debut,fin):
                somme = somme + histo[i][j]
        return somme

def moyenne(histo,debut,fin):
        somme = 0
        for i in range(debut,fin):
            somme = somme + (i*histo[i])
        return somme

def otsu(image):
    histo = np.zeros(256, int)
    for i in range(0,image.shape[0]):
        for j in range(0,image.shape[1]):
            histo[image[i,j]] = histo[image[i,j]] + 1
    plt.plot(histo)
    plt.show()

    hc = np.zeros(256, int)
    hc[0] = histo[0]
    for i in range(1,256):
        hc[i] = histo[i] + hc[i-1]

    threshold = 0
    i = 0
    pAr = 0
    mAr = 0
    pAv = 0
    mAv = 0

    varinter = []



    for i in range(0,len(histo)-1):

        pAr = somme(histo, 0, i) / hc[i]
        mAr = moyenne(histo,0,i) / hc[i]
        defimage2 = somme(histo,1,len(histo)-1)

        pAv = somme(histo, i, len(histo)-1) / defimage2
        mAv = moyenne(histo,i,len(histo)-1) / defimage2
        varinter.append(pAr * pAv * (mAr-mAv) * (mAr-mAv))


    matrice  = np.array(varinter)
    for i in range(0,len(matrice)):
         minim = matrice[matrice > 0].min()
         if matrice[i] == minim:
                threshold = i
    return(threshold)


def binarisation(image,seuil,imbinaire):
    # imbinaire = self.image.copy()
    for i in range(0,image.shape[0]):
                for j in range(0,image.shape[1]):
                    if image[i][j]< seuil :
                        imbinaire[i][j] = 0
                    else:
                        imbinaire[i][j] = 255
    return imbinaire


def lisse(image,image_lisse):
    new_voisinage = [[0,0,0],[0,0,0],[0,0,0]]
    for i in range (1,image.shape[0] - 1 ):
        for j  in range (1,image.shape[1] - 1):
            voisinage = image[i-1:i+2,j-1:j+2]
            for i in range(0,len(voisinage)):
                for j in range(0,len(voisinage)):
                    new_voisinage[i][j] = voisinage[i][j] / 9
            pix = somme2(new_voisinage,0,len(new_voisinage))

            image_lisse[i,j] = pix
    return image_lisse


def getLigne(window):
    ligne = window - 100

    return(ligne)

def getColonne(image,window)    :
    colonne = ((window - 100 ) * image.shape[0]) / image.shape[1]

    return(colonne)



def luminosite(image,intensite):
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)
    lim = 255 - intensite
    v[v>lim] =255
    v[v<=lim] += intensite
    final_hsv = cv2.merge((h,s,v))
    image =cv2.cvtColor(final_hsv,cv2.COLOR_HSV2BGR)
    return(image)

#Rojo


def egalisationHisto(y):
    y = y.astype(np.uint8)

    histo = np.zeros(256, int)
    for i in range(0,y.shape[0]):
        for j in range(0,y.shape[1]):
            histo[y[i,j]] = histo[y[i,j]] + 1

    # calcul l'histogramme cumulé hc

    hc = np.zeros(256, int)
    hc[0] = histo[0]
    for i in range(1,256):
        hc[i] = histo[i] + hc[i-1]

    #egalisation histogramme

    nbpixels = y.size
    hc = hc / nbpixels * 255
    for i in range(0,y.shape[0]):
        for j in range(0,y.shape[1]):
            y[i,j] = hc[y[i,j]]
    return y

def inversionCouleur(image):
    y = image.astype(np.uint8)
    for i in range(0,image.shape[0]):
                for j in range(0,image.shape[1]):
                    image[i][j] = 255 - y[i][j]
    return(image)


def etale(image):
    min = 50
    max = 60
    y = image.astype(np.uint8)
    for i in range(0,image.shape[0]):
        for j in range(0,image.shape[1]):
            image[i][j] = (255*(y[i][j]-min)) / (max-min)
            if image[i][j] < min:
               image[i][j] = min
            elif image[i][j] > max:
               image[i][j] = max
            else:
               image[i][j] = image[i][j]
    return(image)




def floutter(image,value):
    
    kernel_size = (value+1 ,value+1)
    image_blur = cv2.blur(image,kernel_size)
    return(image_blur)


#Andry

def erosion(image,image_erode):
    for i in range (1,image.shape[0] - 1 ):
        for j  in range (1,image.shape[1] - 1):
            voisinage = image[i-1:i+2,j-1:j+2]
            image_erode[i,j] = np.amin(voisinage)
    return image_erode

def dilatation(image,ligne,colonne,image_erode):
    for i in range (1,ligne - 1 ):
        for j  in range (1,colonne - 1):
            voisinage = image[i-1:i+2,j-1:j+2]
            image_erode[i,j] = np.amax(voisinage)
    return image_erode


def conserver(image,image_conserver):
    for i in range (1,image.shape[0] - 1 ):
        for j  in range (1,image.shape[1] - 1):
            voisinage = image[i-1:i+2,j-1:j+2]
            if image[i,j] > np.amax(voisinage):
                image_conserver[i,j] = np.amax(voisinage)
            elif image[i,j] < np.amin(voisinage):
                image_conserver[i,j] = np.amin(voisinage)
            else:
                image_conserver[i,j] = image[i,j]
    return image_conserver

def rotation90(image):
    image_rotate = np.zeros((image.shape[1],image.shape[0],3), np.uint8)
    for i in range(0,image_rotate.shape[0]):
        for j in range(0,image_rotate.shape[1]):
          image_rotate[i][j] = image[image.shape[0] - 1 - j][i]
    return(image_rotate)



def translation(image):
    rows = image.shape[0]
    cols = image.shape[1]
    M = np.float32([[1,0,100],[0,1,100]])
    image_translation = cv2.warpAffine(image,M,(cols,rows))
    
    return(image_translation)

