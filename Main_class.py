import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

"""
    _ Classe Algorihtme blablabla

"""

class Algorithme :

    test_realise = 1

    def __init__(self,img) :
        #self.image = cv.cvtColor(cv.imread(img),cv.COLOR_BGR2RGB) #Sans masque manuel
        self.image =  self.Manual_Mask(cv.cvtColor(cv.imread(img),cv.COLOR_BGR2RGB))
        self.image_init = self.Manual_Mask(cv.cvtColor(cv.imread(img),cv.COLOR_BGR2RGB))

    """
        _ La méthode Affiche() permet d'afficher horizontalement les images entrees en parametre sur une seule figure
        _ Conseil : 5 max

    """

    def Affiche(localVar,*img,marche=False,color=None) :
        NBR_IMAGE = len(img)
        plt.figure()
        n=1
        while n<=NBR_IMAGE :
            SB = 100 + NBR_IMAGE*10 + n
            plt.subplot(SB)
            plt.imshow(img[n-1],color)
            for name,value in localVar :
                if value is img[n-1] :
                    plt.title(name)
            plt.show(block=False)
            n += 1
        plt.show(block=marche)

    """
        _ La méthode Affiche() permet d'afficher horizontalement les histogrammes des images entrees en parametre sur une seule figure
        _ les images sont entrees en tant que tuple avec (image,MIN,MAX) sachant que MIN et MAX forment l'intervalle d'affichage pour l'histogramme

    """

    def Histogramme(localVar,*img,marche=False) :
        NBR_IMAGE = len(img)
        plt.figure()
        n=1
        while n<=NBR_IMAGE :
            MIN = img[n-1][1]
            MAX = img[n-1][2]
            SB = 100 + NBR_IMAGE*10 + n
            plt.subplot(SB)
            plt.hist(img[n-1][0].ravel(), 255, [MIN, MAX])
            for name,value in localVar :
                if value is img[n-1][0] :
                    plt.title("Hist_" + name)
            plt.show(block=False)
            n += 1
        plt.show(block=marche)

    def comptagePixels(self,img) :
        compteur = img[(img[...,0] != 255)&(img[...,1] != 0)&((img[...,2] != 0))]
        taux = (len(compteur)/(img.shape[0]*img.shape[1]))*100
        for name,value in globals().items() :
            if value is type(self) :
                print("Test n°{}, {} : proportion de feuilles dans l'image : {} % ".format(self.test_realise,name,round(taux,1)))  
        
    def Manual_Mask(self,img):
        mask = np.full_like(img, 1) #préparer une image de la même taille que l'image source entièrement composé de 1
        cv.circle(mask, (2000, 2000), 500, (0, 0, 0), thickness=-1) #Création d'un cercle aux positions x=2250 et y=1700, de radius 500 et de couleur blanche (0,0,0) remplie.
        return img * mask #application du mask

"""
    _ Classe algorithme des bassins versants

"""
class Algo_Watershed(Algorithme) :

    def __init__(self,img,EL_ST,IMG_BIN=None) :
        Algorithme.__init__(self,img,)
        self.ELM_FOREGROUND = EL_ST
        self.config_watershed(IMG_BIN)
        self.comptagePixels(self.image)
        Algorithme.test_realise += 1     
        
    def config_watershed(self,img_BIN) :
        #variables locales
        image_init = self.image_init
        image_final = self.image
        if img_BIN is None :         
            ret3, imgToFlood = cv.threshold(cv.cvtColor(self.image,cv.COLOR_RGB2GRAY),0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        else :
            imgToFlood = img_BIN
        # régions certaines d'arrière plan en noir
        kernel_0 = cv.getStructuringElement(cv.MORPH_ELLIPSE,self.ELM_FOREGROUND) 
        background = cv.morphologyEx(imgToFlood.astype(np.uint8), cv.MORPH_DILATE, kernel_0)
        # régions certaines de premier plan en blanc
        foreground = cv.morphologyEx(imgToFlood.astype(np.uint8), cv.MORPH_ERODE, kernel_0)
        # Régions incertaines 
        unknown = cv.subtract(background,foreground)
        # Étiquetage des marqueurs
        ret, markers = cv.connectedComponents(foreground)
        # Ajoute 1 à tous les marqueurs pour que l'arrière-plan soit à 1
        markers = markers + 1
        # Régions inconnues à 0
        markers[unknown==255] = 0
        # Marquage
        markers = np.uint8(markers)
        marqueurs =  cv.applyColorMap(markers,cv.COLORMAP_JET)
        # Innondation
        markers = np.int32(markers)
        watershed = cv.watershed(self.image,markers)
        # Résultat visualisable 
        image_final[((watershed == 1) | (watershed == -1))] = [255,0,0] # & ou | (pas and et or)
        # Affichage
        localVar = locals().items()
        # Algorithme.Affiche(localVar,imgToFlood,color="gray")
        # Algorithme.Affiche(localVar,background,foreground,unknown,color="gray")
        # Algorithme.Affiche(localVar,marqueurs,watershed)
        Algorithme.Affiche(localVar,image_init,image_final, marche = True)     

"""
    _ Classe seuillage espace HSV

"""

class Algo_HSV(Algorithme) :

    def __init__(self,img,T_min,T_max,EL_OP) :
        Algorithme.__init__(self,img)
        self.Threshold_min = T_min
        self.Threshold_max = T_max
        self.ELM_OPEN = EL_OP
        self.config_HSV()
        self.comptagePixels(self.image)
        Algorithme.test_realise += 1    

    def config_HSV(self) :
        #variables locales
        image_init = self.image_init
        image_final = self.image
        # Changement d'espace HSV
        hsv = cv.cvtColor(self.image,cv.COLOR_RGB2HSV)
        hue,sat,val = cv.split(hsv)
        # Seuillage du canal H et S
        lower = np.array(self.Threshold_min,dtype=np.uint8) 
        upper = np.array(self.Threshold_max,dtype=np.uint8) 
        hsv_T = cv.inRange(hsv,lower,upper)
        
        
        # Ouverture morphologique 
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,self.ELM_OPEN) 
        opening = cv.morphologyEx(hsv_T.astype(np.uint8), cv.MORPH_OPEN, kernel)


        # Résultat
        image_final[opening == 0] = [255,0,0]
        # Affichage
        localVar = locals().items() # Liste des varibales locales de la méthode actuelle
        # Algorithme.Histogramme(localVar,(hue,0,50),(sat,0,255),marche=True)
        # Algorithme.Affiche(localVar,hue,sat,val,color="gray",marche=True)
        # Algorithme.Affiche(localVar,hsv_T,opening,color="gray")
        Algorithme.Affiche(localVar,hsv_T,opening,color="gray",marche = True)
        return opening
        
class Algo_Seuil(Algorithme) :

    def __init__(self,img,s) :
        Algorithme.__init__(self,img)
        self.seuil = s
        self.config_Seuil()
        self.comptagePixels(self.image)
        #self.bht(self,hist, min_count)
        Algorithme.test_realise += 1    

    #seuil automatique par la méthode BHT.
    def bht(self, hist, min_count: int = 5):
        """Balanced histogram thresholding."""
        n_bins = len(hist)
        h_s = 0

        #Tant que la valeur de teinte est inférieur à "min_count", h_s + 1 -> Cela donne la valeur de teinte partie inférieur.
        while hist[h_s] < min_count:
            h_s += 1
        

        #h_e = 255
        h_e = n_bins - 1

        #Tant que la valeur de teinte est inférieur à "min_count", h_e - 1 -> Cela donne la valeur de teinte partie supérieur.
        while hist[h_e] < min_count:
            h_e -= 1


        #Création d'un tableau de dimension 1 pour stocker le contenu de l'histogramme.
        table = np.zeros((256))
        for i in range(256):
            table[i] = hist[i][0]
        
        #Moyenne des valeurs inférieure et supérieure. - MÉTHODE 1
        #h_c = (h_s + h_e) / 2

        #Moyenne des valeurs inférieure et supérieure. - MÉTHODE 2
        #Arrondie de la mise en moyenne du tableau pondéré par les valeurs de tables.
        h_c = int(round(np.average(np.linspace(0, 2 ** 8 - 1, n_bins),weights=table)))

        #Somme des poids de la partie de gauche jusqu'à la teinte inférieur et supérieur.
        w_l = np.sum(hist[int(h_s):int(h_c)])
        w_r = np.sum(hist[int(h_c): int(h_e + 1)])

        while h_s < h_e:
            if w_l > w_r:  # left part became heavier
                w_l -= hist[h_s]
                h_s += 1
            else:  # right part became heavier
                w_r -= hist[h_e]
                h_e -= 1
            new_c = int(round((h_e + h_s) / 2))  # re-center the weighing scale

            if new_c < h_c:  # move bin to the other side
                w_l -= hist[h_c]
                w_r += hist[h_c]
            elif new_c > h_c:
                w_l += hist[h_c]
                w_r -= hist[h_c]

            h_c = new_c

        return h_c

    def config_Seuil(self) :
        img = self.image #Commenter pour ne pas appliquer le masque manuel
        image_init = self.image_init

        #test automatic threshold
        hist = cv.calcHist(image_init,[1],None,[256],[0,256])
        hist = hist.astype(int)

        
        b,g,r = cv.split(image_init)

        
        #Visualisation du canal vert de l'image
        #localVar = locals().items() # Liste des varibales locales de la méthode actuelle
        #Algorithme.Affiche(localVar,g,color="gray",marche = True) 
        
        #visualisation de l'histogramme du canal vert de l'image
        color = ('b','g','r')
        for i,col in enumerate(color):
            histr = cv.calcHist([img],[i],None,[256],[0,256])
            plt.plot(histr,color = col)
            plt.xlim([0,256])
        plt.axvline(101, color='orange') #afficher le treshold sur le graphe. 
        plt.title("Histogramme N1_13_11_2020_PE0.png")
        plt.show()
        
        


        #print(hist[0])
        print(self.bht(hist, 50))
        #print(type(hist[0]))
        

        
        #hist = cv.calcHist(image_init,[1],None,[256],[0,256])

        self.seuil = 116

        img[img[...,1]+100-img[...,0] < self.seuil] = [255,0,0] #Canal vert - Canal rouge et seuil à 120, fond mis à la couleur rouge.
        image_init[image_init[...,1]+100-image_init[...,0] < self.seuil] = [255,0,0] #Canal vert - Canal rouge et seuil à 120, fond mis à la couleur rouge.

        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5)) #Construction de l'opérateur pour la morphologie Ellipse
        opening = cv.morphologyEx(img[...,1].astype(np.uint8), cv.MORPH_OPEN, kernel) #Application d'une ouverture à l'image filtrée    
        
        img[opening == 0] = [255,0,0]
        
        
        #localVar = locals().items() # Liste des varibales locales de la méthode actuelle
        #Algorithme.Affiche(localVar,img,color="gray",marche = True) 
        return opening


SEUIL_MIN_HSV = [70/2,120,0],
SEUIL_MAX_HSV = [90/2,255,255]
TAILLE_OPENING = (12,12)
#Test1 = Algo_HSV("N1_13_11_2020_PE0.png",SEUIL_MIN_HSV,SEUIL_MAX_HSV,TAILLE_OPENING)
#Test2 = Algo_Seuil("N1_13_11_2020_PE0.png",120) #116

Test2 = Algo_Seuil("N1_13_11_2020_PE0.png", 139)

#Test2 = Algo_Seuil("Stade_12_1.10.png",120)


#TAILLE_FG = (5,5) 
#Test3 = Algo_Watershed("N1_13_11_2020_PE0.png",TAILLE_FG,Test2.config_Seuil())
#Test3 = Algo_Watershed("N1_13_11_2020_PE0.png",TAILLE_FG,Test1.config_HSV())




"""
TAILLE_FG = (5,5) 
Test2 = Algo_Watershed("N1_13_11_2020_PE0.png",TAILLE_OPENING,Test1.config_HSV())

Test25 = Algo_Watershed("Stade_12_1.10.png",SEUIL_MIN_HSV,SEUIL_MAX_HSV,TAILLE_OPENING)
SEUIL_MIN_HSV = [74/2,90,0]
SEUIL_MAX_HSV = [100/2,255,255]
TAILLE_OPENING = (5,5) # nettoyage
TAILLE_FG = (10,10)
Test3 = Algo_HSV("Stade_13_1.10.png",SEUIL_MIN_HSV,SEUIL_MAX_HSV,TAILLE_OPENING)

SEUIL_MIN_HSV = [74/2,30,0]
SEUIL_MAX_HSV = [120/2,255,255]
TAILLE_OPENING = (15,15) # nettoyage
Test4 = Algo_HSV("cylindre.png",SEUIL_MIN_HSV,SEUIL_MAX_HSV,TAILLE_OPENING)"""