import numpy as np
from scipy.io.wavfile import read
import random

signal = [2, 3, 5, 9, 18, 62, 53, 1, 23, 5, 4, 6, 5, 66, 45, 100, -100]
sig2 = [1, 1, 0, 0, 1, 1, 0, 0, 1, 1]


def norm(signal):
    #Recherche de l'échantillon max et min
    maximum = max(signal)
    minimum = min(signal)
    
    #Cas trivial : signal nul
    if (maximum == minimum == 0):
        normsignal = signal
    
    #Calcul du signal normalisé
    else:
        if (maximum > abs(minimum)):
            normsignal = [float(i) / max(signal) for i in signal]

        else:
            normsignal = [float(i) / -min(signal) for i in signal]

    return normsignal


def split(signal, fe, Twidth, Tstep):  # fe = nbr of point took every second : 1s -> fe samples <=> Nsamples = duration * fe

    #Tests des triviaux
    if (Twidth == Tstep == 0):
        print("la fenetre est de largeur 0 et le saut est de 0")
        return
    else:
        if (Twidth == 0):
            print("la fenetre est de largeur 0")
            return
        #Overlap condition
        else:
            if (Twidth > Tstep):
                print("La condition de non-recouvrement est faussse")
                return

    if (fe == 0):
        print("la fréquence échantillon est nulle")
        return

    #Calcule de Nwidth & Nstep (passade du domaine temporel au domaine discret)
    Nwidth = Twidth * fe  #Nombre de points par frame
    Nstep = Tstep * fe  #Nombre de points par step

    splitlist = [] #Liste qui contiendra les différentes frames
    
    #Variables de calcul
    frame = [] #Liste temporaire qui contient les échantillons d'une frame à la fois
    diff = Nstep - Nwidth #Nombre d'échantillons de différence entre une frame et la suivante
    i = 0 #Compteur : indice de l'échantillon
    temp = Nwidth #Utilité : ne pas modifier Nwidth

    while i < len(signal): #Parcours des échantillons du signal
        #Nouvelle fenêtre
        if (i != 0 and i % temp == 0): #Si l'échantillon n'est pas le premier et si il correspond au denrier échantillon d'une fenêtre
            splitlist.append(frame) #Ajout d'une fenêtre à la splitlist
            frame = [] #frame estvidée pour commencer une nouvelle denêtre
            i += diff #Décallage pour commencer la nouvelle fenêtre en prenant compte le step
            #Vérifie si le step ne va pas sortir du signal (évite l'erreur : index out of range)
            if (temp + Nstep <= len(signal)):
                temp += Nstep #Décallage de la finde la nouvelle fenêtre pour prendre en compte le step
            #Si le step va plus loin que la fin des échantillons du signal
            else:
                return splitlist
        #Frame se remplit au fur et à mesure tant qu'on reste dans la même frame
        frame.append(signal[i])
        #Incrémentation de l'index
        i += 1
    #La dernière frame ne remplit jamais les condition du if qui permet d'ajouter la frame
    #Cependant les informations sont quand même dans la liste frame
    #On ajoute cette dernère frame à la fin de la boucle
    splitlist.append(frame)

    return splitlist


def energy(signal):
    energy = 0 #Déclaration à 0 (valeur neutre de la somme)
    for value in signal:
        energy += value ** 2
    return energy


def randomfichier():
    #Génération d'un nombre de 001-593 (différent de 1-593)
    unite = random.randint(0, 9)
    centaine = random.randint(0, 5)
    if (centaine == 5):
        dizaine = random.randint(0, 3)
    else:
        dizaine = random.randint(0, 9)
    #Concaténation des varaibles pour générer le nom correct du fichier
    nomfichier = "arctic_b0%i%i%i.wav" % (centaine, dizaine, unite)
    return nomfichier


def frameEnergy(splitlist):
    #Liste de langueur égale au nombre de frames
    B = np.zeros(len(splitlist))
    #Calcul des energies de plusieurs frames
    for frame in range(len(splitlist)):
        B[frame] = energy(splitlist[frame])

    return B


# sign= [1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
# splt = split(sign, 1, 2, 4)
# print(splt)


utterance = read("cmu_us_ksp_arctic/wav/%s" % (randomfichier()))
fetot = utterance[0]
fe = int(fetot / 1000)
print(fe)
# print(utterance)
sig = utterance[1]
signal = sig.tolist()
# print(signal)
normsignal = norm(signal)
# print(normsignal)
Twidth = 10
Tstep = 100
splitsignal = split(normsignal, fe, Twidth, Tstep)
# print(splitsignal)

EnergySignal = frameEnergy(splitsignal)

print(EnergySignal)
