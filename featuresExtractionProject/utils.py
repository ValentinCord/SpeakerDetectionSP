import numpy as np
from scipy.io.wavfile import read
import random
import matplotlib.pyplot as plt
import scipy.signal as sgl
from talkbox import lpc_ref
import math


# ============Pre-Processing============
def norm(signal):
    array = np.array(signal)
    # Chercher la valeur maximale du signal en valeur absolue
    absolute = np.abs(array)
    maxima = np.max(absolute)
    # Diviser tout les échantillons par la valeur maximale
    normsignal = array / maxima
    normsignal = normsignal.tolist()
    return normsignal


def split(signal, fe, Twidth, Tstep):
    # Tests des triviaux
    if (Twidth == Tstep == 0):
        print("la fenetre est de largeur 0 et le saut est de 0")
        return 0
    else:
        if (Twidth == 0):
            print("la fenetre est de largeur 0")
            return 0
        # Overlap condition
        else:
            if (Twidth > Tstep):
                print("La condition de non-recouvrement n'est pas vérifiée")
                return 0
    if (fe == 0):
        print("la fréquence d'échantillonage est nulle")
        return 0
    if (signal == 0):
        print("le signal est nul")
        return 0

    # Calcule de Nwidth & Nstep (passade du domaine temporel au domaine discret)
    fe = int(fe / 1000)
    Nwidth = Twidth * fe  # Nombre de points par frame
    Nstep = Tstep * fe  # Nombre de points par step

    splitlist = []  # Liste qui contiendra les différentes frames

    # Variables de calcul
    frame = []  # Liste temporaire qui contient les échantillons d'une frame à la fois
    diff = Nstep - Nwidth  # Nombre d'échantillons de différence entre une frame et la suivante
    i = 0  # Compteur : indice de l'échantillon
    temp = Nwidth  # Utilité : ne pas modifier Nwidth

    while i < len(signal):  # Parcours des échantillons du signal
        # Nouvelle fenêtre
        if (
                i != 0 and i % temp == 0):  # Si l'échantillon n'est pas le premier et si il correspond au denrier échantillon d'une fenêtre
            splitlist.append(frame)  # Ajout d'une fenêtre à la splitlist
            frame = []  # frame estvidée pour commencer une nouvelle denêtre
            i += diff  # Décallage pour commencer la nouvelle fenêtre en prenant compte le step
            # Vérifie si le step ne va pas sortir du signal (évite l'erreur : index out of range)
            if (temp + Nstep <= len(signal)):
                temp += Nstep  # Décallage de la finde la nouvelle fenêtre pour prendre en compte le step
            # Si le step va plus loin que la fin des échantillons du signal
            else:
                return splitlist
        # Frame se remplit au fur et à mesure tant qu'on reste dans la même frame
        frame.append(signal[i])
        # Incrémentation de l'index
        i += 1
    # La dernière frame ne remplit jamais les condition du if qui permet d'ajouter la frame
    # Cependant les informations sont quand même dans la liste frame
    # On ajoute cette dernère frame à la fin de la boucle
    splitlist.append(frame)

    return splitlist


# ===============Files==================
def randomfichier():
    # Génération d'un nombre de 001-593 (différent de 1-593)
    centaine = random.randint(0, 5)
    if (centaine == 5):
        dizaine = random.randint(0, 3)
    else:
        dizaine = random.randint(0, 9)
    if (centaine == 0 and dizaine == 0):
        unite = random.randint(1, 9)
    else:
        unite = random.randint(0, 9)
    # Concaténation des varaibles pour générer le nom correct du fichier
    nomfichier = "arctic_a0%i%i%i.wav" % (centaine, dizaine, unite)
    return nomfichier


def readFiles(choix):
    if (choix == 'male' or choix == 'female'):
        if (choix == 'male'):
            choix = '../cmu_us_bdl_arctic/wav/%s'
        elif (choix == 'female'):
            choix = '../cmu_us_slt_arctic/wav/%s'

        utterance = read(choix % (randomfichier()))
        fe = utterance[0]
        sig = utterance[1]
        signal = sig.tolist()
    else:
        print('Choisissez entre "male" ou "female"')

    return signal, fe


# ===============Features===============
def energy(signal):
    energy = 0  # Déclaration à 0 (valeur neutre de la somme)
    for value in signal:
        energy += value ** 2
    return energy


def frameEnergy(splitlist):
    # Liste de langueur égale au nombre de frames
    B = np.zeros(len(splitlist))
    # Calcul des energies de plusieurs frames
    for frame in range(len(splitlist)):
        B[frame] = energy(splitlist[frame])

    return B


def xcorr(x, y=None, scale='none', maxlag=None):
    """compute correlation of x and y.
    If y not given compute autocorrelation.

    Arguments:
        x {np.ndarray} -- signal
        y {np.ndarray} -- signal

    Keyword Arguments:
        scale {str} -- can be either "none", "biased" or "unbiased" (default: {'none'})
        maxlag {str} -- maximum lag to be returned. This should be <= round(y.size+x.size-1)/2 (default: {'none'})
    Returns:
        [np.ndarry] -- corresponding lags
        [np.ndarray] -- resulting correlation signal
    """
    # If y is None ccmpute autocorrelation
    if y is None:
        y = x
    # Pad shorter array if signals are different lengths
    else:
        if x.size > y.size:
            pad_amount = x.size - y.size
            y = np.append(y, np.repeat(0, pad_amount))
        elif y.size > x.size:
            pad_amount = y.size - x.size
            x = np.append(x, np.repeat(0, pad_amount))
    if maxlag is None:
        maxlag = (x.size + y.size - 1) / 2
    if maxlag > round((y.size + x.size - 1) / 2):
        raise ValueError("maxlag should be <= round(y.size+x.size-1)/2")
    corr = np.correlate(x, y, mode='full')  # scale = 'none'
    lags = np.arange(-maxlag, maxlag + 1)
    # lags = np.arange(-x.size, x.size-1)
    corr = corr[(x.size - 1 - maxlag):(x.size + maxlag)]
    if scale == 'biased':
        corr = corr / x.size
    elif scale == 'unbiased':
        corr /= (x.size - abs(lags))
    elif scale == 'coeff':
        corr /= np.sqrt(np.dot(x, x) * np.dot(y, y))
    # lags = lags[int(round(len(lags)/2)-maxlag+1) : int(round(len(lags)/2)+maxlag-1)]
    # corr = corr[int(round(len(corr)/2)-maxlag+1) : int(round(len(corr)/2)+maxlag-1)]
    return lags, corr


def autocorr(signal, fe, Enerseuil):
    listpitch = []  # on initialise notre liste
    energy = frameEnergy(signal)

    for i in range(len(signal)):  # demarre une boucl for afin de parcourir toutes les frames du signal

        if (energy[i - 1] > Enerseuil):  # On ne prends que les valeurs avec un certains seuil d energie
            arrsignal = np.array(signal[i - 1])  # transforme notre signal en array list afin de faciiliter les calculs
            lags, corr = xcorr(arrsignal, maxlag=int(fe / 50))  # Utilise la fct corr pour etudier les correlations
            # plt.plot(corr)

            maxima, value = sgl.find_peaks(corr, height=0, distance=40)  # on cherche tous les maxima
            Hvalue = value[
                'peak_heights']  # Lorsqu on lit les "values", cela nous affiche {'peak_heights':array([... ])} et on veut juste array
            # "maxima" correspond a la position des sommets sur l axe des x et "value" correspond aux hauteurs de ceux-ci
            maxima = maxima.tolist()  # On transforme "maxima" en array list
            Hvalue, maxima = zip(*sorted(zip(Hvalue, maxima)))  # on trie par ordre croissant les hauteurs
            temp = np.abs(maxima[len(maxima) - 1] - maxima[len(
                maxima) - 2])  # on calcul la distance, sur l axe des x, des 2 hauteurs les plus significatives
            ffond = fe / temp  # on calcul la frequence fondamentale
            listpitch.append(ffond)  # introduit la donnee dans la liste

        else:
            ffond = 0  # si les valeurs sont en dessous du seuil d energie, on considere la valeurs comme nulle
            listpitch.append(ffond)

    Vpitch = np.array(listpitch)  # transforme la liste en array list pour faciliter les calculs
    Fpitch = abs(np.mean(Vpitch))  # calcul la moyenne de l array liste

    return listpitch


def compute_cepstrum(signal, sample_freq):
    """Computes cepstrum."""
    frame_size = signal.size
    windowed_signal = np.hamming(frame_size) * signal
    dt = 1 / sample_freq
    freq_vector = np.fft.rfftfreq(frame_size, d=dt)
    X = np.fft.rfft(windowed_signal)
    log_X = np.log(np.abs(X))
    cepstrum = np.fft.rfft(log_X)
    df = freq_vector[1] - freq_vector[0]
    quefrency_vector = np.fft.rfftfreq(log_X.size, df)
    return quefrency_vector, cepstrum


def cepstrum_f0_detection(signal, sample_freq, fmin=60, fmax=500):
    """Returns f0 based on cepstral processing."""
    quefrency_vector, cepstrum = compute_cepstrum(signal, sample_freq)
    # extract peak in cepstrum in valid region
    valid = (quefrency_vector > 1 / fmax) & (quefrency_vector <= 1 / fmin)
    max_quefrency_index = np.argmax(np.abs(cepstrum)[valid])
    f0 = 1 / quefrency_vector[valid][max_quefrency_index]
    return f0


def cepstrum(signal, fe, threshold):
    listceps = []  # on initialise notre liste
    energy = frameEnergy(signal)

    for i in range(len(signal)):

        if energy[i - 1] > threshold:

            arrsignal = np.array(signal[i - 1])

            ffond = cepstrum_f0_detection(arrsignal, fe)

            listceps.append((ffond))


        else:
            ffond = 0
            listceps.append(ffond)

    Vceps = np.array(listceps)  # transforme la liste en array list pour faciliter les calculs
    Fceps = abs(np.mean(Vceps))  # calcul la moyenne de l array liste
    return listceps


def formant(signal, fe):
    for i in range(len(signal)):
        framefilter = np.array(sgl.lfilter([1., -0.67], 1, signal[i - 1]))

        N = len(framefilter)
        framehamming = framefilter * sgl.hamming(N)

        # get LPC
        A = lpc_ref(framehamming, 12)

        # get roots
        rts = np.roots(A)

        # angz = np.array(np.angle(rts)) ou
        angz = np.arctan2(np.imag(rts), np.real(rts))

        frqs = np.array(np.unique(abs(angz * (fe / (2 * np.pi)))))

        frqs.tolist()
        frqs.sort()
        frq = np.array(frqs)
        frq = np.mean(frq)

    return frq


# ========Discriminatory rules=========
def ruleEnergy(n):
    Twidth = 100
    Tstep = 100
    for i in range(n):
        signal, fe = readFiles('male')
        normsignal = norm(signal)
        splitsignal = split(normsignal, fe, Twidth, Tstep)
        energysignal = frameEnergy(splitsignal)
        plt.plot(energysignal, '+', color='blue')

        signal, fe = readFiles('female')
        normsignal = norm(signal)
        splitsignal = split(normsignal, fe, Twidth, Tstep)
        energysignal = frameEnergy(splitsignal)
        plt.plot(energysignal, '+', color='pink')
    plt.show()


def ruleAutocorr(n):
    Twidth = 100
    Tstep = 100
    for i in range(n):
        signal, fe = readFiles('male')
        normsignal = norm(signal)
        splitsignal = split(normsignal, fe, Twidth, Tstep)
        listpitch = autocorr(splitsignal, fe, 50)
        plt.plot(listpitch, '+', color='blue')

        signal, fe = readFiles('female')
        normsignal = norm(signal)
        splitsignal = split(normsignal, fe, Twidth, Tstep)
        listpitch = autocorr(splitsignal, fe, 50)
        plt.plot(listpitch, '+', color='pink')
    plt.show()


ruleAutocorr(50)


def ruleCepstrum(n):
    Twidth = 100
    Tstep = 100
    for i in range(n):
        signal, fe = readFiles('male')
        normsignal = norm(signal)
        splitsignal = split(normsignal, fe, Twidth, Tstep)
        listpitch = cepstrum(splitsignal, fe, 30)
        plt.plot(listpitch, '+', color='blue')

        signal, fe = readFiles('female')
        normsignal = norm(signal)
        splitsignal = split(normsignal, fe, Twidth, Tstep)
        listpitch = cepstrum(splitsignal, fe, 30)
        plt.plot(listpitch, '+', color='pink')
    plt.show()


ruleCepstrum(50)


def ruleFormant(n):
    Twidth = 50
    Tstep = 50
    for i in range(n):
        signal, fe = readFiles('male')
        normsignal = norm(signal)
        splitsignal = split(normsignal, fe, Twidth, Tstep)
        listformant = formant(splitsignal, fe)
        plt.plot(listformant, '+', color='blue')

        signal, fe = readFiles('female')
        normsignal = norm(signal)
        splitsignal = split(normsignal, fe, Twidth, Tstep)
        listformant = formant(splitsignal, fe)
        plt.plot(listformant, '+', color='pink')
    plt.show()


# ruleFormant(50)


# ==========Rule-based Systeme=========
def ruleBasedSystem(n):
    Twidth = 100
    Tstep = 100
    score = 0
    for i in range(n):
        if (random.randint(0, 100) % 2 == 0):
            typefichier = 'female'
        else:
            typefichier = 'male'
        signal, fe = readFiles(typefichier)
        normsignal = norm(signal)
        splitsignal = split(normsignal, fe, Twidth, Tstep)
        energysignal = frameEnergy(splitsignal)
        # pitch
        # formants
        array = np.array(energysignal)
        if (max(array) < 150):
            if (typefichier == 'male'):
                score += 1
        else:
            if (typefichier == 'female'):
                score += 1

        # return male += 1 or female += 1
    accuracy = score / n
    return accuracy

# print(ruleBasedSystem(50))
