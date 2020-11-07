import numpy as np
from scipy.io.wavfile import read
import random

signal = [2, 3, 5, 9, 18, 62, 53, 1, 23, 5, 4, 6, 5, 66, 45, 100, -100]
sig2 = [1, 1, 0, 0, 1, 1, 0, 0, 1, 1]


def norm(signal):
    maximum = max(signal)
    minimum = min(signal)

    if (maximum == minimum == 0):
        normsignal = signal

    else:
        if (maximum > abs(minimum)):
            normsignal = [float(i) / max(signal) for i in signal]

        else:
            normsignal = [float(i) / -min(signal) for i in signal]

    return normsignal


def split(signal, fe, Twidth,
          Tstep):  # # fe = nbr of point took every second : 1s -> fe samples <=> Nsamples = duration * fe

    if (Twidth == Tstep == 0):
        print("la fenetre est de largeur 0 et le saut est de 0")
        return
    else:
        if (Twidth == 0):
            print("la fenetre est de largeur 0")
            return

        else:
            if (Twidth > Tstep):
                print("overlap condition is false")
                return

    if (fe == 0):
        print("la fréquence échantillon est nulle")
        return

        # compute frame_len & frame_step (seconds to samples)
    Nwidth = Twidth * fe  # nombre de points par frame
    Nstep = Tstep * fe  # nombre de points par step

    splitlist = []
    frame = []
    diff = Nstep - Nwidth

    i = 0
    temp = Nwidth
    print(len(signal))

    while i < len(signal):
        if (i % temp == 0 and i != 0):
            splitlist.append(frame)
            frame = []
            i += diff
            # temp += Nstep

            # Obligé sinon cela ne marche pas
            if (temp + Nstep <= len(signal)):
                temp += Nstep

            else:

                return splitlist

        frame.append(signal[i])
        print(frame)

        i += 1

    splitlist.append(frame)

    return splitlist


def energy(signal):
    energy = 0
    for value in signal:
        energy += value ** 2
    return energy


def randomfichier():
    unite = random.randint(0, 9)
    centaine = random.randint(0, 5)
    if (centaine == 5):
        dizaine = random.randint(0, 3)
    else:
        dizaine = random.randint(0, 9)

    nomfichier = "arctic_b0%i%i%i.wav" % (centaine, dizaine, unite)
    return nomfichier


def frameEnergy(A):
    B = np.zeros(len(A))

    for i in range(len(A)):
        B[i] = energy(A[i])

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















