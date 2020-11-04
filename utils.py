import numpy as np
from scipy.io.wavfile import read
import random

signal = [2, 3, 5, 9, 18, 62, 53, 1, 23, 5, 4, 6, 5, 66, 45, 100, -100]
sig2 = [1, 1, 0, 0, 1, 1, 0, 0, 1, 1]


def norm(signal):
    maximum = max(signal)
    minimum = min(signal)
    
    if (maximum==minimum==0):
        normsignal = signal
    
    else:
        if (maximum> abs(minimum)):
            normsignal = [float(i)/max(signal) for i in signal]

        else : 
            normsignal = [float(i)/-min(signal) for i in signal]

    return normsignal


def split(signal, fe, Twidth,
          Tstep):  # # fe = nbr of point took every second : 1s -> fe samples <=> Nsamples = duration * fe

    if (Twidth > Tstep):
        print("overlap condition is false")
        return

        # compute frame_len & frame_step (seconds to samples)
    Nwidth = Twidth * fe  # nombre de points par frame
    Nstep = Tstep * fe  # nombre de points par step

    splitlist = []
    frame = []
    diff = Nstep - Nwidth

    i = 0
    temp = Nwidth

    while i < len(signal):
        if (i % temp == 0 and i != 0):
            splitlist.append(frame)
            frame = []
            i += diff
            temp += Nstep
        frame.append(signal[i])

        i += 1

    splitlist.append(frame)

    print(splitlist)
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


#utterance = read("/cmu_us_ksp_arctic/wav/%s" % (randomfichier()))

