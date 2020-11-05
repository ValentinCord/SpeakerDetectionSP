from utils import energy

# teste que des valeurs posiftive ou nulle
signal1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
if (energy(signal1) == 385):
    print("true")

else:
    print("wrong")

# teste avec des valeurs négatives
signal2 = [0, -1, -2, -3, -4, -5, 6, 7, 8, 9, 10]
if (energy(signal2) == 385):
    print("true")

else:
    print("wrong")

# teste avec que des valeurs nulles
signal3 = [0, 0, 0, 0, 0, 0, 0]
if (energy(signal3) == 0):
    print("true")

else:
    print("wrong")