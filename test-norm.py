import numpy as np

from utils import norm



# le plus haut point est un maximum 
signal1 = [2, 4, 6 ,8]
if (norm(signal1)==[0.25, 0.5, 0.75, 1]):
    print("true")

else:
    print("wrong")


signal2 = [2, 4, 6, -8]
if (norm(signal2)==[0.25, 0.5, 0.75, -1]):
    print("true")

else:
    print("wrong")
    

signal3 =[0,0,0,0,0]
if (norm(signal3)==[0,0,0,0,0]):
    print("true")

else:
    print("wrong")

