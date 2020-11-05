from utils import split

# Teste width < step
signal1 = [1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
fe1 = 1
width1 = 2
step1 = 4

if (split(signal1, fe1, width1, step1) == [[1, 1], [1, 1], [1, 1]]):
    print("true")
else:
    print("false")

# Teste widht = step
signal2 = [1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
fe2 = 1
width2 = 2
step2 = 2

if (split(signal2, fe2, width2, step2) == [[1, 1], [0, 0], [1, 1], [0, 0], [1, 1]]):
    print("true")
else:
    print("false")

# Teste width > step
signal3 = [1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
fe3 = 1
width3 = 4
step3 = 2

if (split(signal3, fe3, width3, step3) == None):
    print("true")
else:
    print("false")

# Teste en modifiant fe
signal4 = [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]
fe4 = 2
width4 = 2
step4 = 4

if (split(signal4, fe4, width4, step4) == [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]):
    print("true")
else:
    print("false")

# Teste width = step = 0
signal5 = [1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
fe5 = 1
width5 = 0
step5 = 0

if (split(signal5, fe5, width5, step5) == None):
    print("true")
else:
    print("false")

# Teste width = 0 (pas besion de testé step = 0 séparement car on a déjà testé step < width)
signal6 = [1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
fe6 = 1
width6 = 0
step6 = 2

if (split(signal6, fe6, width6, step6) == None):
    print("true")
else:
    print("false")

# Teste fe = 0
signal7 = [1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
fe7 = 0
width7 = 2
step7 = 4

if (split(signal7, fe7, width7, step7) == None):
    print("true")
else:
    print("false")