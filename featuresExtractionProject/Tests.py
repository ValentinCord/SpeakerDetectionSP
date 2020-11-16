#test norm
def testnorm(normsignal):
 for i in range(len(normsignal)):
  if ((normsignal[i]<-1)or(normsignal[i]>1)):
      return "Value error"
 if ((1 not in normsignal) and (-1 not in normsignal)):
        return" Maximum error"
 else:
         return "Test ok"
#print(testnorm(signal1))


#test energy
def testenergy(energy):
    for value in energy:
        if(value<0):
            return "Error"
            break
    return "Test ok"
#print(testenergy(signal1))


#test random
def testrandom(number):
    if ((number>539)or (number<0)):
        return "Error"
    else: 
        return"Test ok"
#print(testrandom(nombre))


#test readfiles
def testreadfiles(signal,fe):
    if (not signal==True):
        return"Erreur, la liste est vide"
        
    elif(not fe==True):
        return"Fe non récupérée"
        
    else:
        return"Test ok"
    
