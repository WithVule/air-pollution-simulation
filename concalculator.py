MCO2 = 44
MCO = 28
MNH3 = 17
MNO = 30
MSO2 = 64

def calculation(Mol, mmt):
  m = 10**6 * mmt
  Vol = 10**12
  mol = m/Mol 
  C = mol/Vol
  print(C)

calculation(MCO2, 1.302)