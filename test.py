import re

tda_block = """

 ---------------------------------------------------
         TDDFT/TDA Excitation Energies              
 ---------------------------------------------------

 Excited state   1: excitation energy (eV) =       7.5954
 Total energy for state   1:                   -76.16212816 au
 
"""

tddft_block = """
 ---------------------------------------------------
             TDDFT Excitation Energies              
 ---------------------------------------------------

 Excited state   1: excitation energy (eV) =        7.5725
 """

TDA_HEADER_PAT = re.compile(r"^\s+TDDFT/TDA Excitation Energies\s*$")
TDA_HEADER_PAT = re.compile(r"^\s*TDDFT/TDA\s+Excitation\s+Energies\s*$")
# TDDFT_HEADER_PAT = re.compile(r"^\s+TDDFT Excitation Energies\s*$")
TDDFT_HEADER_PAT = re.compile(r"^\s*TDDFT\s+Excitation\s+Energies\s*$")

for line in tda_block.split("\n"):
    if TDA_HEADER_PAT.match(line):
        print("TDA pattern in tda block")
    elif TDDFT_HEADER_PAT.match(line):
        print("TDDFT pattern in tda block")

for line in tddft_block.split("\n"):
    if TDA_HEADER_PAT.match(line):
        print("TDA pattern in tddft block")
    elif TDDFT_HEADER_PAT.match(line):
        print("TDDFT pattern in tddft block")
