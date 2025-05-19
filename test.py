import re

analysis_block = """
(3)  G-ENP(liq) elect-nuc-pol free energy of system      -76.456221333 a.u.
(4)  G-CDS(liq) cavity-dispersion-solvent structure 
     free energy                                                1.4731 kcal/mol
(6)  G-S(liq) free energy of system                      -76.453873869 a.u.
 SCF   energy in the final basis set =      -76.4538738687
 Total energy in the final basis set =      -76.4538738687
 
"""

pattern = re.compile(r"^\s*\(3\)\s+G-ENP\(liq\).*?\s*(-?\d+\.\d+)\s*a\.u\.")


for line in analysis_block.split("\n"):
    # Remove leading/trailing whitespace before search
    stripped_line = line.strip()
    match = pattern.search(stripped_line)
    print(match)
    # For demonstration, if a match is found, print its groups:
    if match:
        print(f"Matched: {match.group(1)}")
