import re

analysis_block = """
(4)  G-CDS(liq) cavity-dispersion-solvent structure 
     free energy                                                1.4731 kcal/mol
(6)  G-S(liq) free energy of system                      -75.318460236 a.u.
 SCF   energy in the final basis set =      -75.3184602363
 Total energy in the final basis set =      -75.3184602363
 
"""

pattern = re.compile(r"^\s*free energy*\s*(-?\d+\.\d+)\s*kcal/mol")


for line in analysis_block.split("\n"):
    # Remove leading/trailing whitespace before search
    stripped_line = line.strip()
    match = pattern.search(stripped_line)
    print(match)
    # For demonstration, if a match is found, print its groups:
    if match:
        print(f"Matched: {match.group(1)}")
