import re

analysis_block = """

================================================================================
                             Excited State Analysis
================================================================================

"""

# The pattern doesn't match because it's looking for a line with just "Excited State Analysis"
# surrounded by optional whitespace. But the actual line has "=" characters before and after.

# Here's a pattern that would match:
EXCITED_STATE_ANALYSIS_HEADER_PAT = re.compile(r"^\s*Excited State Analysis\s*$")

for line in analysis_block.split("\n"):
    print(EXCITED_STATE_ANALYSIS_HEADER_PAT.search(line))
