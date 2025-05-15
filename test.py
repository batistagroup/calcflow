import re

analysis_block = """
  Singlet 1 :
  -------------------------
    Decomposition into state-averaged NTOs
      H- 0 -> L+ 0: -0.7067 ( 99.9%)
                     omega = 100.1%

"""

NTO_CONTRIBUTION_PATTERN = re.compile(
    r"""
    ^\s*                             # optional leading spaces
    ([A-Za-z]+[+-]\s*\d+)            # group1: e.g. "H- 0"
    \s*->\s*                         # arrow with optional spaces
    ([A-Za-z]+[+-]\s*\d+)            # group2: e.g. "L+ 0"
    \s*:\s*                          # colon with optional spaces
    (-?\d+\.\d+)                     # group3: coefficient
    \s*\(                            # space* + literal "("
      \s*(\d+\.\d+)\s*%             # group4: percentage, with spaces around
    \)\s*                            # literal ")" + optional trailing spaces
    $                                # end of line
    """,
    re.VERBOSE,
)


for line in analysis_block.split("\n"):
    # Remove leading/trailing whitespace before search
    stripped_line = line.strip()
    match = NTO_CONTRIBUTION_PATTERN.search(stripped_line)
    print(match)
    # For demonstration, if a match is found, print its groups:
    # if match:
    #     print(f"Matched: {match.groups()}")
