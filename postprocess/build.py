"""Build every paper figure, table and number from the committed results:
    python -m postprocess.build
"""
from . import make_figures, make_numbers, make_tables

if __name__ == "__main__":
    make_figures.main()
    make_tables.main()
    make_numbers.main()
