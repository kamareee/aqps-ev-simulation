"""Build every paper figure, table and number from the committed results:
    uv run --frozen python -m postprocess.build

Use the locked environment (uv.lock: matplotlib 3.10.8, numpy 2.4.2, pandas 3.0.0): figure bytes
depend on the matplotlib version, so other versions show every figure as modified. Text outputs
are written with LF line endings on every OS.
"""
from . import make_figures, make_numbers, make_tables

if __name__ == "__main__":
    make_figures.main()
    make_tables.main()
    make_numbers.main()
