"""
Readers for Cameca/IVAS ePOS and RRNG files.

ePOS record layout (big-endian, 11 x 4-byte fields per ion):
    x, y, z          - reconstructed position (nm)
    mn                - mass-to-charge ratio (Da)
    tof               - time of flight (ns)
    standing_v        - standing voltage (V)
    pulse_v           - pulse voltage (V)
    det_x, det_y      - raw detector position (mm)
    delta_pulse       - number of pulses since the previous recorded ion
    multiplicity      - ion order within its multi-hit event (0 = first ion, etc.)

RRNG is a plain XML/text range file exported from IVAS. This parser handles
the standard <ion .../> <range .../> layout used by IVAS/AP Suite.
"""

import numpy as np
import re

EPOS_DTYPE = np.dtype([
    ('x', '>f4'), ('y', '>f4'), ('z', '>f4'),
    ('mn', '>f4'),
    ('tof', '>f4'),
    ('standing_v', '>f4'), ('pulse_v', '>f4'),
    ('det_x', '>f4'), ('det_y', '>f4'),
    ('delta_pulse', '>u4'),
    ('multiplicity', '>u4'),
])


def read_epos(path):
    """Load an ePOS file into a structured numpy array, and add an
    absolute pulse_number column (cumulative sum of delta_pulse)."""
    data = np.fromfile(path, dtype=EPOS_DTYPE)
    pulse_number = np.cumsum(data['delta_pulse'].astype(np.int64))
    return data, pulse_number


def read_rrng(path):
    """
    Parse an IVAS/AP Suite .rrng file.
    Returns a list of dicts: {'name': str, 'lo': float, 'hi': float}

    Charge-state handling: many IVAS-exported .rrng files (including
    files with no 'Name:' tag) only give element:count - the charge state
    is NOT stated explicitly, it's only implicit in which mass window the
    range falls in. Since ranges of the same element but different charge
    states (e.g. Si+ vs Si2+) are physically distinct species and must be
    kept separate for correlation analysis, this parser infers the charge
    by comparing each range's center mass to the known most-abundant mass
    of the specified element combination, and appends the inferred charge
    to the name (e.g. 'Si+', 'Si2+') so they are never silently merged.
    """
    ranges = []
    with open(path, 'r', errors='ignore') as f:
        text = f.read()

    pattern = re.compile(
        r'Range\d+\s*=\s*([\d.Ee+-]+)\s+([\d.Ee+-]+)\s+(.*)'
    )
    for line in text.splitlines():
        m = pattern.search(line)
        if not m:
            continue
        lo, hi, rest = m.groups()
        lo, hi = float(lo), float(hi)

        tokens = rest.split()
        elems = []
        name = None
        for tok in tokens:
            if ':' not in tok:
                continue
            key, val = tok.split(':', 1)
            if key.lower() in ('vol', 'color'):
                continue
            if key.lower() == 'name':
                name = val
                continue
            try:
                n = int(val)
                elems.append((key, n))
            except ValueError:
                continue

        if name is None:
            # No explicit Name: tag - infer charge from range center vs.
            # the known mass of the specified element combination, then
            # build a name like 'Si+' / 'Si2+' rather than a bare 'Si'
            # that would collide across charge states.
            formula = ''.join(f'{el}{n}' if n > 1 else el for el, n in elems)
            total_mass = sum(ATOMIC_MASS.get(el, 0) * n for el, n in elems)
            center = (lo + hi) / 2.0
            charge = 1
            if total_mass > 0 and center > 0:
                charge = max(1, round(total_mass / center))
            # Use parentheses around the charge unconditionally - without
            # this, a formula like 'Si2' (dimer, charge 1) and 'Si'
            # (monomer, charge 2) both collapse to the ambiguous string
            # 'Si2+', silently merging two physically different species.
            # 'Si2(1+)' vs 'Si(2+)' are unambiguous.
            name = f'{formula}({charge}+)'

        ranges.append({'name': name, 'lo': lo, 'hi': hi})

    return ranges


# Approximate atomic masses for common elements, used only to infer
# charge state when a .rrng range has no explicit Name:/charge tag.
# Extend this table if your ranging includes other elements.
ATOMIC_MASS = {
         'H': 1.008, 'O': 15.999, 'Si': 28.085, 'C': 12.011, 'N': 14.007,
    'Al': 26.982, 'Fe': 55.845, 'Cr': 51.996, 'Ni': 58.693, 'Ga': 69.723,
    'W': 183.84, 'Ta': 180.948, 'Nb': 92.906, 'Mo': 95.95, 'Ti': 47.867,
}


def assign_species(mn_array, ranges):
    """
    Vectorised species assignment. Ions falling outside all ranges get
    label '' (unranged/noise) and are excluded from correlation analysis,
    consistent with how the paper treats ranging.

    NOTE: for large datasets (tens of millions of ions) prefer
    assign_species_codes below - this string-label version creates one
    Python string object per ion and can use several GB of RAM at
    30M-ion scale purely from object overhead, independent of the actual
    data size.
    """
    species = np.full(mn_array.shape, '', dtype=object)
    for r in ranges:
        mask = (mn_array >= r['lo']) & (mn_array <= r['hi'])
        species[mask] = r['name']
    return species


def assign_species_codes(mn_array, ranges):
    """
    Memory-light species assignment for large datasets. Returns:
        codes       - int16 array, one code per ion (-1 = unranged/noise)
        code_names  - list of species names, where code_names[c] is the
                      name for code c
    This never creates a Python string object per ion - only `len(ranges)`
    string objects total - so it costs ~2 bytes/ion instead of ~100+ bytes/ion.
    """
    codes = np.full(mn_array.shape, -1, dtype=np.int16)
    code_names = []
    for c, r in enumerate(ranges):
        mask = (mn_array >= r['lo']) & (mn_array <= r['hi'])
        codes[mask] = c
        code_names.append(r['name'])
    return codes, code_names
