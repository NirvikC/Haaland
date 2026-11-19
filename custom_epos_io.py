

import os
import numpy as np
from apt_io import EPOS_DTYPE, read_epos as _read_epos_raw


def read_custom_epos(path):
    """
    Loads the standard ePOS file and returns a dict of arrays matching
    the field names the periodicity pipeline expects (det_x, det_y,
    mn, pulse_number, event_number), so analyze_aperture() and friends
    work unchanged regardless of which underlying file format you have.
    """
    data, pulse_number = _read_epos_raw(path)
    event_number = np.arange(len(data), dtype=np.uint32)

    return {
        'x': data['x'], 'y': data['y'], 'z': data['z'],
        'mn': data['mn'],
        'tof': data['tof'],
        'standing_v': data['standing_v'], 'pulse_v': data['pulse_v'],
        'det_x': data['det_x'], 'det_y': data['det_y'],
        'multiplicity': data['multiplicity'],
        'pulse_number': pulse_number,
        'event_number': event_number,
    }


def verify_epos(path, n_preview=10):
    """
    Diagnostic checks adapted for the standard ePOS layout - same
    pattern as before (record-size check, per-column ranges, monotonic
    sequencing), but checking the fields that actually exist in this
    format instead of the custom export's fields.
    """
    file_size = os.path.getsize(path)
    record_size = EPOS_DTYPE.itemsize
    print(f"File size: {file_size} bytes")
    print(f"Expected record size: {record_size} bytes ({len(EPOS_DTYPE)} fields)")
    ratio = file_size / record_size
    print(f"File size / record size = {ratio}")
    if not ratio.is_integer():
        print("!! WARNING: not a whole number of records - dtype layout "
              "does not match this file.")
    print()

    data = read_custom_epos(path)
    n = len(data['mn'])
    print(f"Loaded {n} records\n")

    print(f"First {n_preview} raw records:")
    for i in range(min(n_preview, n)):
        print(f"  ion {i}: x={data['x'][i]:.3f} y={data['y'][i]:.3f} "
              f"z={data['z'][i]:.3f} mn={data['mn'][i]:.3f} "
              f"tof={data['tof'][i]:.2f} standing_v={data['standing_v'][i]:.1f} "
              f"pulse_v={data['pulse_v'][i]:.1f} det_x={data['det_x'][i]:.3f} "
              f"det_y={data['det_y'][i]:.3f} pulse_number={data['pulse_number'][i]} "
              f"event_number={data['event_number'][i]} "
              f"multiplicity={data['multiplicity'][i]}")
    print()

    print("Per-column summary:")
    for name in ('mn', 'tof', 'standing_v', 'pulse_v', 'det_x', 'det_y'):
        col = data[name]
        n_unique = len(np.unique(col[:min(len(col), 100000)]))
        print(f"  {name:16s} min={col.min():>12.4g}  max={col.max():>12.4g}  "
              f"mean={col.mean():>12.4g}  unique(sample)={n_unique}")
        if n_unique <= 1:
            print(f"    !! WARNING: '{name}' appears constant - not being "
                  f"read correctly, or genuinely not recorded in this "
                  f"acquisition (e.g. standing_v/pulse_v are known to be "
                  f"unavailable in some exports - see earlier diagnosis "
                  f"in this conversation).")
    print()

    for seq_name in ('pulse_number', 'event_number'):
        seq = data[seq_name].astype(np.int64)
        non_monotonic_frac = (np.diff(seq) < 0).mean()
        print(f"{seq_name}: fraction of non-monotonic steps = "
              f"{non_monotonic_frac*100:.4f}%")
        if non_monotonic_frac > 0.01:
            print(f"    !! WARNING: '{seq_name}' should almost always "
                  f"increase - check the dtype layout.")
    print()

    for name in ('det_x', 'det_y'):
        col = data[name]
        print(f"{name}: range [{col.min():.2f}, {col.max():.2f}], "
              f"std={col.std():.2f}")
        if col.std() < 1e-6:
            print(f"    !! WARNING: '{name}' has ~zero spread.")

    return data

