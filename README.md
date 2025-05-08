README

This program crudely estimates the orbital periods and RV slopes of binary stars from the Gaia catalog using static postion information.

If you use this code, please cite the relevant paper (Giovinazzi et al. 2025).

To get started, you should clone a copy of the code to your home machine.

```python
git clone https://github.com/markgiovinazzi/binary_mc
cd binary_mc
python3 setup.py sdist
pip3 install dist/binary_mc-1.0.tar.gz
```

**Example usage

HD 190360 AB is a known, very wide (~3') binary star system. Suppose we are interested in how likely `B` (Gaia DR3 2029432043779954432) is to induce a non-negligible effect on `A` (Gaia DR3 2029433521248546304). We can call `binary_mc` using the two Gaia IDs and let the code do the rest!

```
from binary_mc import binary_mc
# query our two stars (HD 190360 A and HD 190360 B)
binary_mc(['2029433521248546304', '2029432043779954432'])
```

This call produces the following output. We see that with a minimum period of 44,000 yr (and a most likely one of 90,000 yr), HD 190360 B will likely induce an RV slope on HD 190360 A that is < 1 cm/s/yr. This may rightly justify the exclusion of `B` when analyzing the RVs of `A`.

```
--- Binary Orbit Diagnostics ---

[Gaia DR3 2029433521248546304]
Most likely period (approx.): 89812.95 years
Minimum period: 44195.57 years
Most likely RV slope (approx.): 0.0035 m/s/yr
90% of orbits have RV slope < 0.0123 m/s/yr
99% of orbits have RV slope < 0.0444 m/s/yr

--------------------------------
```

The code also supports multiple bound stellar companions. Below we query the HD 26965 (A-BC) star system.

`binary_mc(['3195919528989223040', '3195919254111315712', '3195919254111314816'])`

Here centered on C, the output correctly suggests that `B` induces a far stronger signal, though `A` (while weaker) cannot safely be ignored. Triple systems like HD 26965 are difficult to fit for, but `binary_mc` makes useful 1st-order binary orbit estimates easy for anyone!

```
--- Binary Orbit Diagnostics ---

[Gaia DR3 3195919254111315712]
Most likely period (approx.): 179.55 years
Minimum period: 87.68 years
Most likely RV slope (approx.): 12.2136 m/s/yr
90% of orbits have RV slope > 1.6821 m/s/yr
99% of orbits have RV slope > 0.1950 m/s/yr

[Gaia DR3 3195919528989223040]
Most likely period (approx.): 5018.48 years
Minimum period: 2459.72 years
Most likely RV slope (approx.): 0.2043 m/s/yr
90% of orbits have RV slope < 0.7595 m/s/yr
99% of orbits have RV slope < 2.7359 m/s/yr

--------------------------------
```

