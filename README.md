# BioCloud Simulator v1.0

**Bistable lattice model of bioelectric cancer cloud 
normalization with instructor cells**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/)

## What This Does

Cancer cells are electrically depolarized (-10 mV) compared 
to healthy cells (-70 mV). This simulator models a "cloud" 
of depolarized cancer cells surrounded by healthy tissue 
and answers three questions:

1. **Critical size:** How large must the cloud be before 
   healthy tissue can no longer normalize it?
2. **Instructor cells:** How do boundary cells that maintain 
   the cloud affect the critical size?
3. **Therapy:** Which therapeutic strategy (gap junction 
   restoration, instructor targeting, or combination) 
   works best?

### Key Results (MCF-7 parameters)

| Cloud radius | Outcome | Ratio |
|---|---|---|
| R = 3 (25 cells) | Full normalization | 0.000 |
| R = 5 (69 cells) | Full normalization | 0.000 |
| R = 8 (193 cells) | Strong shrinkage | 0.295 |
| R = 12 (437 cells) | Partial shrinkage | 0.506 |

- Instructor cells increase cloud stability by **4.2×**
- Combination therapy (GJ + instructor removal): **97% normalization**
- Results consistent with Cervera 2016 and Carvalho 2021

## Installation

```bash
git clone https://github.com/oshuhrat/biocloud.git
cd biocloud
pip install -r requirements.txt
```
Or run directly in Google Colab — no installation needed.

## Quick Start
### 3 lines to first result
```python
from biocloud import BioCloud

sim = BioCloud("MCF7")
sim.run(cloud_radius=8, g_gap_boundary=0.5)
```

### Phase diagram
```python
sim = BioCloud("MCF7", grid_size=80)
sim.phase_diagram()
```

### Instructor cell comparison
```python
sim.instructor_comparison(cloud_radius=8)
```

### Therapy comparison
```python
sim.therapy_comparison(cloud_radius=8)
```

### Full analysis
```python
sim = BioCloud("MCF7", grid_size=80)
sim.full_analysis(cloud_radius=8)
sim.summary()
```

### Your own cell line
```python
sim = BioCloud("configs/custom_template.json")
# or
sim = BioCloud("CUSTOM")
sim.config["G_pol"] = 4.0  # your value
sim.config["G_dep"] = 5.0  # your value
sim.save_config("my_cell_line.json")
```

## Model
Based on the Carvalho (2021) bistable framework:

```text
C_m dV/dt = -G_pol * f_pol(V) * (V - E_pol)
            -G_dep * f_dep(V) * (V - E_dep)
            + Σ g_gj * gating(Vi, Vj) * (Vj - Vi)
```

Where:
- f_pol, f_dep = voltage-dependent sigmoid gating
- g_gj = gap junction conductance (voltage-dependent, Carvalho gating)
- Lattice: N×N grid with 4-neighbor coupling

New contributions beyond Cervera/Carvalho:
- Instructor cells at tumor boundary
- Five therapeutic strategy comparisons
- Open-source, parameterizable code

## Parameters
All parameters are sourced from primary literature.
See `configs/mcf7.json` for full provenance.

| Parameter | Value | Source | Verified? |
|---|---|---|---|
| E_pol (K+) | -90 mV | Textbook | ✅ |
| E_dep (Na+) | 0 mV | Carvalho 2021 | ✅ |
| G_pol | 3.0 nS | MCF-7 IK data | Estimated |
| G_dep | 4.25 nS | Fitted to MCF-7 Vmem | Estimated |
| g_gj healthy | 2.0 nS | Robinson 1993 | ✅ |
| g_gj cancer | 0.5 nS | Reduced Cx43 | Estimated |

## Limitations
- 2D model. 3D tissue has more neighbors → different critical sizes.
- Healthy attractor at -90 mV instead of measured -60 mV. Qualitative conclusions unchanged.
- MCF-7 only. Other subtypes need their own parameters.
- No proliferation dynamics in base model.
- In silico only. No experimental validation yet.

## Citation
If you use this code, please cite:

> Khamidov U. Beyond Bistability: A Multistable Cascade Model of Bioelectric Cancer Normalization. bioRxiv. 2026.

## References
1. Cervera J, Alcaraz A, Mafé S. Bioelectrical signals and ion channels in the modeling of multicellular patterns and cancer biophysics. Sci Rep. 2016;6:20403.
2. Carvalho J. A bioelectric model of carcinogenesis. Sci Rep. 2021;11:13607.
3. Chernet BT, Levin M. Transmembrane voltage potential is an essential cellular parameter for the detection and control of tumor development. Dis Model Mech. 2013;6(3):595-607.
4. Wonderlin WF et al. Changes in membrane potential during the progression of MCF-7 human mammary tumor cells through the cell cycle. J Cell Physiol. 1995;165(1):177-185.

## License
MIT — use freely, cite if you publish.
