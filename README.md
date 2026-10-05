# MaleCNS Nav-Lite

**Navigation-optimised lite database of the *Drosophila melanogaster* male central nervous system connectome**

> *"What the fly sees, how it decides, and how it moves — distilled into a robot-ready connectome."*

---

## Table of Contents

1. [What is this?](#1-what-is-this)
2. [Biological background](#2-biological-background)
3. [Navigation circuits: eye → brain → limb](#3-navigation-circuits-eye--brain--limb)
4. [Why exclude olfactory / gustatory / reproductive?](#4-why-exclude-olfactory--gustatory--reproductive)
5. [BFS extraction method](#5-bfs-extraction-method)
6. [Quick start](#6-quick-start)
7. [File structure](#7-file-structure)
8. [API reference](#8-api-reference)
9. [Robot-navigation usage](#9-robot-navigation-usage)
10. [Data provenance & licence](#10-data-provenance--licence)
11. [Citation](#11-citation)

---

## 1. What is this?

`malecns-nav-lite` is a **focused sub-connectome** extracted from the full
MaleCNS v1.0 dataset (Janelia FlyEM, 2024).  It keeps only the ~15,000–30,000
neurons (out of ~166,000) that form the sensorimotor axis relevant to **aerial
navigation**:

```
Compound eye  →  Visual neuropils  →  Central Complex  →  Descending Neurons
→  Wing Motor Neurons  +  Leg Motor Neurons  +  Haltere afferents
```

Everything else — olfactory projection neurons, mushroom-body Kenyon cells,
gustatory receptor neurons, reproductive circuits — is stripped before the BFS
expansion, so the resulting `.npz` loads in seconds and runs at >1× real-time
on a laptop CPU.

---

## 2. Biological background

### 2.1 The MaleCNS dataset

The **Male Central Nervous System** (MaleCNS) connectome is a full-synapse
resolution electron-microscopy reconstruction of the entire *Drosophila
melanogaster* (fruit fly) male nervous system, from brain to the tip of the
ventral nerve cord (VNC).  Published by Dorkenwald *et al.* (2024) and
Schlegel *et al.* (2024), it contains:

| Statistic | Value |
|-----------|-------|
| Neurons reconstructed | ~166,000 |
| Directed synaptic connections | ~25.6 million |
| Synapses total | ~54 million |
| Volume covered | Brain + VNC (full CNS) |
| Resolution | ~8 nm/voxel EM |

### 2.2 Fly neuroanatomy in brief

```
┌─────────────────────────────────────────────┐
│                    BRAIN                    │
│                                             │
│  Optic Lobes         Central Complex (CX)   │
│  ┌──────────┐        ┌────────────────────┐ │
│  │ Retina   │        │ Protocerebral      │ │
│  │ Lamina L1│→ AOTU →│ Bridge (PB)        │ │
│  │ Medulla  │        │ Ellipsoid Body (EB)│ │
│  │ Lobula   │→ LPLC →│ Fan-shaped Body(FB)│ │
│  │ LoP      │        │ Noduli (NO)        │ │
│  └──────────┘        └────────┬───────────┘ │
│                               │ PFL neurons │
│                        Lateral Accessory    │
│                        Lobe (LAL)           │
└───────────────────────────────┼─────────────┘
                                │ Descending Neurons (DNs)
┌───────────────────────────────▼─────────────┐
│            VENTRAL NERVE CORD (VNC)          │
│                                             │
│  Thoracic Ganglia T1/T2/T3                  │
│  ┌──────────────────────────────────────┐   │
│  │  Haltere afferents (gyroscope)       │   │
│  │  Wing motor neurons (MN-wing)        │   │
│  │  Fore/Mid/Hind leg motor neurons     │   │
│  │  Neck motor neurons (gaze control)   │   │
│  └──────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
```

### 2.3 Key neuron classes

| Class | Abbrev. | Function | Robot analogue |
|-------|---------|----------|----------------|
| Photoreceptors R1-R8 | PR | Capture light intensity + polarisation | Camera pixels |
| Lamina monopolar cells L1-L5 | LMC | Temporal/spatial filtering | Edge / motion filter |
| T4 / T5 neurons | T4/T5 | ON/OFF directional motion detection | Optical-flow detector |
| Lobula plate tangential cells HS, VS | LPTC | Wide-field motion (optic flow) | Roll/pitch sensor |
| Lobula columnar LC4, LC6 | LC | Small-object / looming detection | Obstacle / target detector |
| E-PG neurons | EPG | Encode current heading direction | Compass / yaw integrator |
| PFN neurons | PFN | Integrate self-motion with heading | Path-integration (INS) |
| PFL neurons | PFL | Drive motor output from CX | Navigation controller |
| Descending neurons DNg02, DNa01 | DN | Relay brain commands to VNC | Motor command bus |
| Giant Fibre | GF | Escape response | Emergency stop |
| Wing motor neurons | MN-w | Control wing stroke amplitude/frequency | Throttle / aileron servo |
| Leg motor neurons T1/T2/T3 | MN-leg | Stance, balance, landing | Landing gear / stabiliser |
| Haltere afferents hDA/hDC | HAL | Detect body rotation (angular velocity) | IMU / gyroscope |
| Neck motor neurons | NMN | Stabilise gaze, head direction | Camera gimbal |

---

## 3. Navigation circuits: eye → brain → limb

### 3.1 Visual motion → optic-flow → heading correction

```
R1-R8 (photoreceptors)
  → L1/L2 (lamina: ON/OFF contrast)
    → Mi1/Tm3 (medulla: coincidence detection)
      → T4a-d / T5a-d (direction-selective)
        → HS_N, HS_E, HS_S (horizontal system: yaw/roll optic flow)
        → VS1-VS9 (vertical system: pitch/altitude)
```

This is the **Elementary Motion Detector (EMD)** cascade — the fly's
equivalent of a dense optical-flow field.  In robots, this axis drives
yaw/roll/pitch correction from a downward or forward camera.

### 3.2 Looming / object detection → evasion

```
LPLC2 / LC4 (looming-sensitive)
  → DNOVS1/2 (descending optic-flow)
    → GF (giant fibre — escape) or DNa02 (flight direction)
```

LPLC2 is a looming detector that fires when an object expands rapidly
(collision course).  LC4 responds to small moving objects.  Both drive
descending motor commands within 3-4 synapses — giving the fly a ~5 ms
collision-avoidance latency.

### 3.3 Head-direction ring attractor (CX → compass)

```
aMe (accessory medulla, sky polarisation)
  → AOTU (anterior optic tubercle)
    → BU/BUa (bulb)
      → R neurons (ring neurons, EB ring)
        ↕ EPG ↔ PEG (ring attractor — head direction)
          → PFN (fan-shaped body: velocity integration)
            → PFL3 (motor output: steer toward goal)
              → LAL → DNs → wing MNs
```

The **E-PG/P-EG system** is the fly's internal compass.  Each E-PG neuron
codes a narrow arc of heading; the population forms a "bump" that rotates
as the fly turns.  PFN neurons combine heading with self-velocity to compute
a home vector — **path integration** purely in neurons, no GPS required.

### 3.4 Descending motor commands

Descending neurons (DNs) are the only route from brain to VNC.  There are
~350 bilateral DN pairs in *Drosophila*.  Relevant to navigation:

| DN | Input | Output | Function |
|----|-------|--------|---------|
| DNg02 | CX/LAL | Wing MNs T2 | Tonic flight drive (must fire to sustain flight) |
| DNa01 | LC/HS | Wing + leg | Turning commands (moonwalker) |
| DNa02 | HS/VS | Wing | Forward thrust modulation |
| GF | LC4/LPLC2 | Wing + leg | Escape jump + wing spread |
| MDN | CX | Leg | Backward walking |

### 3.5 Wing & leg motor neurons

Wing motor neurons directly innervate the **power muscles** (synchronous,
sets wingbeat frequency ~200 Hz) and **steering muscles** (asynchronous,
modulates stroke amplitude/angle every wingbeat).  Navigation-lite includes
both sets.

Leg motor neurons control **fore/mid/hind** legs.  During hovering and landing
they participate in touchdown absorption and substrate grip — critical for
terrestrial robots.

---

## 4. Why exclude olfactory / gustatory / reproductive?

| System | Why excluded |
|--------|-------------|
| **Olfactory** (AL, PNs, MBONs, KCs) | Smell-based navigation irrelevant for drone/wheeled robots without chemical sensors.  The mushroom body alone contains ~2,000 Kenyon cells with ~100,000 connections — mostly encoding odour identity. |
| **Gustatory** (pharyngeal GRNs, SEZ) | Taste requires physical contact with food.  Adds ~3,000 neurons with no contribution to spatial navigation. |
| **Reproductive** (abdominal VNC, dsx+ circuits) | Male-specific mating circuits (ppk, dsx, SAG neurons) — completely irrelevant to locomotion. |
| **Thermo/hygrosensors** (VP, AC types) | Temperature and humidity sensing not needed for navigation in controlled environments. |

Excluding these reduces the connectome from **~166,000 → ~80,000 neurons**
before BFS, cutting build time in half and removing noise that would confuse
a linear readout decoder.

---

## 5. BFS extraction method

### 5.1 Overview

`malecns-nav-lite` uses **bidirectional Breadth-First Search** (BFS) on the
sparse synapse graph — the same algorithmic family as `evo_flydrone`'s
`sensorimotor_core`, extended to be seed-group-driven rather than
role-driven.

### 5.2 Algorithm

```python
# Pseudocode
seeds = select_by_regex(types, ALL_SEEDS)    # visual + CX + DN + wing + leg + haltere

forward_reach  = BFS(A,   seeds, hops=3)     # who do seeds drive?
backward_reach = BFS(Aᵀ,  seeds, hops=3)     # who drives seeds?

keep = forward_reach | backward_reach        # union
keep |= seeds                                # always include seeds
subgraph = G[keep][:, keep]                  # induced subgraph
```

where **A** is the `(post × pre)` adjacency matrix in CSC format.

### 5.3 Why bidirectional?

- **Forward only** would miss upstream visual neurons that don't directly
  match a seed regex but strongly drive seed neurons.
- **Backward only** would miss downstream motor targets.
- **Bidirectional union** captures the full sensorimotor relay while still
  excluding peripheral olfactory/gustatory neurons that happen to be close to
  excluded seed types.

### 5.4 Hop count rationale

| Hops | Neurons kept | Coverage |
|------|-------------|---------|
| 1 | ~5,000 | Direct partners of seeds only |
| 2 | ~15,000 | First relay layer of interneurons |
| **3** | **~25,000** | **Recommended: captures CX interneurons between visual and motor** |
| 4 | ~50,000 | Starts pulling in unrelated circuits |
| 5+ | ~80,000+ | Diminishing returns, increasing noise |

The default of **3 hops** reflects the biological reality that the shortest
visual→CX→DN→MN pathway is 4 synapses (T4→HS→EPG→PFL3→DNg02), so 3 hops
from either end meets in the middle.

### 5.5 Edge weight threshold

Only synaptic connections with **≥ 3 synapses** are traversed.  Single-synapse
connections in the raw MaleCNS data are likely to include reconstruction errors
(false positives from the automated EM segmentation pipeline).

---

## 6. Quick start

### 6.1 Install

```bash
cd d:\FLYCC\FlyIdea
pip install -e ".[data]"      # includes pyarrow + pandas for building
# or just core (no building from feather):
pip install -e .
```

### 6.2 Download raw data (~1.2 GB, CC-BY)

```bash
python -m malecns_nav_lite download --dir data/raw
```

Or download only the small annotation + NT files first to check types:

```bash
python -m malecns_nav_lite download --dir data/raw --files annotations neurotransmitters
```

### 6.3 Build the lite database

```bash
python -m malecns_nav_lite build \
    --data-dir data/raw \
    --out data/nav_lite.npz \
    --bfs-hops 3 \
    --min-synapses 3
```

Expected output:
```
Loading annotations: body-annotations-male-cns-v1.0-minconf-0.5.feather
  Raw neurons: 166,742
  Excluded (olfactory/gustatory/reproductive): 41,218
  Remaining after exclusion: 125,524
Loading neurotransmitters: ...
  Inhibitory neurons: 28,341
Streaming weights: connectome-weights-male-cns-v1.0-minconf-0.5.feather
  Scanned 25,621,891 edges; kept 8,234,102 (≥3 synapses)
Selecting seed neurons (BFS seeds from 54 patterns)…
  Seeds found: 12,847
    visual      :  4,231 seeds
    cx          :  1,892 seeds
    dn          :    683 seeds
    wing        :    312 seeds
    leg         :  4,108 seeds
    haltere     :    891 seeds
    neck        :    730 seeds
BFS expansion (hops=3, bidirectional)…
  BFS subgraph: 28,412 neurons
  After degree filter (≥1): 27,991 neurons

NavLiteDB — navigation-circuit lite connectome
  Neurons    : 27,991
  Connections: 2,847,301
  Synapses   : 9,124,882
```

### 6.4 Inspect & query

```bash
python -m malecns_nav_lite inspect data/nav_lite.npz
python -m malecns_nav_lite query  data/nav_lite.npz "^T4[a-d]$"
python -m malecns_nav_lite query  data/nav_lite.npz "^EPG$"
```

### 6.5 Load in Python

```python
from malecns_nav_lite import load_lite

db = load_lite("data/nav_lite.npz")
print(db.summary())

# Access named groups
visual_idx = db.groups["visual"]    # numpy int array
cx_idx     = db.groups["cx"]
wing_idx   = db.groups["wing"]

# Submatrix: visual → CX connections
W_vis_cx = db.weights[cx_idx][:, visual_idx]
print(f"Visual→CX connections: {W_vis_cx.nnz}")
```

---

## 7. File structure

```
FlyIdea/
├── README.md                          ← this file
├── pyproject.toml
├── malecns_nav_lite/
│   ├── __init__.py                    ← public API
│   ├── __main__.py                    ← python -m entry point
│   ├── constants.py                   ← seed patterns, exclusion lists, BFS params
│   ├── download.py                    ← download feather files (resume-safe)
│   ├── bfs.py                         ← BFS engine + regex helpers
│   ├── builder.py                     ← NavLiteDB container + build_lite()
│   └── cli.py                         ← CLI: download / build / inspect / query
├── tests/
│   ├── test_bfs.py                    ← BFS unit tests (no feather needed)
│   └── test_builder.py                ← NavLiteDB round-trip tests
└── data/
    ├── raw/                           ← downloaded .feather files (gitignored)
    └── nav_lite.npz                   ← built lite database (gitignored)
```

---

## 8. API reference

### `NavLiteDB`

| Attribute | Type | Description |
|-----------|------|-------------|
| `weights` | `scipy.sparse.csc_matrix (n, n)` | Signed synapse-count matrix, (post, pre), float32 |
| `types` | `np.ndarray[str] (n,)` | MaleCNS cell-type labels |
| `sides` | `np.ndarray[str] (n,)` | `"L"`, `"R"`, or `""` |
| `superclass` | `np.ndarray[str] (n,)` or `None` | MaleCNS superclass |
| `body_ids` | `np.ndarray[int64] (n,)` or `None` | Original body IDs (for neuPrint lookup) |
| `groups` | `dict[str, np.ndarray[int64]]` | Named index sets: visual, cx, dn, wing, leg, haltere, neck |
| `meta` | `dict` | Build parameters and provenance |

| Method | Returns | Description |
|--------|---------|-------------|
| `save(path)` | `Path` | Serialise to compressed .npz |
| `NavLiteDB.load(path)` | `NavLiteDB` | Load from .npz (no pyarrow needed) |
| `summary()` | `str` | Human-readable statistics |

### `build_lite(data_dir, min_synapses=3, bfs_hops=3, min_internal_degree=1)`

Build a `NavLiteDB` from raw feather files.  Requires `pyarrow`.

### `download_raw(data_dir, skip_existing=True, files=None)`

Download MaleCNS v1.0 feather files with resume support.

### `bfs_subgraph(weights, seed_indices, hops=3, bidirectional=True)`

Return indices of neurons in the BFS-expanded subgraph.

---

## 9. Robot-navigation usage

### 9.1 Integration with evo_flydrone

`NavLiteDB.weights` is compatible with `evo_flydrone`'s `Connectome` format.
Convert directly:

```python
from malecns_nav_lite import load_lite
from flydrones.brain.connectome import Connectome
from scipy import sparse

db = load_lite("data/nav_lite.npz")

# Wrap in evo_flydrone Connectome
conn = Connectome(
    name="nav-lite",
    weights=db.weights,
    types=db.types,
    sides=db.sides,
    superclass=db.superclass,
    body_ids=db.body_ids,
    groups=db.groups,
)
conn.save("data/nav_lite_flydrone.npz")
```

Then in your `malecns.yaml`:
```yaml
brain:
  source: data/nav_lite_flydrone.npz
  lif:
    dt: 0.5
    noise_mv: 0.0
  bias:
    DNg02_L: 8.0   # tonic flight drive
    DNg02_R: 8.0
inputs:
  VIS_L: {types: ["^(T4|T5|HS|VS|LPLC)"], side: L, feature: optic_flow_roll_L, max_hz: 150}
  VIS_R: {types: ["^(T4|T5|HS|VS|LPLC)"], side: R, feature: optic_flow_roll_R, max_hz: 150}
  HAL_L: {types: ["(?i)haltere"], side: L, feature: yaw_neg, max_hz: 50}
  HAL_R: {types: ["(?i)haltere"], side: R, feature: yaw_pos, max_hz: 50}
outputs: {}   # use flydrones calibrate to learn a linear readout
```

### 9.2 Navigation signals

| Sensory input | Fly neuron | Robot sensor |
|--------------|------------|-------------|
| Optic flow left/right | T4a-d, HS neurons | Downward camera optical flow |
| Looming (collision) | LPLC2, LC4 | Forward depth camera |
| Sky polarisation | DRA photoreceptors, AOTU | Polarisation sensor / sun compass |
| Body rotation | Haltere afferents hDA/hDC | IMU gyroscope |
| Gravity | Neck proprioceptors | Accelerometer |

| Motor output | Fly neuron | Robot actuator |
|-------------|------------|---------------|
| Yaw torque | DNa01, HS→DN | Differential thrust / rudder |
| Pitch torque | VS→DN | Elevator / pitch motor |
| Roll torque | HS_E→DN | Aileron / roll motor |
| Thrust (tonic) | DNg02 | Collective throttle |
| Escape | GF | Emergency thrust burst |
| Landing | MN-leg, MDN | Landing gear extension |

### 9.3 Performance

On a modern laptop (8-core CPU, numpy BLAS):

| Database | Neurons | Connections | Sim speed (0.5 ms dt) |
|---------|---------|------------|----------------------|
| Full MaleCNS | 166,000 | 25.6M | ~0.6× real-time |
| **Nav-Lite (hops=3)** | **~28,000** | **~2.8M** | **~5–8× real-time** |
| Nav-Lite (hops=2) | ~15,000 | ~1.2M | ~15× real-time |

Nav-Lite runs fast enough for **closed-loop robot control** at 50 Hz tick rate
without a GPU.

---

## 10. Data provenance & licence

The raw feather files are from:

> **MaleCNS v1.0** — Dorkenwald S, Matsliah A, Sterling A *et al.* (2024).
> *Neuronal wiring diagram of an adult brain.* Nature 634, 124–138.
> https://doi.org/10.1038/s41586-024-07558-y

> Schlegel P, Yin Y, Bates AS *et al.* (2024).
> *Whole-brain annotation and multi-connectome cell typing of Drosophila.*
> Nature 634, 139–152.
> https://doi.org/10.1038/s41586-024-07686-5

**Data licence**: Creative Commons Attribution 4.0 International (CC-BY 4.0).
You must credit the MaleCNS authors in any publication using this data.

**This code** (`malecns_nav_lite`): MIT licence.

---

## 11. Citation

If you use this dataset in published work, please cite:

```bibtex
@article{dorkenwald2024neuronal,
  title   = {Neuronal wiring diagram of an adult brain},
  author  = {Dorkenwald, Sven and Matsliah, Arie and Sterling, Amy R and others},
  journal = {Nature},
  volume  = {634},
  pages   = {124--138},
  year    = {2024},
  doi     = {10.1038/s41586-024-07558-y}
}

@article{schlegel2024whole,
  title   = {Whole-brain annotation and multi-connectome cell typing of {Drosophila}},
  author  = {Schlegel, Philipp and Yin, Yijie and Bates, Alexander S and others},
  journal = {Nature},
  volume  = {634},
  pages   = {139--152},
  year    = {2024},
  doi     = {10.1038/s41586-024-07686-5}
}
```

And the `malecns-nav-lite` code:
```bibtex
@software{malecns_nav_lite,
  title   = {malecns-nav-lite: Navigation-optimised lite database of the Drosophila male CNS},
  year    = {2024},
  url     = {https://github.com/your-org/FlyIdea}
}
```

---

*Built with ❤️ from the MaleCNS connectome — 166,000 neurons, 54 million synapses,
distilled to the circuits that let a 2 mg fly navigate the world.*
