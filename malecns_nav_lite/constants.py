"""
constants.py — Cell-type regex patterns and superclass filters for the
navigation-circuit extraction from MaleCNS v1.0.

Biological rationale
--------------------
The Drosophila male CNS contains ~166,000 neurons.  For robot navigation we
care about the visual → central-complex → descending-motor axis:

  VISUAL INPUTS (lamina / medulla / lobula / AOTU)
    → CENTRAL COMPLEX (CX: EB, PB, FB, NO)  — head-direction & path integration
    → DESCENDING NEURONS (DNs)              — brain → VNC relay
    → WING motor (MN-wing) + FORELEG / MIDLEG / HINDLEG motor (MN-leg)

EXCLUDED circuits (irrelevant to drone navigation):
  • Olfactory: antennal-lobe (AL), projection neurons (PN), mushroom body (MB)
    except where the MB also receives visual input
  • Gustatory: pharyngeal / labellar GRNs, SEZ gustatory neurons
  • Reproductive: abdominal ganglia circuits involved in mating / oviposition
  • Hygro / thermo sensors unrelated to flight control

Key neuron classes kept
-----------------------
  Superclass filters (MaleCNS annotation column ``superclass``):
    sensory   → only visual (Ry*, R[1-8], Mi, Tm, T4, T5, LPLC, LC, LLPC …)
    central   → central complex (EPG, PEG, PFN, PFL, FC, hΔ, Delta7 …)
    ascending → ascending neurons relaying VNC feedback to brain
    descending→ DN: brain → VNC efferents
    motor     → MN-wing, MN-leg (all limbs)

  Type-name regex patterns used for seeding BFS:
    VISUAL_SEED    — photoreceptors / medulla / lobula complex cells
    CX_SEED        — central-complex interneurons
    DN_SEED        — descending neurons (flight-relevant)
    WING_SEED      — wing motor neurons
    LEG_SEED       — leg motor neurons (foreleg, midleg, hindleg)

EXCLUDED_SUPERCLASS: superclass labels to drop entirely.
EXCLUDED_TYPE_REGEX: cell-type patterns whose neurons are always excluded.
"""

# ---------------------------------------------------------------------------
# Google-storage source (CC-BY 4.0)
# ---------------------------------------------------------------------------
MALECNS_BASE = (
    "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"
)
MALECNS_FILES = {
    "annotations":      "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "neurotransmitters": "body-neurotransmitters-male-cns-v1.0.feather",
    "weights":          "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
}

# ---------------------------------------------------------------------------
# Superclass-level exclusions  (MaleCNS ``superclass`` annotation column)
# ---------------------------------------------------------------------------
EXCLUDED_SUPERCLASS = {
    # Glia are structural, not information-processing
    "glia",
    # Gustatory periphery
    "gustatory",
    # Reproductive / abdominal-specific
    "reproductive",
    # Olfactory periphery — antennal-lobe input neurons / ORNs
    "olfactory",
}

# ---------------------------------------------------------------------------
# Fine-grained type-regex exclusions applied AFTER superclass filtering
# ---------------------------------------------------------------------------
# Pattern:  if re.search(pattern, cell_type, re.IGNORECASE) -> exclude
EXCLUDED_TYPE_REGEX: list[str] = [
    # Olfactory projection / local neurons
    r"^(vPN|mPN|lPN|adPN|lvPN|DC[1-4]|VA|VC|VL|VM|DL|DC|DM|DA)\d",  # AL PN types
    r"^(LN|AL-LN)",                     # antennal-lobe local neurons
    r"^(MBON|KCab|KCg|KCab-c|APL)",     # mushroom body output / Kenyon cells
    r"^(OA-|TRP-|SMP-DM)",              # modulatory types specific to courtship
    # Gustatory
    r"(?i)gustatory",
    r"(?i)pharyn",
    r"(?i)labellar",
    r"^(GRN|Gr[0-9])",                  # gustatory receptor neurons
    # Reproductive / abdominal motor
    r"(?i)abdominal",
    r"(?i)oviposit",
    r"(?i)copulat",
    r"(?i)genital",
    r"(?i)spermath",
    r"^(ppk|dsx)",                      # sex-specific sensory neurons
    # Thermal / hygro sensors (not relevant to navigation)
    r"(?i)hygro",
    r"(?i)thermosens",
    r"^(VP[0-9]|AC[0-9])",              # VP/AC: antenna-3 thermo/hygro
]

# ---------------------------------------------------------------------------
# SEED neuron patterns — BFS starts from these groups
# ---------------------------------------------------------------------------
# Visual input layer: photoreceptors + early visual
VISUAL_SEED: list[str] = [
    r"^R[1-8]$",          # photoreceptors R1-R8
    r"^Lai$",             # lamina intrinsic
    r"^L[1-5]$",          # lamina monopolar cells
    r"^Mi\d",             # medulla intrinsic (Mi1, Mi4 … motion detection)
    r"^Tm\d",             # transmedullary (Tm1-Tm20 …)
    r"^T4[a-d]$",         # T4: ON-direction selectivity
    r"^T5[a-d]$",         # T5: OFF-direction selectivity
    r"^LC\d",             # lobula columnar (LC4, LC6, LC10 … object detection)
    r"^LPLC\d",           # lobula-plate / lobula columnar
    r"^LLPC\d",           # lobula-plate / lobula columnar
    r"^LT\d",             # lobula tangential
    r"^HS[NECS]$",        # horizontal system (wide-field motion)
    r"^VS\d$",            # vertical system
    r"^AOTU\d",           # anterior optic tubercle — sky polarization
    r"^aMe\d",            # accessory medulla
]

# Central complex: head-direction / path-integration
CX_SEED: list[str] = [
    r"^EPG$",             # E-PG: ring attractor (head-direction)
    r"^PEG$",             # P-EG: feedback
    r"^PFN[a-z]",         # PFN: path-integration fan neurons
    r"^PFL[1-4]$",        # PFL: motor output from FB
    r"^FC[1-3]$",         # fan-shaped body columns
    r"^FR[1-2]$",         # fan-shaped body ring
    r"^hDelta[A-F]$",     # hDelta: columnar FB neurons
    r"^Delta7$",          # protocerebral bridge interneuron
    r"^ExR\d",            # extrinsic ring neurons
    r"^LNO[1-4]$",        # lateral neuropil of the ocelli
    r"^LAL-",             # lateral accessory lobe (CX output relay)
    r"^IbSps\d",          # inferior bridge
]

# Descending neurons: brain → VNC (flight motor relay)
DN_SEED: list[str] = [
    r"^DNa01$",           # moonwalker DN (turns)
    r"^DNa02$",           # fast forward flight
    r"^DNb01$",           # stopping
    r"^DNg\d",            # DNg family (flight initiation/maintenance)
    r"^MDN$",             # moonwalker DN
    r"^oviDN$",           # excluded later via reproductive filter
    r"^GF$",              # giant fibre (escape)
    r"^DG[Ia]\d",         # descending from gnathal ganglia
    r"^DN[a-z]\d",        # catch-all DN pattern — filtered by exclusion lists
]

# Wing / flight motor neurons (VNC)
WING_SEED: list[str] = [
    r"(?i)^MN-wing",
    r"(?i)^wing.*MN",
    r"(?i)DLMn",          # Dorsal Longitudinal Muscle MN
    r"(?i)DVMn",          # Dorsal Ventral Muscle MN
    r"(?i)DTTMn",
    r"(?i)Pleural.*MN",
    r"(?i)Tergopleural.*MN",
    r"(?i)Tergotr.*MN",
    r"^b1MN$",            # basalare muscle 1
    r"^b2MN$",
    r"^b3MN$",
    r"^I1MN$",
    r"^ps1MN$",
    r"^ax\dMN$",          # axillary sclerite muscles
    r"(?i)hg\dMN$",       # hingeplate
    r"^MNad\d+",
]

# Leg motor neurons (VNC)
LEG_SEED: list[str] = [
    r"(?i)flexor MN",     # Ti flexor, Tr flexor, Acc. ti flexor
    r"(?i)extensor MN",   # Ti extensor, Tr extensor
    r"(?i)reductor MN",   # Fe reductor
    r"(?i)depressor MN",  # Ta depressor
    r"(?i)rotator MN",    # Sternal rotator
    r"(?i)Sternotr.*MN",  # Sternotrochanter MN
    r"(?i)ltm.*MN",
    r"(?i)leg.*MN",
    r"(?i)^(f|m|h)Leg",
    r"(?i)^(fore|mid|hind)leg",
    r"^(fl|ml|hl)[A-Z]{2,}MN",
]

# Haltere (gyroscope) afferents and interneurons
HALTERE_SEED: list[str] = [
    r"(?i)haltere",
    r"^hDA\d",            # haltere dorsal anterior
    r"^hDC\d",            # haltere dorsal central
]

# Neck motor neurons (head stabilisation → gaze control)
NECK_SEED: list[str] = [
    r"(?i)neck.*MN",
    r"(?i)^NMN\d",
    r"(?i)^cervical",
    r"^CvN\d",            # Cervical Nerve motor neurons
]

# All seeds combined — starting nodes for BFS
ALL_SEEDS: list[str] = (
    VISUAL_SEED + CX_SEED + DN_SEED + WING_SEED + LEG_SEED + HALTERE_SEED + NECK_SEED
)

# ---------------------------------------------------------------------------
# BFS parameters
# ---------------------------------------------------------------------------
# How many synaptic hops from any seed neuron to include in the subgraph.
# 3 hops captures most relevant interneurons between vision and motor output.
BFS_HOPS: int = 3

# Minimum synapse count on an edge to be traversed by BFS
MIN_SYNAPSES: int = 3

# After BFS, drop any neuron that has fewer than this many connections
# within the subgraph (removes isolated stragglers captured by wide patterns)
MIN_INTERNAL_DEGREE: int = 1

# ---------------------------------------------------------------------------
# Neurotransmitter → synaptic sign (follows Shiu et al. 2024)
# ---------------------------------------------------------------------------
NT_SIGN: dict[str, float] = {
    "acetylcholine": 1.0,
    "ach": 1.0,
    "gaba": -1.0,
    "glutamate": -1.0,
    "glu": -1.0,
    "histamine": -1.0,
    "his": -1.0,
    "dopamine": 1.0,
    "da": 1.0,
    "serotonin": 1.0,
    "5ht": 1.0,
    "octopamine": 1.0,
    "oa": 1.0,
}
