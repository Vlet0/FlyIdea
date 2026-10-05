"""
malecns_nav_lite — Navigation-optimised lite database of the Drosophila male CNS.

Filters the MaleCNS v1.0 connectome (Janelia FlyEM) to the sensorimotor circuits
linking eye → brain → VNC → wing / leg motor neurons, discarding olfactory,
gustatory, and reproductive circuits that are irrelevant to aerial navigation.

BFS-based subgraph extraction mirrors the approach used in evo_flydrone.
"""

__version__ = "0.1.0"
__all__ = ["download_raw", "build_lite", "load_lite", "NavLiteDB"]
