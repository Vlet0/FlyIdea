"""
examples/02_lif_simulation.py — Run a minimal Leaky Integrate-and-Fire
simulation using the nav_lite connectome.

Requires numpy and scipy (no pyarrow needed).
Demonstrates:
  1. Loading the pre-built nav_lite.npz
  2. Injecting Poisson spike trains into visual neurons
  3. Reading out motor neuron firing rates
  4. Printing a simple "control signal" from the firing-rate readout

This is a standalone LIF implementation to avoid the evo_flydrone dependency —
plug in your own preferred simulator as needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from malecns_nav_lite import load_lite


# --------------------------------------------------------------------------- #
# Minimal LIF
# --------------------------------------------------------------------------- #

class SimpleLIF:
    """Vectorised leaky integrate-and-fire network.

    Parameters match evo_flydrone defaults for easy comparison.
    """

    def __init__(
        self,
        W: sparse.csc_matrix,
        dt_ms: float = 0.5,
        tau_m: float = 10.0,   # membrane time constant (ms)
        v_rest: float = -65.0,  # mV
        v_thresh: float = -55.0,  # mV
        v_reset: float = -70.0,   # mV
        scale: float = 0.05,      # weight → current scaling
        seed: int = 0,
    ):
        self.n = W.shape[0]
        self.W = sparse.csr_matrix(W * scale)
        self.dt = dt_ms
        self.tau = tau_m
        self.v_rest = v_rest
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.rng = np.random.default_rng(seed)
        self.v = np.full(self.n, v_rest, dtype=np.float64)
        self.spikes = np.zeros(self.n, dtype=bool)

    def step(self, input_rates: np.ndarray) -> np.ndarray:
        """Advance one dt.  input_rates: Hz per neuron (n,)."""
        # Poisson input current
        p = input_rates * self.dt / 1000.0
        noise = self.rng.uniform(0, 1, self.n) < p

        # Recurrent current from previous spikes
        I_rec = np.asarray(self.W @ self.spikes.astype(np.float64)).ravel()

        # Euler update
        dv = (self.v_rest - self.v) / self.tau * self.dt
        self.v += dv + I_rec + noise.astype(np.float64) * 1.5  # mV

        # Threshold
        self.spikes = self.v >= self.v_thresh
        self.v[self.spikes] = self.v_reset
        return self.spikes.copy()


# --------------------------------------------------------------------------- #
# Main simulation
# --------------------------------------------------------------------------- #

def main():
    db_path = Path("data/nav_lite.npz")
    if not db_path.exists():
        print("Build the database first:")
        print("  python -m malecns_nav_lite build --data-dir data/raw --out data/nav_lite.npz")
        return

    print("Loading nav_lite …")
    db = load_lite(db_path)
    print(f"  {db.n:,} neurons, {db.n_connections:,} connections")

    net = SimpleLIF(db.weights, dt_ms=0.5, seed=42)

    visual_idx = db.groups.get("visual", np.array([], dtype=np.int64))
    wing_idx   = db.groups.get("wing",   np.array([], dtype=np.int64))
    dn_idx     = db.groups.get("dn",     np.array([], dtype=np.int64))

    print(f"\nSimulating 500 ms with visual input (50 Hz left, 10 Hz right) …")
    n_steps = int(500 / 0.5)
    spike_counts = np.zeros(db.n)

    input_rates = np.zeros(db.n)

    # Left visual neurons: 50 Hz (simulating leftward optic flow)
    left_vis  = visual_idx[db.sides[visual_idx] == "L"]
    right_vis = visual_idx[db.sides[visual_idx] == "R"]

    if left_vis.size:
        input_rates[left_vis] = 50.0
    if right_vis.size:
        input_rates[right_vis] = 10.0

    for step in range(n_steps):
        spks = net.step(input_rates)
        spike_counts += spks.astype(float)

    # Convert to Hz
    rates_hz = spike_counts / (500e-3)  # spikes / second

    print("\n=== Output firing rates ===")
    print(f"  Visual neurons  — mean: {rates_hz[visual_idx].mean():.1f} Hz  max: {rates_hz[visual_idx].max():.1f} Hz")
    if dn_idx.size:
        print(f"  Descending DNs  — mean: {rates_hz[dn_idx].mean():.1f} Hz  max: {rates_hz[dn_idx].max():.1f} Hz")
    if wing_idx.size:
        print(f"  Wing motor MNs  — mean: {rates_hz[wing_idx].mean():.1f} Hz  max: {rates_hz[wing_idx].max():.1f} Hz")

    # Simple differential readout: left DN - right DN
    dn_L = dn_idx[db.sides[dn_idx] == "L"]
    dn_R = dn_idx[db.sides[dn_idx] == "R"]
    if dn_L.size and dn_R.size:
        yaw_signal = rates_hz[dn_L].mean() - rates_hz[dn_R].mean()
        print(f"\n  Yaw control signal (L-DN minus R-DN): {yaw_signal:+.2f} Hz")
        if yaw_signal > 0:
            print("  → Robot should turn LEFT")
        elif yaw_signal < 0:
            print("  → Robot should turn RIGHT")
        else:
            print("  → Robot flies straight")

    print("\nDone.")


if __name__ == "__main__":
    main()
