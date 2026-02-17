"""
BioCloud Simulator v1.0
=======================

Bistable lattice model of bioelectric cancer cloud
normalization with instructor cells.

Based on:
  - Carvalho 2021 (Sci Rep 11:13607) — bistable framework
  - Cervera & Mafe 2016 (Sci Rep 6:20403) — GJ coupling
  - Chernet & Levin 2013 (Dis Model Mech 6:595) — instructor cells

New contributions:
  - Instructor cells in phase transition model
  - Five therapeutic strategy comparisons
  - Open-source, parameterizable

Usage:
    from biocloud import BioCloud
    sim = BioCloud("MCF7")
    sim.full_analysis(cloud_radius=8)

Author: Ulugbek Khamidov, Alet Project
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import json
import time
import os

__version__ = "1.0.0"

# ============================================
# CELL LINE CONFIGURATIONS
# ============================================

CELL_LINES = {
    "MCF7": {
        "name": "MCF-7 (Luminal A breast cancer)",
        "source": "PMID 7559799, PMID 11697078",
        "G_pol": 3.0,
        "G_dep": 4.25,
        "V_half": -25.0,
        "z": 3.0,
        "V_T": 26.0,
        "E_pol": -90.0,
        "E_dep": 0.0,
        "C_m": 20.0,
        "gj_healthy": 2.0,
        "gj_cancer": 0.5,
        "gj_instructor": 1.5,
        "gj_V0": 26.0,
        "instructor_vmem_offset": 15.0,
        "vmem_verified": True,
        "conductance_estimated": True,
        "gj_same_line": False,
        "notes": (
            "Vmem from microelectrode: D group -9mV, "
            "H group -40mV (Wonderlin 1995, PMID 7559799). "
            "IK from patch-clamp: 9.4 pA/pF G0/G1, "
            "30.2 pA/pF G1 progressing (PMID 11697078). "
            "Conductances estimated assuming C_m=20pF. "
            "GJ from Cx43 general data (Bukauskas 2002), "
            "not MCF-7 specific. "
            "Dye transfer: 21.6% at 10h (Gava 2018)."
        )
    },
    "SKBR3": {
        "name": "SKBR3 (HER2+ breast cancer)",
        "source": "Marino et al. 2018 Sci Rep 8:6257",
        "G_pol": 3.0,
        "G_dep": 4.25,
        "V_half": -25.0,
        "z": 3.0,
        "V_T": 26.0,
        "E_pol": -90.0,
        "E_dep": 0.0,
        "C_m": 20.0,
        "gj_healthy": 2.0,
        "gj_cancer": 0.3,
        "gj_instructor": 1.0,
        "gj_V0": 26.0,
        "instructor_vmem_offset": 15.0,
        "vmem_verified": False,
        "conductance_estimated": True,
        "gj_same_line": False,
        "notes": (
            "Kir3.2 upregulation observed with BaTiO3+US "
            "(Marino 2018). No direct Vmem measurement. "
            "Parameters copied from MCF-7 as placeholder. "
            "GJ reduced: HER2+ lines typically have less "
            "Cx43 than luminal A."
        )
    },
    "CUSTOM": {
        "name": "Custom cell line (edit values)",
        "source": "YOUR CITATION HERE",
        "G_pol": 3.0,
        "G_dep": 4.0,
        "V_half": -25.0,
        "z": 3.0,
        "V_T": 26.0,
        "E_pol": -90.0,
        "E_dep": 0.0,
        "C_m": 20.0,
        "gj_healthy": 2.0,
        "gj_cancer": 0.5,
        "gj_instructor": 1.0,
        "gj_V0": 26.0,
        "instructor_vmem_offset": 15.0,
        "vmem_verified": False,
        "conductance_estimated": True,
        "gj_same_line": False,
        "notes": "Replace all values with your data."
    }
}


class BioCloud:
    """
    Bioelectric cancer cloud simulator.

    Models a 2D lattice of cells with bistable membrane
    potential dynamics coupled by gap junctions.

    Parameters can be loaded from:
      - Built-in cell lines: "MCF7", "SKBR3", "CUSTOM"
      - JSON config file: "path/to/config.json"
      - Python dict: passed as config argument

    Example:
        sim = BioCloud("MCF7", grid_size=80)
        sim.full_analysis(cloud_radius=8)
    """

    def __init__(self, cell_line="MCF7", grid_size=80,
                 config=None):
        self.N = grid_size

        if config is not None:
            self.config = config
        elif isinstance(cell_line, str) and \
             cell_line.endswith(".json"):
            with open(cell_line) as f:
                self.config = json.load(f)
        elif cell_line in CELL_LINES:
            self.config = CELL_LINES[cell_line].copy()
        else:
            raise ValueError(
                f"Unknown: '{cell_line}'. "
                f"Use {list(CELL_LINES.keys())} "
                f"or a .json path.")

        self._unpack()
        self._find_attractors()
        self.results = {}

        print(f"BioCloud v{__version__}")
        print(f"  Cell line: {self.config['name']}")
        print(f"  Grid: {self.N}x{self.N} "
              f"({self.N**2} cells)")
        print(f"  Attractors: {self.attractors} mV")
        if self.barriers:
            print(f"  Barriers: {self.barriers} mV")
        if not self.config.get("vmem_verified", False):
            print(f"  WARNING: Vmem not verified "
                  f"by primary patch-clamp")

    def _unpack(self):
        c = self.config
        self.G_pol = c["G_pol"]
        self.G_dep = c["G_dep"]
        self.V_half = c["V_half"]
        self.z = c["z"]
        self.V_T = c["V_T"]
        self.E_pol = c["E_pol"]
        self.E_dep = c["E_dep"]
        self.C_m = c["C_m"]
        self.gj_healthy = c["gj_healthy"]
        self.gj_cancer = c["gj_cancer"]
        self.gj_instructor = c["gj_instructor"]
        self.gj_V0 = c["gj_V0"]
        self.inst_offset = c.get("instructor_vmem_offset", 15.0)

    def _ionic_current(self, V):
        f_dep = 1.0 / (1.0 + np.exp(
            -self.z * (V - self.V_half) / self.V_T))
        f_pol = 1.0 / (1.0 + np.exp(
            self.z * (V - self.V_half) / self.V_T))
        return (self.G_pol * f_pol * (V - self.E_pol) +
                self.G_dep * f_dep * (V - self.E_dep))

    def _find_attractors(self):
        V = np.linspace(-100, 30, 2000)
        I = self._ionic_current(V)
        self.attractors = []
        self.barriers = []
        for i in range(len(I) - 1):
            if I[i] * I[i + 1] < 0:
                vz = V[i] - I[i] * (V[i+1] - V[i]) / \
                     (I[i+1] - I[i])
                if 0 < i < len(I) - 2:
                    dIdV = (I[i+1] - I[i-1]) / \
                           (V[i+1] - V[i-1])
                    if dIdV > 0:
                        self.attractors.append(round(vz, 1))
                    else:
                        self.barriers.append(round(vz, 1))

        if len(self.attractors) < 2:
            print(f"  WARNING: {len(self.attractors)} "
                  f"attractor(s): {self.attractors}")
            print(f"  Adjust G_pol, G_dep, or V_half")

    def _create_grid(self, cloud_radius,
                     include_instructors=False,
                     instructor_width=2):
        center = self.N // 2
        healthy_v = self.attractors[0] \
            if self.attractors else -70.0
        cancer_v = self.attractors[-1] \
            if self.attractors else -10.0

        Vmem = np.full((self.N, self.N), healthy_v)
        g_gap = np.full((self.N, self.N), self.gj_healthy)
        cell_type = np.zeros((self.N, self.N), dtype=int)

        Y, X = np.ogrid[:self.N, :self.N]
        dist = np.sqrt((Y - center)**2 + (X - center)**2)

        cancer_mask = dist < cloud_radius
        Vmem[cancer_mask] = cancer_v
        g_gap[cancer_mask] = self.gj_cancer
        cell_type[cancer_mask] = 1

        if include_instructors:
            inst_mask = ((dist >= cloud_radius) &
                         (dist < cloud_radius + instructor_width))
            Vmem[inst_mask] = cancer_v + self.inst_offset
            g_gap[inst_mask] = self.gj_instructor
            cell_type[inst_mask] = 2

        return Vmem, g_gap, cell_type

    def run(self, cloud_radius=8, g_gap_boundary=0.5,
            include_instructors=False, instructor_width=2,
            n_steps=6000, dt=0.01, record_every=100,
            label=None, verbose=False):
        """
        Run a single simulation.

        Args:
            cloud_radius: cancer cloud radius in cells
            g_gap_boundary: GJ conductance at boundary (nS)
            include_instructors: add instructor ring
            instructor_width: instructor ring width
            n_steps: total simulation steps
            dt: time step
            record_every: snapshot interval
            label: name for storing result
            verbose: print progress

        Returns:
            dict with history, cloud_sizes, ratio, etc.
        """
        Vmem, g_gap, cell_type = self._create_grid(
            cloud_radius, include_instructors,
            instructor_width)

        initial = int(np.sum(cell_type >= 1))
        threshold = self.barriers[0] \
            if self.barriers else -30.0

        history = [Vmem.copy()]
        cloud_sizes = []
        t_start = time.time()

        for step in range(n_steps):
            I_ionic = self._ionic_current(Vmem)

            I_gap = np.zeros_like(Vmem)
            for di, dj in [(0, 1), (0, -1),
                           (1, 0), (-1, 0)]:
                V_nb = np.roll(
                    np.roll(Vmem, -di, axis=0),
                    -dj, axis=1)
                g_nb = np.roll(
                    np.roll(g_gap, -di, axis=0),
                    -dj, axis=1)
                t_nb = np.roll(
                    np.roll(cell_type, -di, axis=0),
                    -dj, axis=1)

                ge = np.minimum(g_gap, g_nb)
                bnd = cell_type != t_nb
                ge = np.where(bnd, g_gap_boundary, ge)

                dV = Vmem - V_nb
                gating = 2.0 / (1.0 + np.cosh(
                    np.clip(dV / self.gj_V0, -10, 10)))

                I_gap += ge * gating * (V_nb - Vmem)

            Vmem = Vmem + (-I_ionic + I_gap) * dt / self.C_m
            Vmem = np.clip(Vmem, -100, 50)

            Vmem[0, :] = Vmem[1, :]
            Vmem[-1, :] = Vmem[-2, :]
            Vmem[:, 0] = Vmem[:, 1]
            Vmem[:, -1] = Vmem[:, -2]

            if step % record_every == 0:
                history.append(Vmem.copy())
                cloud_sizes.append(
                    int(np.sum(Vmem > threshold)))

        elapsed = time.time() - t_start
        final = cloud_sizes[-1] if cloud_sizes else 0
        ratio = final / max(initial, 1)

        result = {
            "history": history,
            "final_Vmem": Vmem.copy(),
            "cloud_sizes": cloud_sizes,
            "initial_cloud": initial,
            "final_cloud": final,
            "ratio": ratio,
            "elapsed": elapsed,
            "params": {
                "R": cloud_radius,
                "g_gap_boundary": g_gap_boundary,
                "instructors": include_instructors,
                "n_steps": n_steps
            }
        }

        if label:
            self.results[label] = result
        if verbose:
            print(f"  R={cloud_radius}, "
                  f"g={g_gap_boundary}: "
                  f"ratio={ratio:.3f} "
                  f"({elapsed:.1f}s)")
        return result

    def plot_iv(self, save=True, filename="iv_curve.png"):
        """Plot I-V curve and phase portrait."""
        V = np.linspace(-100, 30, 1000)
        I = self._ionic_current(V)

        fig, (ax1, ax2) = plt.subplots(1, 2,
                                        figsize=(14, 5))

        ax1.plot(V, I, "b-", lw=2)
        ax1.axhline(y=0, color="k", lw=0.5)
        for v in self.attractors:
            ax1.axvline(x=v, color="green", ls="--",
                        alpha=0.7)
            ax1.annotate(f"{v:.0f} mV", xy=(v, 0),
                         xytext=(v + 3, max(I) * 0.2),
                         fontsize=10, color="green",
                         arrowprops=dict(arrowstyle="->",
                                         color="green"))
        for v in self.barriers:
            ax1.axvline(x=v, color="red", ls="--",
                        alpha=0.7)
        ax1.set_xlabel("Vmem (mV)")
        ax1.set_ylabel("I_total (pA)")
        ax1.set_title("I-V Curve")
        ax1.grid(True, alpha=0.3)

        dVdt = -I / self.C_m
        ax2.plot(V, dVdt, "r-", lw=2)
        ax2.axhline(y=0, color="k", lw=0.5)
        ax2.fill_between(V, dVdt, 0, where=dVdt > 0,
                          alpha=0.1, color="red")
        ax2.fill_between(V, dVdt, 0, where=dVdt < 0,
                          alpha=0.1, color="blue")
        ax2.set_xlabel("Vmem (mV)")
        ax2.set_ylabel("dV/dt (mV/ms)")
        ax2.set_title("Phase Portrait")
        ax2.grid(True, alpha=0.3)

        plt.suptitle(self.config["name"], fontsize=13,
                     fontweight="bold")
        plt.tight_layout()
        if save:
            plt.savefig(filename, dpi=150,
                        bbox_inches="tight")
        plt.show()

    def plot_evolution(self, result, save=True,
                       filename="evolution.png"):
        """Plot cloud evolution snapshots."""
        hist = result["history"]
        n = len(hist)
        idx = [0, n // 4, n // 2, 3 * n // 4, -1]
        titles = ["Initial", "25%", "50%", "75%", "Final"]

        if self.barriers:
            vc = (self.attractors[0] + self.attractors[-1]) / 2
        else:
            vc = -30.0
        norm = TwoSlopeNorm(vmin=-100, vcenter=vc, vmax=30)

        fig, axes = plt.subplots(1, len(idx),
                                  figsize=(4 * len(idx), 4))
        for ax, i, t in zip(axes, idx, titles):
            im = ax.imshow(hist[i], cmap="RdBu_r", norm=norm)
            cloud_n = np.sum(hist[i] > (
                self.barriers[0] if self.barriers else -30))
            ax.set_title(f"{t}\n({cloud_n} cells)")
        plt.colorbar(im, ax=axes, label="Vmem (mV)",
                     shrink=0.8)
        R = result["params"]["R"]
        plt.suptitle(f"Cloud Evolution R={R}",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        if save:
            plt.savefig(filename, dpi=150,
                        bbox_inches="tight")
        plt.show()

    def critical_size_scan(self, radii=None,
                            g_gap_boundary=0.5,
                            n_steps=6000, save=True,
                            filename="critical_size.png"):
        """Scan cloud radius to find critical size."""
        if radii is None:
            radii = [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 18]

        print(f"\nCritical size scan "
              f"(g_gap={g_gap_boundary})...")
        ratios = []
        for R in radii:
            if R >= self.N // 2 - 3:
                ratios.append(np.nan)
                continue
            r = self.run(cloud_radius=R,
                         g_gap_boundary=g_gap_boundary,
                         n_steps=n_steps)
            ratios.append(r["ratio"])
            print(f"  R={R:2d}: ratio={r['ratio']:.3f}")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(radii, ratios, "bo-", lw=2, markersize=8)
        ax.axhline(y=0.5, color="r", ls="--", alpha=0.5,
                   label="50% threshold")
        ax.axhline(y=0.1, color="g", ls="--", alpha=0.5,
                   label="90% normalization")
        ax.set_xlabel("Cloud radius R (cells)", fontsize=12)
        ax.set_ylabel("Final/Initial ratio", fontsize=12)
        ax.set_title("Critical Cloud Size", fontsize=13)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save:
            plt.savefig(filename, dpi=150,
                        bbox_inches="tight")
        plt.show()
        return radii, ratios

    def phase_diagram(self, radii=None, g_gaps=None,
                      n_steps=5000, save=True,
                      filename="phase_diagram.png"):
        """Compute R vs g_gap phase diagram."""
        if radii is None:
            radii = [2, 3, 5, 7, 10, 13, 16, 20]
        if g_gaps is None:
            g_gaps = [0.05, 0.1, 0.2, 0.5,
                      1.0, 2.0, 5.0]

        total = len(radii) * len(g_gaps)
        ratios = np.zeros((len(g_gaps), len(radii)))
        count = 0

        print(f"\nPhase diagram ({total} runs)...")

        for i, g in enumerate(g_gaps):
            for j, r in enumerate(radii):
                count += 1
                if r >= self.N // 2 - 5:
                    ratios[i, j] = np.nan
                    continue
                res = self.run(cloud_radius=r,
                               g_gap_boundary=g,
                               n_steps=n_steps,
                               record_every=n_steps)
                ratios[i, j] = res["ratio"]
                if count % max(1, total // 8) == 0:
                    print(f"  {count}/{total} "
                          f"({100 * count // total}%)")

        print("  Done!")

        fig, ax = plt.subplots(figsize=(10, 7))
        im = ax.imshow(ratios, cmap="RdBu_r",
                        vmin=0, vmax=2,
                        aspect="auto", origin="lower")
        ax.set_xticks(range(len(radii)))
        ax.set_xticklabels(radii)
        ax.set_yticks(range(len(g_gaps)))
        ax.set_yticklabels(g_gaps)
        ax.set_xlabel("Cloud radius R (cells)", fontsize=13)
        ax.set_ylabel("Boundary g_gap (nS)", fontsize=13)
        ax.set_title(
            f"Phase Diagram: {self.config['name']}\n"
            f"Attractors: {self.attractors} mV",
            fontsize=13)
        plt.colorbar(im, ax=ax, label="Final/Initial")
        plt.tight_layout()
        if save:
            plt.savefig(filename, dpi=150,
                        bbox_inches="tight")
        plt.show()
        return radii, g_gaps, ratios

    def instructor_comparison(self, cloud_radius=8,
                               g_gap_boundary=0.5,
                               n_steps=6000, save=True,
                               filename="instructor.png"):
        """Compare with/without instructor cells."""
        print(f"\nInstructor comparison (R={cloud_radius})...")

        r_no = self.run(cloud_radius=cloud_radius,
                        g_gap_boundary=g_gap_boundary,
                        include_instructors=False,
                        n_steps=n_steps,
                        label="No instructors")

        r_yes = self.run(cloud_radius=cloud_radius,
                         g_gap_boundary=g_gap_boundary,
                         include_instructors=True,
                         n_steps=n_steps,
                         label="With instructors")

        print(f"  Without: ratio={r_no['ratio']:.3f}")
        print(f"  With:    ratio={r_yes['ratio']:.3f}")

        if r_no["ratio"] > 0.001:
            factor = r_yes["ratio"] / r_no["ratio"]
            print(f"  Instructors increase stability "
                  f"by {factor:.1f}x")

        fig, (ax1, ax2) = plt.subplots(1, 2,
                                        figsize=(14, 5))

        ax1.plot(r_no["cloud_sizes"], "b-", lw=2,
                 label=f"Without ({r_no['ratio']:.2f})")
        ax1.plot(r_yes["cloud_sizes"], "r-", lw=2,
                 label=f"With ({r_yes['ratio']:.2f})")
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Cloud size")
        ax1.set_title("Cloud Size Over Time")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        vc = (self.attractors[0] + self.attractors[-1]) / 2 \
            if len(self.attractors) >= 2 else -30
        norm = TwoSlopeNorm(vmin=-100, vcenter=vc, vmax=30)

        ax2.imshow(r_yes["final_Vmem"], cmap="RdBu_r",
                   norm=norm)
        ax2.set_title(f"Final State (with instructors)\n"
                      f"Cloud={r_yes['final_cloud']}")

        plt.suptitle(
            f"Instructor Cell Effect (R={cloud_radius})",
            fontsize=13, fontweight="bold")
        plt.tight_layout()
        if save:
            plt.savefig(filename, dpi=150,
                        bbox_inches="tight")
        plt.show()
        return r_no, r_yes

    def therapy_comparison(self, cloud_radius=8,
                            n_steps=6000, save=True,
                            filename="therapy.png"):
        """Compare five therapeutic strategies."""
        print(f"\nTherapy comparison (R={cloud_radius})...")

        strategies = [
            ("Untreated",
             dict(g_gap_boundary=0.1,
                  include_instructors=True)),
            ("Restore GJ (g=2)",
             dict(g_gap_boundary=2.0,
                  include_instructors=True)),
            ("Strong GJ (g=5)",
             dict(g_gap_boundary=5.0,
                  include_instructors=True)),
            ("GJ + no instructors",
             dict(g_gap_boundary=2.0,
                  include_instructors=False)),
            ("Strong GJ + no inst",
             dict(g_gap_boundary=5.0,
                  include_instructors=False)),
        ]

        colors = ["black", "blue", "cyan", "green", "red"]
        fig, ax = plt.subplots(figsize=(12, 6))

        for (name, kwargs), color in zip(strategies, colors):
            r = self.run(cloud_radius=cloud_radius,
                         n_steps=n_steps,
                         label=name, **kwargs)
            ax.plot(r["cloud_sizes"], color=color, lw=2,
                    label=f"{name} ({r['ratio']:.2f})")
            print(f"  {name}: ratio={r['ratio']:.2f}")

        ax.set_xlabel("Step", fontsize=12)
        ax.set_ylabel("Cloud size", fontsize=12)
        ax.set_title(
            f"Therapy Comparison (R={cloud_radius})",
            fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save:
            plt.savefig(filename, dpi=150,
                        bbox_inches="tight")
        plt.show()

    def full_analysis(self, cloud_radius=8,
                      save=True, prefix="biocloud"):
        """Run complete analysis suite."""
        print("=" * 60)
        print("FULL ANALYSIS")
        print("=" * 60)

        self.plot_iv(save=save,
                     filename=f"{prefix}_iv.png")

        self.critical_size_scan(
            save=save,
            filename=f"{prefix}_critical.png")

        self.phase_diagram(
            save=save,
            filename=f"{prefix}_phase.png")

        self.instructor_comparison(
            cloud_radius=cloud_radius, save=save,
            filename=f"{prefix}_instructor.png")

        self.therapy_comparison(
            cloud_radius=cloud_radius, save=save,
            filename=f"{prefix}_therapy.png")

        self.summary()

    def summary(self):
        """Print summary of all stored results."""
        print(f"\n{'=' * 60}")
        print(f"SUMMARY: {self.config['name']}")
        print(f"{'=' * 60}")
        print(f"Attractors: {self.attractors} mV")
        print(f"Barriers:   {self.barriers} mV")
        print(f"Grid:       {self.N}x{self.N}")
        print(f"\nStored runs ({len(self.results)}):")
        for name, r in self.results.items():
            print(f"  {name}: "
                  f"init={r['initial_cloud']}, "
                  f"final={r['final_cloud']}, "
                  f"ratio={r['ratio']:.3f}")
        print(f"\nSource: {self.config['source']}")
        print(f"Notes:  {self.config.get('notes', 'none')}")
        print("=" * 60)

    def save_config(self, filename):
        """Save config to JSON for reproducibility."""
        with open(filename, "w") as f:
            json.dump(self.config, f, indent=2)
        print(f"Config saved: {filename}")

    def save_results(self, filename):
        """Save numerical results (no arrays) to JSON."""
        out = {}
        for name, r in self.results.items():
            out[name] = {
                "initial_cloud": r["initial_cloud"],
                "final_cloud": r["final_cloud"],
                "ratio": r["ratio"],
                "elapsed": r["elapsed"],
                "params": r["params"]
            }
        with open(filename, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Results saved: {filename}")


# ============================================
# CLI entry point
# ============================================

if __name__ == "__main__":
    import sys

    cell_line = sys.argv[1] if len(sys.argv) > 1 else "MCF7"
    grid = int(sys.argv[2]) if len(sys.argv) > 2 else 80
    radius = int(sys.argv[3]) if len(sys.argv) > 3 else 8

    sim = BioCloud(cell_line, grid_size=grid)
    sim.full_analysis(cloud_radius=radius)
    sim.save_config("output_config.json")
    sim.save_results("output_results.json")
