"""
BioCloud Quick Start
====================

Run this file to generate all figures from the paper.

Usage:
    python examples/quickstart.py
    python examples/quickstart.py MCF7
    python examples/quickstart.py configs/my_line.json
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

from biocloud import BioCloud

def main():
    cell_line = sys.argv[1] if len(sys.argv) > 1 else "MCF7"

    print("BioCloud Quick Start")
    print("=" * 40)

    sim = BioCloud(cell_line, grid_size=80)

    # 1. I-V curve
    print("\n[1/5] I-V curve...")
    sim.plot_iv()

    # 2. Critical size
    print("\n[2/5] Critical size scan...")
    radii, ratios = sim.critical_size_scan()

    # Find critical R
    for r, ratio in zip(radii, ratios):
        if ratio is not None and not (ratio != ratio):
            if ratio > 0.5:
                print(f"\n  Critical radius: ~{r} cells "
                      f"(first R with ratio > 0.5)")
                break

    # 3. Phase diagram
    print("\n[3/5] Phase diagram...")
    sim.phase_diagram()

    # 4. Instructor cells
    print("\n[4/5] Instructor comparison...")
    sim.instructor_comparison(cloud_radius=8)

    # 5. Therapy
    print("\n[5/5] Therapy comparison...")
    sim.therapy_comparison(cloud_radius=8)

    # Summary
    sim.summary()

    # Save everything
    sim.save_config("output_config.json")
    sim.save_results("output_results.json")

    print("\nDone! Check generated .png files.")

if __name__ == "__main__":
    main()
