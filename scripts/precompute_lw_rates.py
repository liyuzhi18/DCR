"""
Precompute Lyman-Werner rate coefficients for molecular RPL calculation.

Integrates MCCC electron-impact excitation cross sections (B1Su, Bp1Su, C1Pu, D1Pu)
over a Maxwellian EEDF at divertor-relevant temperatures (0.1–20 eV).

Output: atomic_data/lw_rates.npz
  Te_eV      : (N_Te,)     temperature grid [eV]
  k_tot      : (4, 15, N_Te) rate coefficients [cm3/s], axes: (state, vi, Te)
  E_photon_eV: (4,)        representative photon energy per state [eV]
  state_names: (4,)        state labels

Usage:
  python scripts/precompute_lw_rates.py
  python scripts/precompute_lw_rates.py --show-ratios --state B1Su --vi 0
  python scripts/precompute_lw_rates.py --show-state-channels --vi 0
  python scripts/precompute_lw_rates.py --show-vf-subchannels --vi 0 --vf 1
"""

import argparse
import numpy as np
import os
import glob
from pathlib import Path

import matplotlib.pyplot as plt

# --- Configuration -----------------------------------------------------------
DATA_DIR = "atomic_data/vibex_channel/EIE_results"
OUTPUT    = "atomic_data/lw_rates.npz"
RATIO_OUTDIR = Path("output/lw_channel_ratios")

# Ungerade states that radiatively decay to X1Sg (Lyman-Werner / Werner bands)
# (state_tag_in_filename, representative_photon_energy_eV)
STATES = [
    ("B1Su",  11.18),
    ("Bp1Su", 11.87),
    ("C1Pu",  12.40),
    ("D1Pu",  14.03),
]

N_VI = 15          # H2 vibrational levels v=0..14 in the model
N_TE = 80
TE_GRID = np.logspace(np.log10(0.1), np.log10(20.0), N_TE)  # eV

# Physical constants (CGS)
ME_G   = 9.10938e-28   # electron mass [g]
EV_ERG = 1.60218e-12   # 1 eV [erg]

# -----------------------------------------------------------------------------

def maxwellian_rate(E_eV, sigma_cm2, Te_eV):
    """Compute <v*sigma> for a Maxwellian EEDF by numerical trapz integration."""
    # v(E) in cm/s
    v = np.sqrt(2.0 * E_eV * EV_ERG / ME_G)
    # Maxwellian EEDF: F(E) = (2/Te)*sqrt(E/(pi*Te))*exp(-E/Te)
    F = (2.0 / Te_eV) * np.sqrt(E_eV / (np.pi * Te_eV)) * np.exp(-E_eV / Te_eV)
    integrand = v * F * sigma_cm2
    return np.trapezoid(integrand, E_eV)


def channel_files(state, vi):
    pattern = os.path.join(DATA_DIR,
        f"MCCC-el-H2-{state}.X1Sg_vi={vi}_X_sec_vf=*.out")
    return sorted(glob.glob(pattern))


def load_channels(state, vi):
    channels = []
    for path in channel_files(state, vi):
        data = np.loadtxt(path)
        if data.ndim < 2 or data.shape[0] < 2:
            continue
        E_eV = data[:, 0]
        sigma = data[:, 1]
        mask = E_eV > 0
        if not np.any(mask):
            continue
        E_eV = E_eV[mask]
        sigma = sigma[mask]
        vf = Path(path).stem.split("vf=")[-1]
        channels.append({
            "vf": vf,
            "path": path,
            "E_eV": E_eV,
            "sigma": sigma,
        })
    return channels


def load_channel_for_vf(state, vi, vf):
    target = str(vf)
    for ch in load_channels(state, vi):
        if str(ch["vf"]) == target:
            return ch["E_eV"], ch["sigma"]
    return None


def available_states(vi):
    pattern = os.path.join(DATA_DIR, f"MCCC-el-H2-*.X1Sg_vi={vi}_X_sec_vf=*.out")
    states = set()
    for path in glob.glob(pattern):
        name = Path(path).name
        stem = name.split(".X1Sg_vi=")[0]
        states.add(stem.replace("MCCC-el-H2-", "", 1))
    return sorted(states)


def build_ratio_table(state, vi):
    channels = load_channels(state, vi)
    if not channels:
        raise SystemExit(f"No channels found for state={state}, vi={vi}")

    # Use the union of tabulated energies so ratios are evaluated pointwise in incident energy.
    grid = np.unique(np.concatenate([ch["E_eV"] for ch in channels]))
    sigma_matrix = np.zeros((len(channels), grid.size))
    for i, ch in enumerate(channels):
        sigma_matrix[i, :] = np.interp(grid, ch["E_eV"], ch["sigma"], left=0.0, right=0.0)

    sigma_total = sigma_matrix.sum(axis=0)
    ratios = np.zeros_like(sigma_matrix)
    mask = sigma_total > 0.0
    ratios[:, mask] = sigma_matrix[:, mask] / sigma_total[mask]
    return channels, grid, sigma_matrix, sigma_total, ratios


def sum_channels_for_state(state, vi):
    channels = load_channels(state, vi)
    if not channels:
        return None
    grid = np.unique(np.concatenate([ch["E_eV"] for ch in channels]))
    sigma_total = np.zeros(grid.size)
    for ch in channels:
        sigma_total += np.interp(grid, ch["E_eV"], ch["sigma"], left=0.0, right=0.0)
    return grid, sigma_total


def export_state_channel_plot(vi, outdir):
    states = available_states(vi)
    if not states:
        raise SystemExit(f"No electronic-state channels found for vi={vi}")

    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.8, 5.8))

    plotted = 0
    for state in states:
        summed = sum_channels_for_state(state, vi)
        if summed is None:
            continue
        grid, sigma_total = summed
        ax.plot(grid, np.where(sigma_total > 0.0, sigma_total, np.nan), lw=1.6, label=state)
        plotted += 1

    if plotted == 0:
        raise SystemExit(f"No plottable state-summed channels found for vi={vi}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Incident energy $E$ [eV]")
    ax.set_ylabel(r"$\sum_{v_f}\sigma_{\mathrm{state},v_i\to v_f}(E)$")
    ax.set_title(f"Electronic-state channel cross sections summed over $v_f$: vi={vi}")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, ncol=2)

    fig.tight_layout()
    pdf_path = outdir / f"lw_state_channel_cross_sections_vi{vi}.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {pdf_path}")


def export_vf_subchannel_plot(vi, vf, outdir):
    states = available_states(vi)
    if not states:
        raise SystemExit(f"No electronic-state channels found for vi={vi}")

    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.8, 5.8))

    plotted = 0
    for state in states:
        loaded = load_channel_for_vf(state, vi, vf)
        if loaded is None:
            continue
        grid, sigma = loaded
        ax.plot(grid, np.where(sigma > 0.0, sigma, np.nan), lw=1.6, label=state)
        plotted += 1

    if plotted == 0:
        raise SystemExit(f"No subchannels found for vi={vi}, vf={vf}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Incident energy $E$ [eV]")
    ax.set_ylabel(r"$\sigma_{\mathrm{state},\,v_i=%d\to v_f=%d}(E)$" % (vi, vf))
    ax.set_title(f"Subchannel cross sections: vi={vi}, vf={vf}")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, ncol=2)

    fig.tight_layout()
    pdf_path = outdir / f"lw_vf_subchannel_cross_sections_vi{vi}_vf{vf}.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {pdf_path}")


def export_ratio_outputs(state, vi, outdir):
    channels, grid, sigma_matrix, sigma_total, ratios = build_ratio_table(state, vi)
    outdir.mkdir(parents=True, exist_ok=True)

    csv_path = outdir / f"lw_channel_cross_sections_{state}_vi{vi}.csv"
    header = ["E_eV", "sigma_total"] + [f"sigma_vf{ch['vf']}" for ch in channels] \
        + [f"ratio_vf{ch['vf']}" for ch in channels]
    stacked = [grid, sigma_total]
    stacked.extend(list(sigma_matrix))
    stacked.extend(list(ratios))
    np.savetxt(
        csv_path,
        np.column_stack(stacked),
        delimiter=",",
        header=",".join(header),
        comments="",
    )

    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    ax.plot(grid, np.where(sigma_total > 0.0, sigma_total, np.nan),
            color="black", lw=2.0, label="total")
    for ch, sigma in zip(channels, sigma_matrix):
        ax.plot(grid, np.where(sigma > 0.0, sigma, np.nan), lw=1.1, label=f"vf={ch['vf']}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Incident energy $E$ [eV]")
    ax.set_ylabel(r"$\sigma(E)$")
    ax.set_title(f"Channel cross sections: {state}, vi={vi}")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, which="both", alpha=0.25)

    fig.tight_layout()
    pdf_path = outdir / f"lw_channel_cross_sections_{state}_vi{vi}.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {csv_path}")
    print(f"Saved: {pdf_path}")


def compute_k_for_state_vi(state, vi):
    """Returns k_tot(Te) array for one (state, vi) pair."""
    channels = load_channels(state, vi)
    if not channels:
        return np.zeros(N_TE)

    k = np.zeros(N_TE)
    for ch in channels:
        E_eV = ch["E_eV"]
        sigma = ch["sigma"]
        k += np.array([maxwellian_rate(E_eV, sigma, Te) for Te in TE_GRID])
    return k


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show-ratios", action="store_true",
                        help="Export per-vf channel cross sections for one (state, vi) pair")
    parser.add_argument("--show-state-channels", action="store_true",
                        help="Plot one curve per electronic state, summed over vf, for one vi")
    parser.add_argument("--show-vf-subchannels", action="store_true",
                        help="Plot one curve per electronic state for a single selected vf")
    parser.add_argument("--state", choices=[s[0] for s in STATES], default="B1Su",
                        help="Electronic state for --show-ratios")
    parser.add_argument("--vi", type=int, default=0,
                        help="Initial vibrational level for --show-ratios")
    parser.add_argument("--vf", type=int, default=1,
                        help="Final vibrational subchannel for --show-vf-subchannels")
    parser.add_argument("--ratio-outdir", type=Path, default=RATIO_OUTDIR,
                        help="Output directory for ratio CSV/PDF")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.show_vf_subchannels:
        export_vf_subchannel_plot(args.vi, args.vf, args.ratio_outdir)
        return
    if args.show_state_channels:
        export_state_channel_plot(args.vi, args.ratio_outdir)
        return
    if args.show_ratios:
        export_ratio_outputs(args.state, args.vi, args.ratio_outdir)
        return

    n_states = len(STATES)
    k_tot      = np.zeros((n_states, N_VI, N_TE))
    E_photon   = np.array([s[1] for s in STATES])
    state_names = np.array([s[0] for s in STATES])

    for si, (state, _) in enumerate(STATES):
        print(f"\nState: {state}")
        for vi in range(N_VI):
            print(f"  vi={vi} ...", end="", flush=True)
            k_tot[si, vi, :] = compute_k_for_state_vi(state, vi)
            k_peak = k_tot[si, vi, :].max()
            print(f"  peak k = {k_peak:.3e} cm3/s")

    np.savez(OUTPUT,
             Te_eV=TE_GRID,
             k_tot=k_tot,
             E_photon_eV=E_photon,
             state_names=state_names)

    print(f"\nSaved: {OUTPUT}")
    print(f"  k_tot shape: {k_tot.shape}  (states={n_states}, vi={N_VI}, Te={N_TE})")


# --- Main --------------------------------------------------------------------
if __name__ == "__main__":
    main()
