"""Plot the final energy spectrum for the Re_lambda ~= 100, T=10 HIT case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def energy_spectrum(velocity: np.ndarray, length: float) -> tuple[np.ndarray, np.ndarray]:
    """Return shell-integrated E(k) from a [B, 3, N, N, N] velocity field."""
    n = velocity.shape[-1]
    uh = np.fft.rfftn(velocity, axes=(-3, -2, -1), norm="forward")
    kx = 2.0 * np.pi * np.fft.fftfreq(n, d=length / n)
    kz = 2.0 * np.pi * np.fft.rfftfreq(n, d=length / n)
    k2 = (
        kx[:, None, None] ** 2
        + kx[None, :, None] ** 2
        + kz[None, None, :] ** 2
    )
    weights = np.ones_like(k2)
    weights[..., 1:-1] = 2.0
    modal_energy = 0.5 * np.sum(np.abs(uh) ** 2, axis=(0, 1)) * weights
    shell = np.rint(np.sqrt(k2)).astype(np.int64)
    ek = np.bincount(shell.ravel(), weights=modal_energy.ravel())
    ek[0] = 0.0
    return np.arange(ek.size, dtype=float), ek


def analyze(snapshot: Path, output_dir: Path) -> dict[str, float | str]:
    with np.load(snapshot) as data:
        velocity = np.asarray(data["UU"])
        time = float(np.asarray(data["tt"]))
        nu = float(np.asarray(data["nu"]))
        x = np.asarray(data["xx"])

    length = float((x[1] - x[0]) * x.size)
    k, ek = energy_spectrum(velocity, length)
    valid_k = (k > 0) & (k <= np.sqrt(2.0) * velocity.shape[-1] / 3.0)
    epsilon = float(2.0 * nu * np.sum(k[valid_k] ** 2 * ek[valid_k]))
    energy = float(np.sum(ek[valid_k]))
    u_rms = float(np.sqrt(2.0 * energy / 3.0))
    taylor_scale = float(np.sqrt(15.0 * nu * u_rms**2 / epsilon))
    re_taylor = float(u_rms * taylor_scale / nu)
    eta = float((nu**3 / epsilon) ** 0.25)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "energy_spectrum_data.npz",
        time=time,
        nu=nu,
        k=k[valid_k],
        Ek=ek[valid_k],
        epsilon=epsilon,
        eta=eta,
        ReTaylor=re_taylor,
    )

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "figure.dpi": 140,
        }
    )
    fig, ax = plt.subplots(figsize=(6.4, 4.8))

    ax.loglog(k[valid_k], ek[valid_k], "o-", ms=4, lw=1.4, color="#2166ac")
    inertial = valid_k & (k >= 4) & (k <= 20)
    if np.any(inertial):
        anchor = np.median(ek[inertial] * k[inertial] ** (5.0 / 3.0))
        ax.loglog(
            k[inertial],
            anchor * k[inertial] ** (-5.0 / 3.0),
            "k--",
            lw=1,
            label=r"$k^{-5/3}$",
        )
    ax.set(
        xlabel=r"$k$",
        ylabel=r"$E(k)$",
        title=rf"Energy spectrum at $t={time:g}$ ($Re_\lambda={re_taylor:.1f}$, $N={velocity.shape[-1]}^3$)",
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    figure_path = output_dir / "final_observation.png"
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)

    summary: dict[str, float | str] = {
        "snapshot": str(snapshot),
        "time": time,
        "N": int(velocity.shape[-1]),
        "nu": nu,
        "energy": energy,
        "epsilon": epsilon,
        "u_rms": u_rms,
        "taylor_scale": taylor_scale,
        "ReTaylor": re_taylor,
        "eta": eta,
        "kmax_eta": float(np.sqrt(2.0) * velocity.shape[-1] / 3.0 * eta),
    }
    (output_dir / "final_observation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "snapshot",
        nargs="?",
        type=Path,
        default=Path("../data/N128_nu2.000n_T10.0/YZXDNS-00010000.npz"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/ReT100_T10"))
    args = parser.parse_args()
    print(json.dumps(analyze(args.snapshot, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
