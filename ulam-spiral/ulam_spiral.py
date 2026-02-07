import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from sympy import isprime


@dataclass
class Vec2:
    y: int
    x: int

    def __add__(self, other):
        return type(self)(y=self.y + other.y, x=self.x + other.x)

    def __iadd__(self, other):
        self.y += other.y
        self.x += other.x
        return self

    def __hash__(self):
        return hash((self.y, self.x))


is_prime_vec = np.vectorize(isprime)


def make_spiral(*, size: int, initial: Vec2, turn: dict[Vec2, Vec2]) -> np.ndarray:
    """
    Make height x width spiral starting from center and following `initial` and `turn` instructions.
    """
    assert size % 2 == 1, "Size must be odd to construct a complete matrix."
    mat = np.zeros((size, size), dtype=int)

    pos = Vec2(size // 2, size // 2)
    delta = initial
    new_pos = None
    new_delta = None

    for idx in range(1, size**2 + 1):
        mat[pos.y, pos.x] = idx

        new_delta = initial if new_delta is None else turn[delta]
        new_pos = pos + new_delta

        # can turn
        if mat[new_pos.y, new_pos.x] == 0:
            pos = new_pos
            delta = new_delta
        # should move forward
        else:
            pos += delta

    return mat


def interpolate(f):
    assert 0 <= f <= 1
    cmap = mcolors.LinearSegmentedColormap.from_list("custom", ["#ff00ff", "black"])
    c = np.array(cmap(f)[:3])  # don't need alpha
    return (255 * c).astype(int)


def make_ulam(mat: np.ndarray) -> np.ndarray:
    n = len(mat)
    out = np.empty((*mat.shape, 3), dtype=int)

    h, w = mat.shape
    cy, cx = (h // 2, w // 2)
    y, x = np.ogrid[:h, :w]

    # dist = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    # dist = np.abs(y - cy) + np.abs(x - cx)
    # dist = np.maximum(np.abs(y - cy), np.abs(x - cx))

    # dist = dist / dist.max()
    # dist = np.power(dist, 0.2)

    # TODO: color by number of neighbours in an m*m grid

    mask = is_prime_vec(mat)
    out[mask] = [0] * 3  # [interpolate(i) for i in dist[mask]]
    out[~mask] = [255] * 3

    return out


def main():
    N, S, W, E = map(lambda t: Vec2(*t), [(-1, 0), (1, 0), (0, -1), (0, 1)])
    turn_left = {E: N, N: W, W: S, S: E}

    s = make_spiral(size=args.size, initial=E, turn=turn_left)
    u = make_ulam(s)

    plt.figure(figsize=(8,) * 2)
    plt.imshow(u)
    plt.grid(False)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(args.out_file, dpi=300, bbox_inches="tight", pad_inches=0.5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Ulam spiral image.")
    parser.add_argument(
        "--size", type=int, default=201, help="Size of the spiral (must be odd)."
    )
    parser.add_argument(
        "--out-file", type=Path, required=True, help="Path to the output png file."
    )
    args = parser.parse_args()

    main()
