"""
Checks the optical flow against pairs of frames whose true motion is known.

The pairs are drawn rather than loaded: a textured disc slides a fraction of a pixel in front of
diagonal stripes that stay still, like the course's sphere pair. So the tests need no image files,
and they can say exactly how far the flow is from the truth.

    python -m pytest test_flow.py       or       python test_flow.py
"""

import importlib

import numpy as np

lk = importlib.import_module("340PA5_OpticalFlow_MontgomeryDimitri")


def test_finds_the_true_motion():
    img1, img2, truth = lk.drawn_pair(0.6, -0.4)
    s = lk.score(*lk.flow(img1, img2), truth)
    assert s["error_px"] < 0.1
    assert s["angle_deg"] < 6
    assert s["disc_covered"] > 0.8


def test_every_direction():
    """Sideways, up, down and diagonal. The first version lost the x term and failed sideways."""
    for dx, dy in [(-0.5, 0.0), (0.5, 0.0), (0.0, 0.5), (0.0, -0.5), (0.35, 0.35), (-0.4, 0.3)]:
        img1, img2, truth = lk.drawn_pair(dx, dy, size=120)
        s = lk.score(*lk.flow(img1, img2), truth)
        assert s["error_px"] < 0.1, (dx, dy, s)


def test_speed_is_full_size():
    """The first version did not halve the gradient, so every speed came out half size."""
    img1, img2, _ = lk.drawn_pair(0.5, 0.0, size=120)
    u, v, known = lk.flow(img1, img2)
    assert abs(np.median(u[known]) - 0.5) < 0.05


def test_stripes_and_flat_areas_get_no_estimate():
    """A still striped background under camera noise has nothing to pin a motion to."""
    img1, img2, truth = lk.drawn_pair(0.6, -0.4, noise=2.0)
    s = lk.score(*lk.flow(img1, img2), truth)
    assert s["back_covered"] < 0.05
    assert s["disc_covered"] > 0.9
    flat = np.full((40, 40), 120.0)
    _, _, known = lk.flow(flat, flat + np.random.default_rng(1).normal(0, 1, flat.shape))
    assert not known.any()


def test_saved_images_use_the_full_range():
    """The first version saved the float flow, which an 8-bit file stores as nearly black."""
    img1, img2, _ = lk.drawn_pair(0.6, -0.4, size=100)
    u, v, known = lk.flow(img1, img2)
    colour = lk.colour_image(u, v, known).astype(int)
    assert (colour[..., 0] + colour[..., 1]).max() >= 254  # blue and green share the brightness
    assert lk.unsigned_8bit(lk.magnitude(u, v)).max() == 255
    signed = lk.signed_8bit(u, known)
    assert signed.min() < 60 or signed.max() > 196


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"ok  {name}")
