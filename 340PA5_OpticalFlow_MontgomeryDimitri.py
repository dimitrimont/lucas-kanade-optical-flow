"""
Lucas-Kanade optical flow, written by hand: how far each pixel moved between two frames.

    gradients (Ix, Iy, It)  ->  sums over a window  ->  2 x 2 solve per pixel  ->  colour and arrows

CSC 340, spring 2025. Every step of Lucas-Kanade is written out with plain loops, as the
assignment required: the gradients, the window sums, the determinant, u and v, the magnitude.
OpenCV only loads, shows and saves images and draws the arrows.

Revised 2026-10-18. Measured on the drawn pair in test_flow.py, whose true motion is known:

- The x-motion term was lost. The line that collected the gradient images unpacked Ix*It and
  then Iy*It into the same name, so the solve used Iy*It twice and the flow pointed the wrong way.
- The spatial gradient took the difference across two pixels without halving it, so every speed
  came out half its true size.
- Every window was solved, including flat or striped ones that cannot pin down a motion (the
  aperture problem), where noise turned into made-up motion. Those windows now get no estimate.
- Saved images were the raw float flow, which JPEG stores as almost black. They are now scaled
  to 0-255, and the arrow view the assignment asked for is drawn and saved too.
- Window sums come from a summed-area table, so a 21 x 21 window costs the same as a 7 x 7 one.

Usage:
    python 340PA5_OpticalFlow_MontgomeryDimitri.py first.jpg second.jpg [--window 21]
    python 340PA5_OpticalFlow_MontgomeryDimitri.py --drawn       # a drawn pair with known motion

The course's image pairs (sphere, house, tree, 2D shapes) are not in this repository.
"""

import argparse
import math
import sys

import cv2
import numpy as np

WINDOW = 7  # side of the square window each pixel's sums run over; odd
MIN_EIG = 4.0  # grey levels squared per pixel: below it the window cannot pin down a motion; raise it for noisy footage


# ── the algorithm ─────────────────────────────────────────────────────────────────────────


def gradients(img1, img2):
    """Ix and Iy on the first frame (central differences), and It (second frame minus first)."""
    rows, cols = img1.shape
    Ix = np.zeros((rows, cols), np.float64)
    Iy = np.zeros((rows, cols), np.float64)
    It = np.zeros((rows, cols), np.float64)
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            # Half the difference across the pixel, so a slope of one grey level per pixel reads 1.
            Ix[i, j] = (float(img1[i, j + 1]) - float(img1[i, j - 1])) / 2
            Iy[i, j] = (float(img1[i + 1, j]) - float(img1[i - 1, j])) / 2
            It[i, j] = float(img2[i, j]) - float(img1[i, j])
    return Ix, Iy, It


def products(Ix, Iy, It):
    """The five images the window sums need: Ix*Ix, Iy*Iy, Ix*Iy, Ix*It, Iy*It."""
    rows, cols = Ix.shape
    Ixx, Iyy, Ixy, Ixt, Iyt = (np.zeros((rows, cols), np.float64) for _ in range(5))
    for i in range(rows):
        for j in range(cols):
            Ixx[i, j] = Ix[i, j] * Ix[i, j]
            Iyy[i, j] = Iy[i, j] * Iy[i, j]
            Ixy[i, j] = Ix[i, j] * Iy[i, j]
            Ixt[i, j] = Ix[i, j] * It[i, j]
            Iyt[i, j] = Iy[i, j] * It[i, j]
    return Ixx, Iyy, Ixy, Ixt, Iyt


def window_sums(img, window):
    """The sum of img over the window centred on each pixel, from a summed-area table.

    table[i, j] holds the sum of everything above and to the left of pixel (i, j), so any window's
    sum is four lookups however big the window is. Pixels whose window would reach the image's
    outermost row or column (where there are no gradients) are left at zero.
    """
    rows, cols = img.shape
    table = np.zeros((rows + 1, cols + 1), np.float64)
    for i in range(rows):
        along = 0.0
        for j in range(cols):
            along += img[i, j]
            table[i + 1, j + 1] = table[i, j + 1] + along
    half = window // 2
    out = np.zeros((rows, cols), np.float64)
    for i in range(half + 1, rows - half - 1):
        for j in range(half + 1, cols - half - 1):
            top, bottom = i - half, i + half + 1
            left, right = j - half, j + half + 1
            out[i, j] = table[bottom, right] - table[top, right] - table[bottom, left] + table[top, left]
    return out


def flow(img1, img2, window=WINDOW, min_eig=MIN_EIG):
    """Lucas-Kanade at every pixel: u, v, and whether the window could pin the motion down.

    Inside a window, every pixel is assumed to move by the same (u, v), so Ix*u + Iy*v + It = 0
    holds at each of them. Least squares over the window gives

        [ Sxx  Sxy ] [u]     [ Sxt ]
        [ Sxy  Syy ] [v] = - [ Syt ]

    solved with the 2 x 2 inverse. A window whose matrix has a small eigenvalue has no texture in
    that direction (a flat patch, or a stripe that looks the same as it slides along itself), so
    the motion there cannot be told from noise; those pixels get no estimate.
    """
    if window % 2 == 0 or window < 3:
        raise ValueError("window must be odd and at least 3")
    Ix, Iy, It = gradients(img1, img2)
    Sxx, Syy, Sxy, Sxt, Syt = (window_sums(p, window) for p in products(Ix, Iy, It))

    rows, cols = img1.shape
    u = np.zeros((rows, cols), np.float64)
    v = np.zeros((rows, cols), np.float64)
    known = np.zeros((rows, cols), bool)
    floor = min_eig * window * window
    half = window // 2
    for i in range(half + 1, rows - half - 1):
        for j in range(half + 1, cols - half - 1):
            sxx, syy, sxy = Sxx[i, j], Syy[i, j], Sxy[i, j]
            # The smaller eigenvalue of [[sxx, sxy], [sxy, syy]]: texture in the weakest direction.
            smallest = (sxx + syy) / 2 - math.sqrt(((sxx - syy) / 2) ** 2 + sxy * sxy)
            if smallest <= floor:
                continue
            det = sxx * syy - sxy * sxy
            u[i, j] = (-syy * Sxt[i, j] + sxy * Syt[i, j]) / det
            v[i, j] = (sxy * Sxt[i, j] - sxx * Syt[i, j]) / det
            known[i, j] = True
    return u, v, known


def magnitude(u, v):
    """sqrt(u*u + v*v) at every pixel."""
    rows, cols = u.shape
    mag = np.zeros((rows, cols), np.float64)
    for i in range(rows):
        for j in range(cols):
            mag[i, j] = math.sqrt(u[i, j] ** 2 + v[i, j] ** 2)
    return mag


# ── pictures ──────────────────────────────────────────────────────────────────────────────


def near_top(values):
    """The value 98% of the way up the sorted list: full scale for a picture, so that a few
    stray pixels at a motion boundary do not dim everything else."""
    values = sorted(values)
    return values[int(0.98 * (len(values) - 1))] if values else 0.0


def colour_image(u, v, known):
    """The assignment's colouring: direction picks a mix of green and blue, speed the brightness.

    atan2 gives the direction from -pi to pi. At -pi the pixel is all green, at pi all blue, and
    in between the two are mixed in proportion. Brightness is the pixel's speed over the image's
    top speed, so still areas are black. Pixels with no estimate stay black too.
    """
    mag = magnitude(u, v)
    rows, cols = u.shape
    fastest = near_top([mag[i, j] for i in range(rows) for j in range(cols) if known[i, j]])
    out = np.zeros((rows, cols, 3), np.uint8)
    if fastest == 0:
        return out
    for i in range(rows):
        for j in range(cols):
            if not known[i, j]:
                continue
            blue = (math.atan2(v[i, j], u[i, j]) + math.pi) / (2 * math.pi)
            bright = 255 * min(1.0, mag[i, j] / fastest)
            out[i, j, 0] = round(blue * bright)
            out[i, j, 1] = round((1 - blue) * bright)
    return out


def arrow_image(img1, u, v, known, step=12):
    """The first frame with an arrow every `step` pixels, pointing where that patch moved.

    Motions this small are under a pixel, so the arrows are stretched until the fastest is `step`
    pixels long (going by the image's top speed). Returns the image and the stretch.
    """
    base = cv2.cvtColor(img1.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    mag = magnitude(u, v)
    rows, cols = u.shape
    fastest = near_top([mag[i, j] for i in range(rows) for j in range(cols) if known[i, j]])
    stretch = step / fastest if fastest else 0.0
    for i in range(step // 2, rows, step):
        for j in range(step // 2, cols, step):
            if known[i, j] and mag[i, j] > 0.05 * fastest:
                tip = (round(j + u[i, j] * stretch), round(i + v[i, j] * stretch))
                cv2.arrowedLine(base, (j, i), tip, (80, 220, 120), 1, cv2.LINE_AA, tipLength=0.3)
    return base, stretch


def signed_8bit(a, known):
    """Zero at mid grey; the image's top value either way at white or black."""
    rows, cols = a.shape
    most = near_top([abs(a[i, j]) for i in range(rows) for j in range(cols) if known[i, j]])
    out = np.full((rows, cols), 128, np.uint8)
    if most:
        for i in range(rows):
            for j in range(cols):
                out[i, j] = round(128 + 127 * max(-1.0, min(1.0, a[i, j] / most)))
    return out


def unsigned_8bit(a):
    """Zero at black, the image's top value at white."""
    rows, cols = a.shape
    most = near_top([a[i, j] for i in range(rows) for j in range(cols) if a[i, j] > 0])
    out = np.zeros((rows, cols), np.uint8)
    if most:
        for i in range(rows):
            for j in range(cols):
                out[i, j] = round(255 * min(1.0, a[i, j] / most))
    return out


# ── a pair with a known answer ────────────────────────────────────────────────────────────


def drawn_pair(dx=0.6, dy=-0.4, size=160, noise=0.0, seed=7):
    """Two frames of a drawn scene, and where its moving part is.

    A textured disc moves by (dx, dy) pixels in front of diagonal stripes that stay still, like
    the course's sphere pair. Both frames are computed from formulas rather than drawn with pixels,
    so a shift of a fraction of a pixel is exact. `noise` adds Gaussian noise of that many grey
    levels to each frame, as a camera would. Returns frame 1, frame 2 (float, 0-255) and the truth.
    """
    y, x = np.mgrid[0:size, 0:size].astype(np.float64)
    centre, radius = size / 2, size * 0.34

    def frame(ox, oy):
        bx, by = x - centre - ox, y - centre - oy
        stripes = 128 + 70 * np.sin((x + y) * 0.16)
        texture = 128 + 60 * np.sin(bx * 0.21) * np.cos(by * 0.17) + 30 * np.cos((bx - by) * 0.11)
        inside = np.clip(radius - np.hypot(bx, by) + 0.5, 0, 1)  # a one-pixel soft edge
        return stripes * (1 - inside) + texture * inside

    rng = np.random.default_rng(seed)
    one, two = frame(0, 0), frame(dx, dy)
    if noise:
        one = one + rng.normal(0, noise, one.shape)
        two = two + rng.normal(0, noise, two.shape)
    truth = {"dx": dx, "dy": dy, "centre": centre, "radius": radius}
    return np.clip(one, 0, 255), np.clip(two, 0, 255), truth


def score(u, v, known, truth, margin=WINDOW + 1):
    """How the flow compares with the truth, away from the disc's edge where windows straddle it.

    Returns the median distance between found and true motion on the disc (px), the median angle
    between them (degrees), the median speed found on the still background (px), and the share of
    each region that got an estimate.
    """
    dx, dy = truth["dx"], truth["dy"]
    rows, cols = u.shape
    off, angle, still = [], [], []
    disc = back = disc_known = back_known = 0
    for i in range(rows):
        for j in range(cols):
            r = math.hypot(j - truth["centre"], i - truth["centre"])
            if r < truth["radius"] - margin:
                disc += 1
                if known[i, j]:
                    disc_known += 1
                    off.append(math.hypot(u[i, j] - dx, v[i, j] - dy))
                    turn = math.atan2(v[i, j], u[i, j]) - math.atan2(dy, dx)
                    angle.append(abs(math.degrees(math.atan2(math.sin(turn), math.cos(turn)))))
            elif r > truth["radius"] + margin and margin < i < rows - margin and margin < j < cols - margin:
                back += 1
                if known[i, j]:
                    back_known += 1
                    still.append(math.hypot(u[i, j], v[i, j]))

    def median(xs):
        xs = sorted(xs)
        return xs[len(xs) // 2] if xs else 0.0

    return {
        "error_px": median(off),
        "angle_deg": median(angle),
        "still_px": median(still),
        "disc_covered": disc_known / disc if disc else 0.0,
        "back_covered": back_known / back if back else 0.0,
    }


# ── run ───────────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Lucas-Kanade optical flow, by hand.")
    parser.add_argument("first", nargs="?", help="the first frame")
    parser.add_argument("second", nargs="?", help="the second frame")
    parser.add_argument("--window", type=int, default=WINDOW, help=f"window side, odd (default {WINDOW})")
    parser.add_argument("--min-eig", type=float, default=MIN_EIG, help=f"texture floor (default {MIN_EIG})")
    parser.add_argument("--drawn", action="store_true", help="use a drawn pair with known motion")
    parser.add_argument("--no-menu", action="store_true", help="save the images and exit")
    args = parser.parse_args()

    if args.drawn:
        img1, img2, truth = drawn_pair(noise=2.0)
    elif args.first and args.second:
        img1 = cv2.imread(args.first, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(args.second, cv2.IMREAD_GRAYSCALE)
        if img1 is None or img2 is None or img1.shape != img2.shape:
            sys.exit(f"Could not load {args.first} and {args.second} as two images of the same size.")
        truth = None
    else:
        parser.error("give two frames, or --drawn")

    u, v, known = flow(img1, img2, args.window, args.min_eig)
    share = known.sum() / known.size
    print(f"{img1.shape[1]} x {img1.shape[0]}, {args.window} x {args.window} window: "
          f"{share:.0%} of pixels had enough texture for an estimate")
    if truth:
        s = score(u, v, known, truth)
        print(f"true motion ({truth['dx']}, {truth['dy']}) px. On the disc: off by {s['error_px']:.3f} px, "
              f"{s['angle_deg']:.1f} degrees. Still background reads {s['still_px']:.3f} px.")

    arrows, stretch = arrow_image(img1, u, v, known)
    images = {
        "flow_colour": colour_image(u, v, known),
        "flow_arrows": arrows,
        "flow_u": signed_8bit(u, known),
        "flow_v": signed_8bit(v, known),
        "flow_magnitude": unsigned_8bit(magnitude(u, v)),
    }
    for name, img in images.items():
        cv2.imwrite(f"{name}.png", img)
    print(f"Saved {', '.join(n + '.png' for n in images)}. Arrows are drawn {stretch:.0f} times longer than the motion.")
    if args.no_menu:
        return

    names = list(images)
    while True:
        print("\nOptical Flow Menu:")
        for k, name in enumerate(names, 1):
            print(f"{k}. View {name.replace('_', ' ')}")
        print(f"{len(names) + 1}. Exit")
        choice = input(f"Enter your choice (1-{len(names) + 1}): ")
        if choice == str(len(names) + 1):
            print("Exiting program...")
            break
        if choice.isdigit() and 1 <= int(choice) <= len(names):
            cv2.imshow(names[int(choice) - 1], images[names[int(choice) - 1]])
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
