# Lucas-Kanade optical flow, by hand

Coursework for CSC 340 (Computer Vision), spring 2025. Given two frames with a little motion
between them, it finds how far each pixel moved. Every step of Lucas-Kanade is written out in
plain loops, as the assignment required: the gradients, the window sums, the 2 × 2 solve and the
speed. OpenCV only loads, shows and saves the images and draws the arrows.

**Live walkthrough:** https://dimitri-montgomery-site.pages.dev/vision/optical-flow/ runs the same
algorithm in the browser on a drawn scene, with the first version beside the revised one.

## What is here

| File | What it is |
|---|---|
| `340PA5_OpticalFlow_MontgomeryDimitri.py` | The flow: gradients, window sums from a summed-area table, the solve at every pixel, then a colour image, an arrow image, and u, v and speed as greyscale images |
| `test_flow.py` | Checks the flow on drawn pairs whose true motion is known |
| `MontgomeryDimitri-Optical Flow-PA4.pdf` | The report, written on the first version |
| `assignment_opticalFlow (1).pdf` | The course handout |

## How it works

Inside a small window (7 × 7 pixels by default) every pixel is assumed to move by the same amount,
(u, v), and to keep its brightness as it moves. That gives one equation per pixel,
`Ix·u + Iy·v + It = 0`, where Ix and Iy are how fast brightness changes across and down the first
frame and It is how much it changed between the frames. Least squares over the window turns those
into a 2 × 2 system, solved with its inverse.

A window with texture in only one direction, such as a stripe or a straight edge, cannot tell how
far it slid along itself (the aperture problem), and a flat window cannot tell anything. Those
windows get no estimate: the smaller eigenvalue of the window's matrix has to clear `--min-eig`,
4 grey levels squared per pixel by default. Raise it for noisier footage.

The colour image follows the assignment: the direction from `atan2` mixes green (−π) into blue
(π), and the speed sets the brightness, so still areas are black.

### What changed on 2026-10-18

- **The x-motion term was lost.** The line that collected the gradient images unpacked `Ix·It`
  and then `Iy·It` into the same name, so the solve used `Iy·It` twice and the flow pointed the
  wrong way.
- **Every speed was half size.** The gradient took the difference across two pixels without
  halving it.
- **Windows that cannot pin down a motion now get no estimate.** Without that check, the fixed
  solve reads 0.15 px of motion on a still striped background once camera noise is added.
- **The saved images were almost black.** The float flow went straight into an 8-bit file. The
  images are now scaled to 0–255, and the arrow image the assignment asked for is drawn.
- **Any window size.** Window sums come from a summed-area table, so `--window 21` (the size the
  handout's examples used) costs the same as 7.

Measured on the drawn pair in `test_flow.py`, a textured disc moving 0.72 px up and to the right
in front of still stripes, comparing found and true motion on the disc:

| | Off from the true motion | Direction off by |
|---|---|---|
| First version | 1.02 px | 108° |
| Now | 0.05 px | 3.5° |
| First version, with camera noise | 0.93 px | 106° |
| Now, with camera noise | 0.13 px | 6.6° |

Camera noise here is Gaussian, 2 grey levels on each frame. The first version did best on straight
up-and-down motion (0.36 px, 33° off), where the lost x term matters least.

## Limits

- One scale only. Lucas-Kanade assumes the motion is small, about a pixel; larger motions need an
  image pyramid, which is not built.
- The image pairs from the course and from the report are not in this repository.

## Run

```bash
pip install opencv-python numpy pytest
python -m pytest test_flow.py                                      # no images needed
python 340PA5_OpticalFlow_MontgomeryDimitri.py --drawn             # a drawn pair with known motion
python 340PA5_OpticalFlow_MontgomeryDimitri.py first.jpg second.jpg --window 21
```

It saves `flow_colour.png`, `flow_arrows.png`, `flow_u.png`, `flow_v.png` and
`flow_magnitude.png`, then opens a menu to view them (`--no-menu` skips it).
