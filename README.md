# Lucas-Kanade Optical Flow (From Scratch)

This project implements the Lucas-Kanade optical flow algorithm in Python without using OpenCV's built-in flow functions. It visualizes motion between two image frames using both vector field and color-based techniques.

## Features
- Manual gradient computation (Ix, Iy, It)
- Local structure matrix and flow vector calculation (u, v)
- Color-coded visualization based on direction and magnitude
- Arrow-line visualization (quiver plot)
- Adjustable neighborhood size and image pairs

## Tools Used
- Python
- NumPy
- OpenCV (for image loading and displaying only)
- Matplotlib (optional)

## Example Output
" (examples/flow_color_mapping.png) "

## Run
```bash
python optical_flow.py
