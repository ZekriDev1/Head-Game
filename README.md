# Face Flappy — Head to Fly

Face Flappy is a modern Flappy Bird-style game where you control the bird by moving your head! It uses **Pygame** for graphics and optionally **Mediapipe** for webcam-based head tracking. You can also play using your keyboard.

---

## 🎮 Features

- Modern UI with gradient backgrounds, soft shadows, and floating blobs.
- Smooth bird animation with wing tilt and eye graphics.
- Randomized pipes with pastel colors and subtle highlights.
- Score and best score tracking (saved in `face_flappy_best.json`).
- Head tracking jump using **Mediapipe**:
  - Move your head up to make the bird jump.
  - Calibration system for baseline head position.
- Keyboard fallback controls (`SPACE` to jump, `ESC` to quit).
- Webcam Picture-in-Picture (PiP) for visual feedback.

---

## ⚙️ Requirements

- Python 3.8+
- [Pygame](https://www.pygame.org/) (`pip install pygame`)
- [OpenCV](https://pypi.org/project/opencv-python/) (`pip install opencv-python`)
- [NumPy](https://pypi.org/project/numpy/) (`pip install numpy`)
- [Mediapipe](https://pypi.org/project/mediapipe/) (optional, for head tracking) (`pip install mediapipe`)

> Mediapipe is optional. Without it, you can play using the keyboard only.

---

## 🚀 How to Run

1. Clone the repository or download the files.
2. Install the required packages:

```bash
pip install pygame opencv-python numpy mediapipe
