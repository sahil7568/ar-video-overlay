# 🔍 Augmented Reality: Video Overlay on Target Image 🎥

This project uses **OpenCV** and **Python** to create an Augmented Reality experience where a target image (like a card or poster) is detected through the webcam, and a video is projected on top of it in real time.

---

## 🚀 Features
- Detects custom target image using ORB feature matching
- Plays video only when the image is visible
- Real-time video warping with homography
- Works with any webcam
- Easy to customize with your own image & video

---

## 🧠 Tech Stack
- Python 3
- OpenCV
- NumPy

---

## 🖼️ How It Works
1. Load your target image (e.g., `targetimage.jpg`)
2. Load the video you want to overlay
3. When the webcam detects the image, it warps and overlays the video perfectly on it
4. If image is removed → video stops

---

## 📂 Project Structure
