# Object_Detection
**Unleash the power of computer vision in real-time**

Transform your ordinary webcam into an AI-powered surveillance system that sees and identifies objects with lightning-fast precision. This cutting-edge script harnesses the raw power of YOLO (You Only Look Once) neural networks to deliver blazing-fast object detection that processes video streams in real-time, making your machine see the world like never before.

## What Makes This Special

**Lightning-Fast Processing**: Your webcam feed becomes an intelligent eye that instantly recognizes and tracks objects as they move through the frame.

**Cinema-Quality Output**: Every detection session is automatically recorded in crystal-clear AVI format, creating a permanent record of what your AI witnessed.

**Command-Line Mastery**: Full control at your fingertips with customizable model selection and output destinations.

**Live Visual Feedback**: Watch as bounding boxes and confidence scores appear in real-time, showing you exactly what your AI is thinking.

## Core Features
- **Real-Time Detection Engine**: Process video frames at breakneck speeds using state-of-the-art YOLO architecture
- **Fully Customizable Pipeline**: Drop in any YOLO model and specify custom output paths
- **Intuitive Visual Interface**: Watch your AI work its magic with live bounding boxes and confidence metrics
- **Auto-Recording System**: Never miss a moment with automatic video capture and storage

## Arsenal Requirements
- **Python 3.x** + pip (your coding foundation)
- **YOLOv10 Model** (the AI brain that powers detection)
- **Webcam** (your digital eye to the world)

## Installation Sequence

**Step 1: Clone the Repository**
```bash
git clone https://github.com/Ashrafgalib-beep/Object_capture.git
cd Object_Detection
```

**Step 2: Install Dependencies**
```bash
pip install -r requirements.txt
```

**Step 3: Acquire YOLOv10**
Get your hands on the [YOLOv10 model](https://docs.ultralytics.com/models/yolov10/#comparisons) - the neural network that will power your detection system.

## Launch Sequence

Fire up your detection system with this command structure:
```bash
python main.py <model_path> --output <output_path>
```

**Parameters:**
- `model_path`: Your YOLO model file (`.pt` format)
- `--output`: Optional output video destination (defaults to `output.avi`)

## Example Mission

Launch a detection session with maximum firepower:
```bash
python main.py yolov10x.pt --output detected_objects.avi
```

**Emergency Stop**: Hit `Ctrl + C` in your terminal to gracefully terminate the detection session.

## License
This project operates under the Unlicense - complete freedom to use, modify, and distribute. See the [LICENSE](LICENSE) file for full details.

---

*Ready to give your machine the gift of sight? Launch your detection system and watch AI come alive!*