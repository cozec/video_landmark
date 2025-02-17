# Video Landmark Processing

This project provides tools for processing videos to detect and work with facial landmarks.

## Features

- Facial landmark detection from video files
- Landmark-based face cropping
- Support for video processing and frame extraction

## Setup

1. Install the required dependencies:

## Usage

### Generate Landmarks

Use `generate_landmarks.py` to detect and extract facial landmarks from a video:

### Crop Video Using Landmarks

Use `landmark_crop.py` to crop video frames based on detected facial landmarks:

## Project Structure

- `generate_landmarks.py`: Script for detecting and saving facial landmarks from videos
- `landmark_crop.py`: Tool for cropping videos based on landmark positions

## Requirements

- Python 3.7+
- OpenCV
- MediaPipe (for facial landmark detection)
- NumPy

## License

This project is licensed under the MIT License - see the LICENSE file for details.