import cv2
import numpy as np
import mediapipe as mp
from typing import Tuple, Optional, List
import os
from tqdm import tqdm
import argparse
import ffmpeg
import shutil
import subprocess

class FaceLandmarker:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

    def get_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Get facial landmarks from an image.
        
        Args:
            image: RGB image array
            
        Returns:
            landmarks: numpy array of shape (478, 3) if face detected, None otherwise
        """
        results = self.face_mesh.process(image)
        if not results.multi_face_landmarks:
            return None
            
        landmarks = results.multi_face_landmarks[0].landmark
        h, w = image.shape[:2]
        landmarks_array = np.array([[l.x * w, l.y * h, l.z * w] for l in landmarks])
        return landmarks_array

def get_head_bbox(landmarks: np.ndarray, expansion_factor: float = 1.5) -> Tuple[int, int, int, int]:
    """
    Get bounding box coordinates for the head region.
    
    Args:
        landmarks: numpy array of shape (478, 3) containing facial landmarks
        expansion_factor: factor to expand the bounding box
        
    Returns:
        tuple of (x1, y1, x2, y2) coordinates
    """
    min_x, min_y = landmarks[:, :2].min(axis=0)
    max_x, max_y = landmarks[:, :2].max(axis=0)
    
    # Calculate center and size
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    width = max_x - min_x
    height = max_y - min_y
    
    # Make square bounding box
    size = max(width, height) * expansion_factor
    
    # Calculate new coordinates
    x1 = int(center_x - size / 2)
    y1 = int(center_y - size / 2)
    x2 = int(center_x + size / 2)
    y2 = int(center_y + size / 2)
    
    return x1, y1, x2, y2

def crop_head(image: np.ndarray, landmarks: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Crop the head region from an image based on facial landmarks.
    
    Args:
        image: input image array
        landmarks: facial landmarks array
        
    Returns:
        tuple of (cropped image, crop size)
    """
    x1, y1, x2, y2 = get_head_bbox(landmarks)
    
    # Ensure coordinates are within image bounds
    h, w = image.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    
    # Crop
    cropped = image[y1:y2, x1:x2]
    crop_size = cropped.shape[:2]
    
    return cropped, crop_size[0]  # Return square size since we made bbox square

def draw_landmarks(image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """
    Draw landmarks on the image.
    
    Args:
        image: input image array
        landmarks: facial landmarks array
        
    Returns:
        image with landmarks drawn
    """
    img_copy = image.copy()
    for x, y, _ in landmarks:
        cv2.circle(img_copy, (int(x), int(y)), 1, (0, 255, 0), -1)
    return img_copy

def add_audio_to_video(video_path: str, audio_path: str, output_path: str):
    """
    Add audio to video using ffmpeg
    """
    temp_path = output_path + '.temp.mp4'
    shutil.move(video_path, temp_path)
    
    try:
        stream = ffmpeg.input(temp_path)
        audio = ffmpeg.input(audio_path).audio
        
        # Combine video and audio
        stream = ffmpeg.output(
            stream,
            audio,
            output_path,
            vcodec='libx264',
            acodec='aac',
            **{'b:a': '192k'}
        )
        
        # Overwrite output and hide ffmpeg output
        stream = stream.overwrite_output()
        ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def process_video(input_video: str, 
                 output_video: str, 
                 draw_points: bool = True) -> None:
    """
    Process a video file: detect landmarks, crop faces,
    and create two output videos - one with landmarks and one without.
    """
    # Initialize face landmarker
    face_landmarker = FaceLandmarker()
    
    # Open video
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # First pass: determine crop size
    crop_sizes = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks = face_landmarker.get_landmarks(frame_rgb)
        
        if landmarks is not None:
            _, crop_size = crop_head(frame_rgb, landmarks)
            crop_sizes.append(crop_size)
    
    # Use median crop size for consistency
    final_crop_size = int(np.median(crop_sizes)) if crop_sizes else 256
    
    # Reset video capture
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Create temporary output paths
    temp_crop = output_video.replace('.mp4', '_temp.mp4')
    temp_landmarks = output_video.replace('.mp4', '_landmarks_temp.mp4')
    landmark_output = output_video.replace('.mp4', '_landmarks.mp4')
    
    # Create video writers
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_crop = cv2.VideoWriter(temp_crop, fourcc, fps, (final_crop_size, final_crop_size))
    out_landmarks = cv2.VideoWriter(temp_landmarks, fourcc, fps, (final_crop_size, final_crop_size))
    
    try:
        # Process frames
        with tqdm(total=frame_count, desc="Processing frames") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                landmarks = face_landmarker.get_landmarks(frame_rgb)
                
                if landmarks is not None:
                    # Crop head region
                    cropped_frame, _ = crop_head(frame_rgb, landmarks)
                    # Resize to consistent size
                    processed_frame = cv2.resize(cropped_frame, (final_crop_size, final_crop_size))
                    
                    # Save cropped frame
                    crop_bgr = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
                    out_crop.write(crop_bgr)
                    
                    # Always create landmark frame
                    landmarks_resized = face_landmarker.get_landmarks(processed_frame)
                    if landmarks_resized is not None:
                        landmark_frame = draw_landmarks(processed_frame, landmarks_resized)
                        landmark_frame = cv2.cvtColor(landmark_frame, cv2.COLOR_RGB2BGR)
                        out_landmarks.write(landmark_frame)
                    else:
                        # If landmarks not detected in resized frame, use original landmarks
                        landmark_frame = draw_landmarks(processed_frame, landmarks)
                        landmark_frame = cv2.cvtColor(landmark_frame, cv2.COLOR_RGB2BGR)
                        out_landmarks.write(landmark_frame)
                else:
                    print(f"No face detected in frame")
                    # Write black frame if no face detected
                    black_frame = np.zeros((final_crop_size, final_crop_size, 3), dtype=np.uint8)
                    out_crop.write(black_frame)
                    out_landmarks.write(black_frame)
                
                pbar.update(1)
    
    finally:
        cap.release()
        out_crop.release()
        out_landmarks.release()
    
    # Add audio using ffmpeg
    print("Adding audio to output videos...")
    try:
        # For cropped video
        cmd = [
            'ffmpeg', '-i', temp_crop,
            '-i', input_video,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-map', '0:v:0',
            '-map', '1:a:0',
            '-y',
            output_video
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        
        # For landmark video
        cmd = [
            'ffmpeg', '-i', temp_landmarks,
            '-i', input_video,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-map', '0:v:0',
            '-map', '1:a:0',
            '-y',
            landmark_output
        ]
        subprocess.run(cmd, check=True, capture_output=True)
    except Exception as e:
        print(f"Warning: Could not add audio to output videos: {str(e)}")
        # If audio addition fails, at least save the video without audio
        if os.path.exists(temp_crop):
            shutil.move(temp_crop, output_video)
        if os.path.exists(temp_landmarks):
            shutil.move(temp_landmarks, landmark_output)
    finally:
        # Clean up temporary files
        for temp_file in [temp_crop, temp_landmarks]:
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    print(f"Created cropped video: {output_video}")
    print(f"Created landmark video: {landmark_output}")

def main():
    """
    Process a video to generate landmark video
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_video', type=str, required=True, help='Path to input video')
    parser.add_argument('--output_video', type=str, required=True, help='Path to save processed video')
    parser.add_argument('--draw_landmarks', action='store_true', help='Draw landmarks on faces')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_video), exist_ok=True)
    
    print(f"Processing video...")
    try:
        process_video(
            input_video=args.input_video,
            output_video=args.output_video,
            draw_points=args.draw_landmarks
        )
        print(f"Successfully processed video")
    except Exception as e:
        print(f"Error processing video: {str(e)}")

if __name__ == "__main__":
    main()
