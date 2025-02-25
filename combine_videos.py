import cv2
import numpy as np
from pathlib import Path
import argparse
import logging
from tqdm import tqdm
import mediapipe as mp

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_face_bounds(landmarks, frame_shape, padding=0.2):
    """Get face bounding box with padding"""
    if not landmarks:
        return None
    
    h, w = frame_shape[:2]
    x_coords = [landmark.x for landmark in landmarks.landmark]
    y_coords = [landmark.y for landmark in landmarks.landmark]
    
    # Convert relative coordinates to absolute
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    # Add padding
    width = x_max - x_min
    height = y_max - y_min
    x_min = max(0, x_min - width * padding)
    x_max = min(1, x_max + width * padding)
    y_min = max(0, y_min - height * padding)
    y_max = min(1, y_max + height * padding)
    
    # Convert to pixel coordinates
    x_min, x_max = int(x_min * w), int(x_max * w)
    y_min, y_max = int(y_min * h), int(y_max * h)
    
    return (x_min, y_min, x_max, y_max)

def process_frame(frame, face_mesh):
    """Process frame to detect landmarks and crop to face"""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    if not results.multi_face_landmarks:
        return frame, None  # Return original frame if no face detected
    
    face_landmarks = results.multi_face_landmarks[0]  # Use first face
    bounds = get_face_bounds(face_landmarks, frame.shape)
    
    if bounds is None:
        return frame, None
    
    x_min, y_min, x_max, y_max = bounds
    cropped_frame = frame[y_min:y_max, x_min:x_max]
    
    # Draw landmarks on cropped frame
    for landmark in face_landmarks.landmark:
        # Adjust landmark positions for cropped frame
        rel_x = (landmark.x * frame.shape[1] - x_min) / (x_max - x_min)
        rel_y = (landmark.y * frame.shape[0] - y_min) / (y_max - y_min)
        
        if 0 <= rel_x <= 1 and 0 <= rel_y <= 1:
            pos = (
                int(rel_x * cropped_frame.shape[1]),
                int(rel_y * cropped_frame.shape[0])
            )
            cv2.circle(cropped_frame, pos, 1, (0, 255, 0), -1)
    
    return cropped_frame, bounds

def combine_videos(video1_path, video2_path, output_path):
    """Combine two videos side by side horizontally with facial landmarks"""
    try:
        # Initialize MediaPipe Face Mesh
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Open both videos
        cap1 = cv2.VideoCapture(str(video1_path))
        cap2 = cv2.VideoCapture(str(video2_path))
        
        # Get total frame count for progress bar
        total_frames = min(
            int(cap1.get(cv2.CAP_PROP_FRAME_COUNT)),
            int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
        )
        fps = min(cap1.get(cv2.CAP_PROP_FPS), cap2.get(cv2.CAP_PROP_FPS))
        
        # We'll determine output size after processing first frames
        first_frames = []
        for cap in [cap1, cap2]:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read first frame")
                return False
            processed_frame, _ = process_frame(frame, face_mesh)
            first_frames.append(processed_frame)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to start
        
        # Calculate output dimensions
        target_height = max(frame.shape[0] for frame in first_frames)
        aspect_ratios = [frame.shape[1] / frame.shape[0] for frame in first_frames]
        new_widths = [int(target_height * ar) for ar in aspect_ratios]
        combined_width = sum(new_widths)
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (combined_width, target_height)
        )
        
        # Process frames
        with tqdm(total=total_frames, desc="Combining videos") as pbar:
            while True:
                ret1, frame1 = cap1.read()
                ret2, frame2 = cap2.read()
                
                if not ret1 or not ret2:
                    break
                
                # Process both frames
                frame1, _ = process_frame(frame1, face_mesh)
                frame2, _ = process_frame(frame2, face_mesh)
                
                # Resize frames to target height
                frame1 = cv2.resize(frame1, (new_widths[0], target_height))
                frame2 = cv2.resize(frame2, (new_widths[1], target_height))
                
                # Combine frames horizontally
                combined_frame = np.hstack((frame1, frame2))
                
                # Write the combined frame
                out.write(combined_frame)
                pbar.update(1)
        
        # Release everything
        face_mesh.close()
        cap1.release()
        cap2.release()
        out.release()
        
        logger.info(f"Successfully combined videos to: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error combining videos: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Combine two videos side by side')
    parser.add_argument('--video1', type=str, required=True,
                      help='Path to first video')
    parser.add_argument('--video2', type=str, required=True,
                      help='Path to second video')
    parser.add_argument('--output', type=str, default='combined_video.mp4',
                      help='Output path for combined video')
    
    args = parser.parse_args()
    
    # Convert paths to Path objects
    video1_path = Path(args.video1)
    video2_path = Path(args.video2)
    output_path = Path(args.output)
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Combine videos
    if combine_videos(video1_path, video2_path, output_path):
        logger.info("Video combination completed successfully")
    else:
        logger.error("Failed to combine videos")

if __name__ == "__main__":
    main() 