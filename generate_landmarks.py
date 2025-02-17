import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import imageio_ffmpeg as ffmpeg
import subprocess

class FacialLandmarkVideo:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.face_spec = self.mp_drawing.DrawingSpec(
            color=(0, 255, 0),
            thickness=1,
            circle_radius=1
        )
        self.lips_spec = self.mp_drawing.DrawingSpec(
            color=(0, 0, 255),
            thickness=2,
            circle_radius=1
        )
        self.contour_spec = self.mp_drawing.DrawingSpec(
            color=(255, 255, 255),
            thickness=1,
            circle_radius=1
        )

    def process_video(self, input_path, output_path):
        # Create temporary and final output paths
        temp_path = Path(output_path).parent / f"{Path(output_path).stem}_temp.mp4"
        final_path = Path(output_path).parent / f"{Path(output_path).stem}.mp4"
        
        # Convert to strings for OpenCV
        temp_output = str(temp_path)
        final_output = str(final_path)
        
        # Open input video
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open input video: {input_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\nProcessing video:")
        print(f"Resolution: {width}x{height}")
        print(f"FPS: {fps}")
        print(f"Total frames: {total_frames}")
        
        # Create output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
        
        if not out.isOpened():
            raise RuntimeError("Failed to create video writer")
        
        try:
            # Process frames
            with tqdm(total=total_frames, desc="Processing frames") as pbar:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = self.face_mesh.process(frame_rgb)
                    output_frame = frame.copy()
                    
                    if results.multi_face_landmarks:
                        for face_landmarks in results.multi_face_landmarks:
                            # Draw face mesh
                            self.mp_drawing.draw_landmarks(
                                image=output_frame,
                                landmark_list=face_landmarks,
                                connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                                landmark_drawing_spec=self.face_spec,
                                connection_drawing_spec=self.face_spec
                            )
                            
                            # Draw contours
                            self.mp_drawing.draw_landmarks(
                                image=output_frame,
                                landmark_list=face_landmarks,
                                connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=self.contour_spec
                            )
                            
                            # Draw lips
                            self.mp_drawing.draw_landmarks(
                                image=output_frame,
                                landmark_list=face_landmarks,
                                connections=self.mp_face_mesh.FACEMESH_LIPS,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=self.lips_spec
                            )
                    
                    out.write(output_frame)
                    pbar.update(1)
                
        finally:
            # Release resources
            cap.release()
            out.release()
            cv2.destroyAllWindows()
        
        # Add audio using ffmpeg
        try:
            print("\nAdding audio to video...")
            # Get ffmpeg path
            ffmpeg_path = ffmpeg.get_ffmpeg_exe()
            
            # Construct ffmpeg command
            cmd = [
                ffmpeg_path,
                '-i', temp_output,  # Video input
                '-i', str(input_path),  # Audio input
                '-c:v', 'copy',  # Copy video stream
                '-c:a', 'aac',  # AAC audio codec
                '-map', '0:v:0',  # Use video from first input
                '-map', '1:a:0',  # Use audio from second input
                '-y',  # Overwrite output file
                final_output
            ]
            
            # Run ffmpeg command
            subprocess.run(cmd, check=True, capture_output=True)
            
            # Remove temporary file
            if temp_path.exists():
                temp_path.unlink()
            
            print(f"\nSuccessfully created video with audio: {final_output}")
            
        except Exception as e:
            print(f"\nError adding audio: {str(e)}")
            # If audio processing fails, keep the video without audio
            if temp_path.exists():
                temp_path.rename(final_path)
            print(f"Saved video without audio: {final_output}")

def main():
    parser = argparse.ArgumentParser(description='Create facial landmark video with audio')
    parser.add_argument('input', help='Input video path')
    parser.add_argument('-o', '--output', help='Output video path (without extension)')
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")
    
    if args.output:
        output_path = args.output
    else:
        output_path = str(input_path.parent / f"{input_path.stem}_landmarks")
    
    # Create output directory
    Path(output_path).parent.mkdir(exist_ok=True)
    
    try:
        processor = FacialLandmarkVideo()
        processor.process_video(input_path, output_path)
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

if __name__ == "__main__":
    main()