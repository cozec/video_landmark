import subprocess
from pathlib import Path
import logging
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def cut_video(input_path, output_path, duration=10):
    """Cut video to specified duration using ffmpeg"""
    try:
        command = [
            'ffmpeg',
            '-i', str(input_path),  # Input file
            '-ss', '0',             # Start time
            '-t', str(duration),    # Duration
            '-c:v', 'libx264',      # Video codec
            '-c:a', 'aac',          # Audio codec
            '-y',                   # Overwrite output file
            str(output_path)
        ]
        
        # Run ffmpeg command
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for the process to complete
        stdout, stderr = process.communicate()
        
        if process.returncode == 0:
            logger.info(f"Successfully cut video: {input_path.name}")
            return True
        else:
            logger.error(f"Error cutting video {input_path.name}: {stderr.decode()}")
            return False
            
    except Exception as e:
        logger.error(f"Error processing {input_path.name}: {str(e)}")
        return False

def main():
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description='Cut videos to specified duration')
    parser.add_argument('--input_dir', type=str, default='talking_face_videos',
                        help='Input directory containing videos')
    parser.add_argument('--output_dir', type=str, default='talking_face_videos_cut',
                        help='Output directory for cut videos')
    parser.add_argument('--duration', type=int, default=10,
                        help='Duration in seconds to cut videos to')
    args = parser.parse_args()
    
    # Create output directory
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Get all video files
    video_files = list(input_dir.glob('*.mp4')) + \
                 list(input_dir.glob('*.avi')) + \
                 list(input_dir.glob('*.mpg')) + \
                 list(input_dir.glob('*.mov'))
    
    if not video_files:
        logger.error(f"No video files found in {input_dir}")
        return
    
    # Process videos
    success_count = 0
    for video_path in tqdm(video_files, desc="Cutting videos"):
        output_path = output_dir / f"{video_path.stem}_cut{video_path.suffix}"
        if cut_video(video_path, output_path, args.duration):
            success_count += 1
    
    logger.info(f"\nProcessed {success_count}/{len(video_files)} videos")
    logger.info(f"Cut videos saved to: {output_dir}")

if __name__ == "__main__":
    main() 