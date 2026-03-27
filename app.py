import sys
import os

# Ensure the src package is importable when running app.py from root
root = os.path.dirname(os.path.abspath(__file__))
if root not in sys.path:
    sys.path.insert(0, root)
if os.path.join(root, 'src') not in sys.path:
    sys.path.insert(0, os.path.join(root, 'src'))

from video_processor import process_video, process_webcam


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--webcam":
        process_webcam()
    else:
        input_file = sys.argv[1] if len(sys.argv) > 1 else "input.mp4"
        output_file = sys.argv[2] if len(sys.argv) > 2 else "output.mp4"
        process_video(input_file, output_file)


if __name__ == "__main__":
    main()
