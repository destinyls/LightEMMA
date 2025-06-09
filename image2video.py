#!/usr/bin/env python3
import cv2
import os
import argparse
import glob
from natsort import natsorted

def images_to_video(image_folder, output_video, fps=30, img_formats=None):
    """
    Convert image sequence in folder to video
    
    Parameters:
        image_folder: Path to folder containing image sequence (will recursively search subdirectories)
        output_video: Output video file path
        fps: Frame rate
        img_formats: List of image formats, e.g. ['*.jpg', '*.png']
    """
    if img_formats is None:
        img_formats = ['*.jpg', '*.png']
    
    # Get all supported format images and merge (recursively search subdirectories)
    images = []
    for fmt in img_formats:
        # Use recursive mode to search all subdirectories
        format_images = glob.glob(os.path.join(image_folder, '**', fmt), recursive=True)
        images.extend(format_images)
    
    # Sort all images in natural order
    images = natsorted(images)
    
    if not images:
        print(f"No supported format images found in folder {image_folder} and its subdirectories")
        return
    
    print(f"Found {len(images)} images distributed in the following directories:")
    # Display directories containing images
    directories = set(os.path.dirname(img) for img in images)
    for directory in sorted(directories):
        count = sum(1 for img in images if os.path.dirname(img) == directory)
        print(f"  {directory}: {count} images")
    
    # Read first image to get image dimensions
    frame = cv2.imread(images[0])
    height, width, channels = frame.shape
    
    # Define video codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use MP4V codec
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    # Iterate through all images and add to video
    total_images = len(images)
    for i, image_path in enumerate(images):
        img = cv2.imread(image_path)
        if img is None:
            print(f"Unable to read image: {image_path}")
            continue
        
        video_writer.write(img)
        print(f"Processing... {i+1}/{total_images}: {os.path.basename(image_path)}", end='\r')
    
    # Release video writer
    video_writer.release()
    print(f"\nVideo successfully generated: {output_video}")
    print(f"Total processed {total_images} images")

def main():
    parser = argparse.ArgumentParser(description='Convert image sequence to video (supports recursive subdirectory search)')
    parser.add_argument('-i', '--input', required=True, help='Image folder path (will recursively search subdirectories)')
    parser.add_argument('-o', '--output', required=True, help='Output video file path')
    parser.add_argument('-f', '--fps', type=int, default=4, help='Frame rate (default: 12)')
    parser.add_argument('--formats', nargs='+', default=['*.jpg', '*.png'], 
                        help='Image format list (default: *.jpg, can specify multiple formats like: --formats *.jpg *.png)')
    
    args = parser.parse_args()
    
    images_to_video(args.input, args.output, args.fps, args.formats)

if __name__ == "__main__":
    main()
