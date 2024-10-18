import os
import sys
import time
import yaml
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
from ultralytics import YOLO
from ultralytics.utils.torch_utils import select_device
from tkinter import Tk, filedialog, simpledialog
from concurrent.futures import ThreadPoolExecutor

# Function to open a file explorer to select the video directory
def select_video_directory():
    root = Tk()
    root.withdraw()  # Hide the root window
    video_dir = filedialog.askdirectory(title="Select Video Directory")
    return video_dir

def sku_name():
    root = Tk()
    root.withdraw()  # Hide the root window
    user_input = ""

    user_input = simpledialog.askstring("Input", "Enter SKU Name:")
    return user_input

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
print(ROOT)
os.chdir(ROOT)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Function to select ROI in the first frame of the video
def select_roi(video_path, name):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()

    if not ret:
        print(f"Error: Could not read frame from {video_path}")
        return None
    
    frame = cv2.resize(frame, (1920,1080), interpolation = cv2.INTER_AREA)


    roi = cv2.selectROI(f"Select ROI {name}", frame)
    cv2.destroyWindow(f"Select ROI {name}")

    # Convert roi from (x, y, width, height) to (xmin, ymin, xmax, ymax)
    roi = (int(roi[0]), int(roi[1]), int(roi[0] + roi[2]), int(roi[1] + roi[3]))

    cap.release()
    return roi

def process_video(video_file, input_video_dir):
    # Extract the main directory and subdirectory names
    main_dir, sub_dir, sub_ch = video_file.split('_', 2)
    ai = sub_ch.split('_', 1)
    # model_name = f'best_{ai[0]}_v1.pt'
    model_name = f'43SKU.pt'
    output_dir = os.path.join(main_dir, sub_dir, sub_ch)
    
    os.makedirs(output_dir, exist_ok=True)
    video_path = os.path.join(input_video_dir, video_file)

    output_label = os.path.join(main_dir, sub_dir, sub_ch)
    Path(output_label).mkdir(parents=True, exist_ok=True)

    yaml_file = ROOT / main_dir / f"{main_dir}_SKU.yaml"

    # Load YAML content from the file
    with open(yaml_file, 'r') as file:
        data = yaml.safe_load(file)

    # Get the index of a specific value
    value_to_find = sub_dir

    try:
        index = data["names"].index(value_to_find)
    except:
        data["names"].append(value_to_find)
        with open(yaml_file, 'w') as file:
            yaml.safe_dump(data, file)
        index = data["names"].index(value_to_find)

    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    SKIP_FRAME = total_frames // num_skip
    SKIP_FRAME = max(1, min(SKIP_FRAME, 200))

    name = sub_dir
    model_path = r'D:\cpf\model\weights' + f'\{model_name}'

    frame_count = 0
    img_count = 0
    i = 0
    error_count = 0

    # Select ROI in the first frame
    roi = select_roi(video_path, sub_ch)
    if not roi:
        print("Error: ROI selection failed.")
        return
    
    buffered_frames = []

    time.sleep(30)


    # Load the trained YOLOv5 model
    model = YOLO(f'{model_path}')
    device = select_device(f'cuda:0')
    model.to(device)
    time.sleep(1)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_file}")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            error_count += 1
            if error_count >= 100:
                break
            continue

        if frame_count % SKIP_FRAME == 0:
            frame = cv2.resize(frame, (1920,1080), interpolation = cv2.INTER_AREA)
            buffered_frames.append(frame)
        
        frame_count += 1

        if len(buffered_frames) >= BUFFER_SIZE:
            for buffered_frame in buffered_frames:
                filePath = os.path.join(output_dir, f'{sub_dir}_{sub_ch}_{i:04d}.jpg')
                cv2.imwrite(filePath, buffered_frame)
                
                if save_txt:
                    image = Image.open(filePath)
                    results = model.predict(image, agnostic_nms=True)
                    frame = np.array(results[0].plot(labels=True, conf=False, line_width=2))
                    detected_objects = results[0].boxes.xyxy.cpu().numpy()
                    class_id = results[0].boxes.cls.cpu().tolist()
                    # cv2.imshow(f"auto label {sub_ch}", frame)
                    # cv2.waitKey(1)

                    filtered_objects = []
                    for obj, cls in zip(detected_objects, class_id):
                        x_center = (obj[0] + obj[2]) / 2
                        y_center = (obj[1] + obj[3]) / 2
                        if roi[0] <= x_center <= roi[2] and roi[1] <= y_center <= roi[3]:
                            filtered_objects.append((obj, cls))

                    if filtered_objects:
                        label_file = os.path.join(output_label, f'{sub_dir}_{sub_ch}_{i:04d}.txt')
                        with open(label_file, 'w') as files:
                            for obj, cls in filtered_objects:
                                x_center = (obj[0] + obj[2]) / 2
                                y_center = (obj[1] + obj[3]) / 2
                                width = obj[2] - obj[0]
                                height = obj[3] - obj[1]
                                image_width, image_height = image.size
                                x_center_norm, y_center_norm, width_norm, height_norm = (
                                    x_center / image_width, y_center / image_height,
                                    width / image_width, height / image_height
                                )
                                files.write(f"{index} {x_center_norm} {y_center_norm} {width_norm} {height_norm}\n")

                        img_count += 1
                        print(f"write frame {i} from {video_file} _____ image Count :{img_count} _____ numpic:{i}")
                    else:
                        os.remove(filePath)

                else:
                    img_count += 1

                i += 1
                
            buffered_frames = []

        if img_count >= num_skip - 100:
            break

    cap.release()
    print(f"Captured {frame_count} frames from {video_file}")

# Select the video directory using a file explorer
print("Select The Video Directory")
input_video_dir = select_video_directory()
if not input_video_dir:
    print("No directory selected.")
    exit()

# Get a list of all files in the video directory
video_files = [f for f in os.listdir(input_video_dir) if f.endswith(('.mp4', '.avi', '.mkv'))]
if not video_files:
    print("No video files found in the directory.")
    exit()

BUFFER_SIZE = 20
num_skip = 300#2500
save_txt = True

# Process each video file in a separate thread
with ThreadPoolExecutor() as executor:
    futures = [executor.submit(process_video, video_file, input_video_dir) for video_file in video_files]
    for future in futures:
        future.result()  # Wait for all threads to complete

print("All videos have been processed.")
