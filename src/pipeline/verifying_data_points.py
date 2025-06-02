__author__ = 'Win Aung'
__modified_by__ = 'Pedro Oliveira'
__credits__ = ['Win Aung', 'Pedro Oliveira']

# Import necessary libraries
import os                                                                                                                                                                                        #
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2


def load_data(path):
    X_data = []
    data_frames_list = []
    file_config = ("d__landmarks.csv", "p__landmarks.csv", "ft__landmarks.csv", "ht__landmarks.csv", "alt__landmarks.csv")

    try:
        for folder in os.listdir(path):  # Navigate each subfolder
            folder_path = os.path.join(path, folder)

            if not os.path.isdir(folder_path): continue  # Skip if it's not a folder

            for file in os.listdir(folder_path):  # Navigate each folder per subfolder
                file_path = os.path.join(folder_path, file)
                if (file.endswith(file_config)):
                    pose_df = pd.read_csv(file_path)

                    if pose_df.shape != (50, 99 + 1):
                        continue  # +1 for 'Unnamed: 0'

                    frame_data = pose_df.drop(columns=pose_df.columns[0]).values.astype(np.float32)  # Shape: (50, 99)
                    data_frames_list.append(frame_data)
                    X_data.append(frame_data)
    except Exception as e:
        print(f"Error: Unknown error occurred: {e}")

    return X_data


def create_frame_images(frame_data_list, pose, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame(frame_data_list[pose])

    for frame_index in range(len(df)):
        # Extract x and y coordinates for all landmarks
        x_coords = df.iloc[frame_index, ::3]  # Every 3rd value starting from index 0 (x coordinates)
        y_coords = df.iloc[frame_index, 1::3]  # Every 3rd value starting from index 1 (y coordinates)

        # Plot the landmarks
        plt.figure()
        plt.scatter(x_coords, y_coords, color="red", s=10)
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.gca().invert_yaxis()
        plt.title(f"Frame {frame_index + 1}")

        # Save the frame as an image
        plt.savefig(f"{output_dir}/frame_{frame_index + 1:04d}.png")
        plt.close()


def create_video(frame_dir, output_path, frame_rate=10):
    frame_files = sorted([os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.endswith(".png")])

    if not frame_files:
        raise ValueError("No frame files found in the output directory")

    # Read the first frame to get dimensions
    first_frame = cv2.imread(frame_files[0])
    height, width, _ = first_frame.shape

    # Create a video writer
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (width, height))

    # Write frames to the video
    for frame_file in frame_files:
        frame = cv2.imread(frame_file)
        video_writer.write(frame)

    video_writer.release()


def main():
    # Read the CSV file
    path = "/Users/pedrootavionascimentocamposdeoliveira/PycharmProjects/hiveLabResearch/data"
    frame_data_list = load_data(path)

    if not frame_data_list:
        print("No data loaded. Please check the input path and file contents.")
        return

    # Create frames
    output_dir = "frames"
    create_frame_images(frame_data_list, 4, output_dir)

    # Create video
    video_path = "/Users/pedrootavionascimentocamposdeoliveira/PycharmProjects/hiveLabResearch/output/validation.mp4"
    create_video(output_dir, video_path)

    print("Video saved as validation.mp4")


if __name__ == "__main__":
    main()
