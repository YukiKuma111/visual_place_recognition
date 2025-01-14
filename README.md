# visual_place_recognition

This project provides a solution to address the issue of manually using the `Estimate 2D Pose` tool in RViz for re-localization during SLAM processes. The implemented scripts automate the collection, storage, and matching of camera images and feature data during the mapping process to help the robot provide an approximation of its current pose when relocalization is required after restarted.


## Installation

1. Prerequisites

    - ROS
    - OpenCV with Python bindings
    - numpy

2. Clone the Repository

    ```
    git clone https://github.com/YukiKuma111/visual_place_recognition.git
    ```


## Usage

### 1. Feature Extraction

Run the feature extraction node during the SLAM mapping process:

```
python feature_extraction_with_tf.py
```

There are __3__ parameters should be checked before running:

1) camera_topic
2) tf_topic
3) save_interval

__Purpose:__

 - Collects camera images and extracts ORB features at regular intervals during the SLAM process.

__Key Features:__

 - Subscribes to the camera topic (e.g., `/usb_cam/image_raw`).
 - Extracts and saves keypoints, descriptors, and robot poses to a `features_database.pkl` file.
 - Saves data periodically for minimal computing power.


### 2. Re-Localization

Run the feature matching node after restarting the robot:

```
python pose_estimation_from_features.py
```

There are __2__ parameters should be checked before running:

1) feature_db_path
2) image_topic

__Purpose:__

 - Matches live camera images with previously saved features to determine the robot's current pose.

__Key Features:__

 - Subscribes to the live camera topic (e.g., `/usb_cam/image_raw`).
 - Loads the saved feature database (`features_database.pkl`).
 - Matches ORB descriptors using a brute-force matcher.
 - Publishes the matched pose to a ROS topic (e.g., `/matched_pose`).


### 3. Static Image Publishing (For Testing)

To test the feature matching node with a static image:

```
python image_publisher.py
```

There are __1__ parameters should be checked before running:

1) image_path

__Purpose:__

 - Publishes static images to a specified ROS topic for testing or simulation purposes.

__Key Features:__

 - Reads an image file and publishes it at a specified frequency.
 - Useful for testing the feature matching functionality without live camera input.


## File Structure

```
visual_place_recognition/
├── feature_extraction_with_tf.py
├── image_publisher.py
├── pose_estimation_from_features.py
└── README.md

0 directories, 4 files
```