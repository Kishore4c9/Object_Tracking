# Object Tracking and Velocity Estimation

This project implements a real-time computer vision pipeline to detect, track, and calculate the speed of moving objects in video streams.

### Demo
<video src="videos/sample_output.mp4" width="800" controls muted loop>
  Your browser does not support the video tag.
</video>


## Core Technologies

### Object Detection (YOLO)
The system utilizes **YOLO (You Only Look Once)**. This model treats object detection as a single regression problem, straight from image pixels to bounding box coordinates and class probabilities. It is optimized for high frame rates while maintaining high accuracy.



### SORT Tracking
**SORT (Simple Online and Realtime Tracking)** handles object identity. Even if an object is briefly hidden or moves, SORT ensures the system remembers which object is which by using:
* **Kalman Filters:** To predict the future location of an object.
* **Hungarian Algorithm:** To associate new detections with existing tracks.

### Object Velocity Calculation
Velocity is derived from the change in the object's position over a known time interval.
* **Euclidean Distance:** Calculates the gap between coordinates $(x_1, y_1)$ and $(x_2, y_2)$.
* **Mathematical Formula:**
    $$Velocity = \frac{\sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2} \times \text{ppm}}{FPS}$$
    *(where **ppm** is pixels-per-meter and **FPS** is frames per second)*

---

## 🛠 Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Kishore4c9/Object_Tracking.git](https://github.com/Kishore4c9/Object_Tracking.git)
   cd Object_Tracking

---

## ▶️ Execution

2. **Use the following command to start the object tracking and velocity calculation:**
    ```bash
    python object_velocity.py

