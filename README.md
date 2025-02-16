# Fire Fighting Truck

## Overview

This project introduce an autonomous fire truck model that uses artificial intelligence to automatically detect, ignite and extinguish fires. The vehicle is equipped with diverse sensors, camera systems and automatic fire extinguishing equipment. Test results on real models show that self-propelled fire trucks are capable of effectively detecting and igniting fires. The artificial intelligence system is trained to recognize signs of a fire and automatically adjust the vehicle's direction of movement for quick and safe access. The truck while also allowing for manual control by hand tracking. 

## Hardware 
Main component:
- Camera USB 2.0
- Raspberry Pi 4
- DC engine and DC motor controller
- Power supply (Battery 12V)
- Water Pump

![Install Circuit](/Images/image-1.png)
## Fire detection
Use CNN model to detect fire use real-time images captured by USB camera. Output from CNN model is then fed into the fire handling algorithm, if it is fire the truck triggers an alarm and send mail to our e-mail, navigate to the fire source and extinguishes it by using water pump.

### Train Model
We use two different datasets for training. 
![Fire Dataset](/Images/image-2.png)
- The first dataset contains fire images collected through a camera and uses a fire cascade.xml file to detect fire containers. We then crop out and keep only the parts of the image that contain fire for use in training the model.
![Fire Images](/Images/image-3.png)
- The second dataset contains nonfire images collected through a camera. Non-fire photos can be photos of objects or light to avoid misidentification. We then crop out and keep only the parts of the image that contain object for use in training the model.
![Nonfire Images](/Images/image-4.png)

## Manual Control by handtracking
Use an LSTM model to recognize hand gestures using MediaPipe. Images are captured from a PC camera, then processed by the model and results are sent through Flask. The Raspberry Pi receives the API containing the model results, then performs control actions for the vehicle based on the results.

### Train Model
We use MediaPipe to draw hand features, then capture images from the camera, saving them as a matrix of points on the hand.
The dataset is divided into 5 categories of gestures corresponding to 5 actions: forward, backward, left, right, and stop.
You can see details of dataset in our notebook.

## How it work ?
- **If fire is detected**, the car will trigger an alarm. The system will analyze the frame to determine if the fire is on the left or right side, then turn in the corresponding direction.
![Turn Left/Right](/Images/image-8.png)
- If the truck approaches a fire, it will compare the frame of the fire to a threshold to determine if the fire is close or far away. If the fire is small (not within the range of the water pump), the car will continue moving.
![Forward to fire source](/Images/image-9.png)
- If the fire is large, the system will automatically pump water to extinguish the fire.
![Water Pumping](/Images/image-10.png)


## Demo
[![Watch the video](https://img.youtube.com/vi/VYjh2gSFyoI/0.jpg)](https://www.youtube.com/watch?v=VYjh2gSFyoI)

[![Watch the video](https://img.youtube.com/vi/30Bi-uD-vxc/0.jpg)](https://www.youtube.com/watch?v=30Bi-uD-vxc)
## Results
Firefighting truck that autonomously detects fire, navigates to the fire, and extinguishes it, while also allowing for manual control by hand tracking. 
