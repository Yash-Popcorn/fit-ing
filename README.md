# Fit-ing (Windows)
Won 1st place in Tri Vally Hacks
#### Hackathon Link: https://www.trivalleyhacks.org/#gallery

![pose](https://developers.google.com/static/mediapipe/images/solutions/pose_landmarks_index.png)

Fitness Companion replacing your trainer with a cost-effective solution. Change the future with a virtual assistant

## What it Does
The fitness companion opens a new window of your camera and starts listening for input. You have four options for workout and yoga: cobra pose, tree pose, pushup, or squats. The app will start listening to an input if the user says a keyword a command will run. You can start doing the workout and the companion will start counting the amount of work and calories burnt throughout the process.

## Installation & Setup
YOU MUST HAVE A WINDOWS FOR THIS TO WORK

YOU MUST INSTALL ALL PACKAGES ALSO

```python
# Clone the github code
git clone git@github.com:Yash-Popcorn/fit-ing.git

python index.py
```

## Tools
Open-CV (Livestremaing the video)

Pico Voice (Used porcupine to listen for keywords)

Media Pipe (Pose estimation and calculated angles between lardmark keypoints)

## Roles

Yash Seth: Combined Pico Voice with Open CV by using multiprocessing module and coded the UI functionality on the window screen. Worked on the visuals of the slide presentation.

Prashant Vaid: Wrote the math calculations for squats and pushups using trignometry and wrote the entire logic for the two poses. Worked with Arnav on tree pose and cobra pose.

Arnav Aggarwal: Improved the math calculations for tree pose and cobrapose and helped testing. Collected the statistics for the slide presentation.
