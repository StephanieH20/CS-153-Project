# CS-153-Project
Spying on grasshoppers

The dataset and videos were too large, so they need to be added into the same folder as the rest of the files.
This includes data_recording.mat and the video clips. Add the videos to a folder and name it "video_clips"

First, run locust.precop.m to obtain frame data for sequences 1-6 (corresponding to 133 videos 225, 650, 750, 850, 950, and 115). I only got the data for 133 videos; due to angle scaling, we decided the 96 and 146 videos could be excluded from the data and I also had some trouble handling the other videos.

Once that's complete, run locust_.py, the main script that generates the augmented video along with the results. The script will ask you which sequence you want to run the algorithm on (sequences 1-7, and 9). Once you select, wait for around a minute (depending on which sequence you chose), then the video should pop-up with the augmented orientations. 

After the script runs, .txt files will be generated containing the ID's of locusts present in each frame, their centers, and their calculated angles. Additionally, a window should appear showing a graph of the error over time. The script also prints out the net orientation and net axis errors. 
