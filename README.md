# mediapipe_faceswap
faceswap with media pipe model and dlib model.
# Applications: 
 ## realtime_face_swaping.py:
 face swap with Dlib model. 
 ### usage: 
 put your base images in the *change_face_list = []* variable. If the developer prefers to store the photos outside this workspace, please specify the whole path. 
 example: change_face_list = ["unai.jpg"]
 ```python
 usage: python3 realtime_face_swapping.py
 ```
 ## media_pipe_sample.py
 face swap with mediapipe landmark model.
 ### usage: 
 put your base images in the *change_face_list = []* variable. If the developer prefers to store the photos outside this workspace, please specify the whole path. 
 example: change_face_list = ["unai.jpg"]
  ```python
 usage: python3 media_pipe_sample.py
 ```
# dependencies: 
- numpy
- mediapipe
- dlib 
- opencv python

