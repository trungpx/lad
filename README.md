Partial source code for paper: [LAD: A Hybrid Deep Learning System for Benign Paroxysmal Positional Vertigo Disorders Diagnostic](https://ieeexplore.ieee.org/document/9924165), IEEE Access 2022.
# Using the pupil location to detect the horizontal and vertical beatings. Real time pupil detection in noisy images

Finding pupil location inside the eye image is the prerequisite for eye tracking. A baseline is a hybrid model by inspiring YOLO, Network in Network and using Inception as the core CNN to predict the pupil location inside the image of the eye. From the [repo](https://github.com/isohrab/Pupil-locator): To evaluate the model, the publicly available datasets [ExCuse, ElSe, PupilNet](http://www.ti.uni-tuebingen.de/Pupil-detection.1827.0.html) were used. The results surpass previous state of the art result (PuRe) by 9.2% and achieved 84.4%. The model's speed in an Intel CPU Core i7 7700 is 34 fps and in GPU gtx1070 is 124 fps. The images for training are noise free and the pupil is evident. The images are automatically labeled by [PuRe](https://arxiv.org/pdf/1712.08900.pdf) and the ground truth is the parameter of an ellipse.

### New algorithm for detecting beatings
We developed the algorithm using the detected pupil positions in the given videos to get exact the time/frames when the beating happens, i.e. horizontal and vertival beatings, which are crucial for diagnosis of BPPV types.

### Run model
to predict the pupil location in a video, and detect the beatings, use the command:
```
python inferno.py PATH_TO_VIDEO_FILE

to run for beating detection, execute the below command:
# bash detect_beating.sh

```

### Demos
There are some demos videos for beating detected of each eye with *.avi format in this repo.

### Acknowledgement 
This work was supported in part by the Institute for Information \& communications Technology Promotion (IITP) grant funded by the Korean government (MSIT) (No. 2021-0-01381, Development of Causal AI through Video Understanding) and was partly supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (No. 2022R1A2C201270611).

Code base for pupil locator was adopted from this [repo](https://github.com/isohrab/Pupil-locator).
