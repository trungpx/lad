**Partial code for paper** [LAD: A Hybrid Deep Learning System for Benign Paroxysmal Positional Vertigo Disorders Diagnostic](https://ieeexplore.ieee.org/document/9924165), IEEE Access, 10/2022. \
**Authors**: [Trung Xuan Pham](https://scholar.google.com/citations?user=4DkPIIAAAAAJ), [Jin Woong Choi](https://scholar.google.com/citations?user=4yrEskMAAAAJ), [Rusty John Lloyd Mina](https://scholar.google.com/citations?user=2pJdTTsAAAAJ&hl=en), Thanh Nguyen, Sultan Rizky Madjid, and [Chang Dong Yoo](https://scholar.google.com/citations?user=Dp3s8JQAAAAJ&hl=en).

Overall system for BPPV diagnosis process using artificial intelligent:
![Network](https://github.com/trungpx/lad/blob/main/images/bppv_diagnosis.png)

Seven postures for patient setup with Dix-Hallpike manoveurs: \
![Postures](https://github.com/trungpx/lad/blob/main/images/postures.png)

### Using the pupil location to detect the horizontal and vertical beatings. Real time pupil detection in noisy images

From the [repo](https://github.com/isohrab/Pupil-locator), finding pupil location inside the eye image is the prerequisite for eye tracking. A baseline is a hybrid model by inspiring YOLO, Network in Network and using Inception as the core CNN to predict the pupil location inside the image of the eye. To evaluate the model, the publicly available datasets [ExCuse, ElSe, PupilNet](http://www.ti.uni-tuebingen.de/Pupil-detection.1827.0.html) were used. The results surpass previous state of the art result (PuRe) by 9.2% and achieved 84.4%. The model's speed in an Intel CPU Core i7 7700 is 34 fps and in GPU gtx1070 is 124 fps. The images for training are noise free and the pupil is evident. The images are automatically labeled by [PuRe](https://arxiv.org/pdf/1712.08900.pdf) and the ground truth is the parameter of an ellipse.

### Algorithm for detecting beatings
We developed the algorithm using the detected pupil positions in the given videos to get exact the time/frames when the beating happens, i.e. horizontal and vertival beatings, which are crucial for diagnosis of BPPV types.

### Run model
to run for beating detection, execute the below command:
```
bash detect_beating.sh
```

### Demos
There are demos videos for beating detected of each eye with *.avi format in this repo. Below are examples:

https://user-images.githubusercontent.com/37993448/203952786-b8174bcf-9799-4c48-ac7b-09c040eb6c23.mp4


https://user-images.githubusercontent.com/37993448/203953351-eb03cca9-1b6c-439a-8b7f-8637fcf24a00.mp4



### Acknowledgement 
This work was supported in part by the Institute for Information \& communications Technology Promotion (IITP) grant funded by the Korean government (MSIT) (No. 2021-0-01381, Development of Causal AI through Video Understanding) and was partly supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (No. 2022R1A2C201270611).

Codebase for pupil locator was adopted from this [repo](https://github.com/isohrab/Pupil-locator).

### Cite this paper ([Goolge Scholar](https://scholar.google.com/scholar?cluster=5944041991038126099&hl=en&as_sdt=2005)):
```
@article{pham2022lad,
title={LAD: A Hybrid Deep Learning System for Benign Paroxysmal Positional Vertigo Disorders Diagnostic},
author={Pham, Trung Xuan and Choi, Jin Woong and Mina, Rusty John Lloyd and Nguyen, Thanh Xuan and Madjid, Sultan Rizky and Yoo, Chang D.},
journal={IEEE Access},
year={2022},
volume={10},
pages={113995-114007},
doi={10.1109/ACCESS.2022.3215625}
}
```

