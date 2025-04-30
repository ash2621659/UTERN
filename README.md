## Experimental setup:
The experimental framework is implemented in MATLAB 2024b 
## Dataset and Feature Descriptions:
### Visual Features (128):
Stored in the directory video_features_128. These features are extracted using the VGGFace network, pre-trained and optimized on the AffectNet dataset.

### Visual Features (9):
Located in the directory video_features_9. These are extracted using MediaPipe facial landmarks, focusing on manually marked facial regions relevant to emotion.

### Audio Features (13):
Found in the directory audio_features_13_new. These are MFCC features derived from the mel spectrum. A total of 13 features are extracted per audio segment.

### Labels:
The labels directory contains the ground truth class labels for the Cream-D audio-video dataset.

## Code Files:
###  extracting_features_4096_128.m
Extracts 128-dimensional visual features using the VGGFace network.

###  extracting_features_9.m
Extracts 9 facial region-based features using MediaPipe.

###  extracting_features_audio_13.m
Extracts 13 mel spectrum-based MFCC audio features.

### training_testing_all_models.m
Trains all models, performs CCA-based feature alignment, and conducts testing.

### ga_wts_optimization.m
Optimizes the weights for the three models using Genetic Algorithm (GA).

## Features: 
Pre-extracted auddio and visual features for the CREMA-D dataset are available to download and use at:  https://livewarwickac-my.sharepoint.com/:f:/g/personal/u2066241_live_warwick_ac_uk/Ep_IfUnTbexOljsZ9ptTMw0BAy-33sHsNgo_QgzPqTdoXA?e=U2xnGc

