# OAK-HumanPoseEstimation

1. [Introduction](#introduction)
2. [Setup](#setup)
3. [Usage](#usage)
   1. [Command line options](#command-line-options-common-to-all-tools)
   2. [view_image](#view_image)
   3. [view_video](#view_video)
   4. [view_coco](#view_coco)
   5. [eval_coco](#eval_coco)
4. [Adding your own model/decoder](#adding-your-own-modeldecoder)
   1. [How to add your own model](#how-to-add-your-own-model)
   2. [How to add your own decoder](#how-to-add-your-own-decoder)
5. [Roadmap](#roadmap)

## Introduction
This project aims to explore and benchmark human pose estimation systems on the OpenCV AI Kit OAK-1 and OAK-D devices. There are many different general pose estimation methods and each often comes with a range of different implementations/backbones which make different trade-offs between efficiency and accuracy.\
The idea is to be able to easily and quickly be able to switch between different models and decoding algorithms to have a way to evaluate which model and algorithm may be most suitable for ones given use-case based on performance needs. More precisely, for a given project one may need to answer questions such as
* Which pose estimation model family to use, e.g. OpenPose vs EfficientHRNet
* Which size/version of the network/feature extractor backbone to use, e.g. MobileNetV1 vs. MobileNetV2
* Most appropriate hyperparameters of the decoding algorithm, e.g. how does sub-pixel accuracy when extracting keypoints from heatmaps affect accuracy and run time?

The answers to these questions may strongly depend on the individual use-case (in particular factors such as numper of people in the scene, size of people in the scene, illumination, clothing, etc). Thus the question of choosing the right pose estimation system for an application is non-trivial and requires evaluation on a case-by-case basis.

To help with this, this project provides tools to (Tools will be described below in more detail):
* Visually inspect results on single images, videos and live camera stream from an OAK device
* Visually inspect results on the COCO dataset, which contains a wide variety of images
* Evaluate the performance on the COCO dataset using standard pose estimation metrics
* Tune parameters for the decoding algorithm and see the effect in real time in any of the visualisation tools
* Where applicable runtimes of inference on the OAK device and decoding on the host are measured. Overall averages are displayed when the program finishes 
* In video views frame rates are measured and displayed. An average FPS over the entire runtime is displayed when the program finishes

Besides the rough introduction how to use this tool the aim is to produce a fairly clean set of code with a reasonable amount of comments to help getting up to speed with pose estimation on OAK devices. The project is work in progress, questions and contributions welcome. 

## Setup
To visualise the results on image, video or camera stream
* install the requirements by running `pip install -r requirements.txt`
* Download the included model blob files from https://drive.google.com/file/d/1AUszSCMSc5dCATnZn1jK8PzMMyLQ__5M/view?usp=sharing
* Unzip the archive with the model blob files into the models folder.


If you want to use the  tools for evaluation on COCO you need to download the validation data for the COCO keypoint detection task 2017 from here:\
[https://cocodataset.org/#download](https://cocodataset.org/#download)\
and install the official COCO API from here:\
[https://github.com/cocodataset/cocoapi](https://cocodataset.org/#download)

## Usage
In any visualisation window, press q to exit.
### Command line options common to all tools
#### General args
* `-m --model` Select the model to use, current options are:
  openpose1, openpose2_small, openpose2_large, openpose3, efficienthrnet1, efficienthrnet2, efficienthrnet3, global_cpm, openpose3d
* `-sh SHAVES, --shaves SHAVES` Number of shaves to use (model must have been compiled for this number) Defaults to 8 when running on an image/video, 6 when running on camera.
* `--num_inference_threads NUM_INFERENCE_THREADS` Optionally set the number of inference threads for the NN.
* `--nce_per_inference_thread NCE_PER_INFERENCE_THREAD` Optionally set the number of NCEs to use per inference thread.
* `-d DECODER, --decoder DECODER` Optionally select a decoder other different from the models default decoder.
* `--detection_threshold DETECTION_THRESHOLD` Set the confidence threshold for keypoint detection in %. (default is 30)

#### Pose estimator args
Moreover, hyperparameters for individual algorithms to decode the opose estimator results can be made avaialble. Currently these are:
* `--num_paf_samples NUM_PAF_SAMPLES` Number of samples to take from the paf along a potential connection. (default is 10) (only used for OpenPose)
* `--min_paf_score_th MIN_PAF_SCORE_TH` Minimal paf value for a paf sample point to be considered 'good'. (default is 2) (only used for OpenPose)
* `--paf_sample_th PAF_SAMPLE_TH` Percentage of paf samples that need to be good for the connection to be accepted. (default is 4) (only used for OpenPose)

#### COCO args
When using the COCO dataset use `-ds --dataset_path` to specify the path to the COCO dataset.

The COCO dataset contains a number of annotations which have very few keypoints (e.g. only a single wrist). You can optionally drop these ground truth annotations by specifying a minimal amount of keypoints for an annotation to be considered valid, using the `--min_keypoints` flag. Images which do not contain any annotations with at least this number of keypoints are dropped from the dataset in this case.

### view_image
Run pose estimation on a single image from your hard drive and visualise the results on the image. Hyperparameters of the decoding algorithm can be changed and the change of results is visible in real time.

Use the `-i --image` flag to specify the image to use.

### view_video
Run pose estimation on a video file from your hard drive (in infinite loop) or on the OAK camera stream and visualise the results on the video stream. Both the original (full FPS) video stream and a video stream of only the frames that can be processed by the OAK in real time are shown to get an idea of 'what the NN sees'. Hyperparameters of the decoding algorithm can be changed and the change of results is visible in real time.

Use the `-v --video` flag to specify the video to use, if not given the OAK camera will be used.

### view_coco
Run pose estimation one image from the COCO dataset containing keypoint annotations at a time and visualise the results on the image. Also displays the image with ground truth annotations for comparison. Images can be iterated (with the n & p keys for previous/next image) to get an impression of performance on a variety of different images. Hyperparameters of the decoding algorithm can be changed and the change of results is visible in real time.

Specify the location of the dataset using the `-ds --dataset_path` flag.

### eval_coco
Run pose estimation on all COCO images with keypoint annotations and evaluates the results using the COCO API. Hyperparameters of the decoding algorithm can be set for each individual run. Results for a given model and detection threshold are saved and re-used on a later run to provide a summary. Delete the file "results_model_detectionthreshold.json" to re-run the model.

Specify the location of the dataset using the `-ds --dataset_path` flag.

## Supported pose estimators
 - [x] OpenPose
    - openpose1: MobileNetV1 based (from [OpenVino open model zoo](https://github.com/openvinotoolkit/open_model_zoo))
	- openpose2_small/openpose2_large: MobileNetV2 based, different complexity (from [PINTO model zoo](https://github.com/PINTO0309/PINTO_model_zoo))
	- openpose3: MobileNetV3 based (from [PINTO model zoo](https://github.com/PINTO0309/PINTO_model_zoo))
	- openpose3d: Combines OpenPose 2d keypoint estimation with a lift to 3d keypoints. Here only the first step is used (from [OpenVino open model zoo](https://github.com/openvinotoolkit/open_model_zoo))
 - [x] EfficientHRNet
    - efficienthrnet1: small EfficientHRNet (from [OpenVino open model zoo](https://github.com/openvinotoolkit/open_model_zoo))
    - efficienthrnet2: medium EfficientHRNet (from [OpenVino open model zoo](https://github.com/openvinotoolkit/open_model_zoo))
    - efficienthrnet3: large EfficientHRNet (from [OpenVino open model zoo](https://github.com/openvinotoolkit/open_model_zoo))
 - [x] Convolutional Pose Machine
    - global_cpm: Convolutional Pose Machine with global context (from [OpenVino open model zoo](https://github.com/openvinotoolkit/open_model_zoo))
 - [ ] PoseNet
 
 Some more information about each of the individual models is printed when running any of the tools using the model.

## Adding your own model/decoder
### How to add your own model
1. Create a .blob file of your model using the OPenVino myriad_compile tool
2. Place it in the models folder, following the naming convention UserDefinedName_shX.blob, where X denotes the number of shaves the blob was compiled for. If you include `ovYYYY.X` in the filename, where YYYY.X denotes the OpenVino version used to compile the blob, then the OpenVino version will automatically extracted when loading the blob and the DepthAI pipeline is set to use this version.
3. Add an entry to models.json of the form
```
model_identifier: {
	"source": "Where the model is from",
	"model": "Name of the model",
	"url": "URL where the model can be found",
	"blob": "Filename of the blob, excluding the _shX.blob",
	"feature_extractor": "Network architecure used as feature extractor",
	"input_size": [width, height],
	"color_order": "BGR" or "RGB",
	"output_layers": ["name of first output layer", "name of second output layer"],
	"decoder": "Name of default decoder class"
}
```
Note that the order in which you specify the outputs layers in the configuration is exactly the order in which they will provided to your decoder class.\
As an example consider:
```
"openpose1": {
    "source": "openvino_open_model_zoo",
    "model": "human-pose-estimation-0001",
    "url":
    "https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/human-pose-estimation-0001",
    "blob": "human-pose-estimation-0001_290321_ov2021.2",
    "feature_extractor": "MobileNetv1",
    "input_size": [456, 256],
    "color_order": "BGR",
    "output_layers": ["Mconv7_stage2_L2", "Mconv7_stage2_L1"],
	"decoder": "OpenPose"
},
```
4. Assuming you have a compatible decoder you can now run your model in any of the tools by using -m "model_identifier" as command line argument

### How to add your own decoder
1. In the poseestimators directory create a new file for your new decoder, with the filename the decoder name in all lower case, e.g. openpose.py
2. In this file define the class for your decoder, deriving from the abstract base class PoseEstimator, e.g.
```
from .poseestimator import PoseEstimator
class OpenPose(PoseEstimator):
	...
```
3. Define as static properties the keypoint order output by the model and the connections (as pairs of keypoint ids) to be drawn when visualising the results, e.g.
```
class EstimatorClass(PoseEstimator):
	landmarks = [
		'nose', 'neck', 'right shoulder', 'right elbow', 'right wrist',
		'left shoulder', 'left elbow', 'left wrist', 'right hip', 'right knee',
		'right ankle', 'left hip', 'left knee', 'left ankle', 'right eye',
		'left eye', 'right ear', 'left ear'
	]
	connections = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8],
		[8, 9], [9, 10], [1, 11], [11, 12], [12, 13], [1, 0],
		[0, 14], [14, 16], [0, 15], [15, 17]]
```
(Note you should use exactly these names for COCO evaluation so that landmarks can automatically be matched/rearranged)

4. Define as static property any hyperparameters for the algorithm that should be tuneable, in the form:
```
class EstimatorClass(PoseEstimator):
	_specific_options = {
		"parameter_name": {
			"max_val": 20,
			"divider": 20,
			"default": 10,
			"description": "Help text for command line argument."
		}
	}
```
Parameter values here have to be specified as integer values to be tuneable by sliders in an OPenCV window. The value will be within the range 0 to `max_val` and the slider value, divided by `divider` will be available in the class as `self._parameter_name`. Unless specified as command line option the value defaults to `default`. E.g. above would create a slider ranging from 0 to 20, defaulting to 10 and in the EstimatorClass the value `self._parameter_name` would be available, it would be in the range 0 to 1, with a step size of 0.05 and a default of 0.5. You won't need to transform the integer value yourself, if you are looking to define a percentage with a stepsize of 1% define an option with max_val=100 and divider=100 and the variable in your class will automatically be mapped into the range 0 to 1 by dividing the integer value by the divider.\
Note that every decoder by default has a parameter 'detection_threshold' ranging from 0 to 100 with a divider of a 100 (i.e. a percentage value with a stepsize of 1%, which can be accessed in the class as self._detection_threshold, where the value will be between 0 and 1).

5. In the class constructor define the shape of the outputs of your model, e.g. OpenPose heatmaps are one eigth of the size of the input image.
```
def __init__(self, model_config, **kwargs):
    super().__init__(model_config, **kwargs)
	self._output_shape = (model_config["input_size"][1] // 8,
		                  model_config["input_size"][0] // 8)
```
6. Implement the abstract method `decode_results(self, outputs)` from the PoseEstimator base class. This class takes the network outputs that have been extracted from the raw data packet and reshaped into their target shape and handles decoding the results into an array of personwise keypoints (i.e. of the shape [num_persons, num_landmarks, (x,y,confidence)]).
As an example from the OpenPose implementation: 
```
def decode_results(self, outputs):
	heatmaps = outputs[0]
	pafs = outputs[1]

	keypoints, landmarkwise_keypoints = self._get_keypoints(heatmaps)
	# Get lists of detected keypoint connections from pafs
	pairs = self._get_pairs(pafs, keypoints, landmarkwise_keypoints)
	# Assemble the keypoints and connections into people
	personwise_keypoints = self._get_personwise_keypoints(pairs, keypoints)
	return personwise_keypoints
```
Here `_get_keypoints`, `_get_pairs` and `_get_personwise_keypoints` are functions containing the individual logic to decode the heatmaps, part affinity fields and assembling those results into individual people.

7. To register command line arguments for the classes hyperparameters import it in poseestimators/__init__.py and add it to the pose estimator list in the line
`for estimator in (OpenPose, EfficientHRNet, ConvolutionalPoseMachine):`
in `add_poseestimator_args`.

For a full example look at e.g. poseestimators/openpose.py

## Roadmap
 - [ ] Add more in depth performance evaluation with various standard pose estimation metrics
 - [ ] Add a grid search of hyperparameters to maximise performance on a given metric
