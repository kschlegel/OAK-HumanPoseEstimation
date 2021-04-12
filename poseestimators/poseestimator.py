from abc import ABC, abstractmethod

import numpy as np
import cv2


class PoseEstimator(ABC):
    """
    Abstract base class for pose estimators decoding the NN results.

    Provides a common interface for all decoders. Specifically provides:
    - get_input_frame method to convert an arbitrary image into the right
      size/shape and array configuration for the NN input.
    - get_original_frame method to reverse the transformations of
      get_input_frame to convert back frames obtained through the NNs
      passthrough channel
    - get_pose_data method that takes the raw NN output packet and decodes it
      to return a personwise keypoint array. Keypoints are automatically
      transformed back into coordinates for the original frame from NN input
      coordinates
    - draw_results method to draw personwise keypoints onto an image
    - handling of hyperparameters for the algorithm, parameters can be
      specified in _specific_options dictionary

    Derived classes need to:
    - implement decode_results method, which takes a list of the individual
      network outputs, already reshaped into their true shape and decodes them
      into personwise keypoints in NN input coordinates. Transformation back
      into original coordinates is handled automatically.
    - set self._output_shape in the constructor as this depends on the network
      architecture and determines the correct shape of the output arrays
    - lists of the landmarks and their order output by the system and the
      connections between landmarks.
    """
    # empty placeholders for landmark and connection lists
    landmarks = []
    connections = []

    # Possible options(hyperparameters) for tuning the decoding algorithm
    # Dict of the form
    #     option_name: {'max_val': maximum value for slider,
    #                   'divider': divide slider by this value for use,
    #                   'default': default value,
    #                   'description': Description for command line arg}
    # Example: detection_threshold:{"max_val": 100, "divider": 100, "default":
    # 30, "description": "..."}  means the slider for detection_threshold will
    # range from 0 to 100 with a default of 30 and the slider value will be
    # divided by 100 before use as threshold. You don't have to do this
    # transformation yourself, e.g. if you want to specify a percentage value
    # with a step size of 1% define an option with max_val 100 and divider 100
    # and the self._option variable will automatically be set to
    # integer_val/divider and thus be percentage value in the range 0 to 1.
    # The command line arg will be described as "..." when running -h
    # _general_options will apply to all PoseEstimator objects,
    # _specific_options allows to define extra options in subclasses which only
    # apply there
    _general_options = {
        "detection_threshold": {
            "max_val":
            100,
            "divider":
            100,
            "default":
            15,
            "description":
            "Set the confidence threshold for keypoint detection in %%."
        }
    }
    # As _general_options but to be overwritten by specific implementations for
    # defining their unique parameters
    _specific_options = {}

    def __init__(self, model_config, **kwargs):
        """
        Parameters
        ----------
        model_config : dict
            Dictionary with configuration parameters of the model read from the
            'models.json' file
        kwargs:
            Command line arguments to determine selected hyperparameters.
        """
        self._input_shape = tuple(model_config["input_size"])
        self._output_layers = model_config["output_layers"]
        self._num_keypoints = len(self.landmarks)

        for option_name, option in self.get_options():
            if option_name in kwargs:
                # kwargs (command line args take priority)
                value = kwargs[option_name]
            else:
                value = option["default"]
            self.set_option(option_name, value)

        self._pad_top = None
        self._pad_left = None
        self._scale_factor = None

    @classmethod
    def get_options(cls):
        """
        Iterate over all options, general and class specific.
        """
        for option in cls._general_options.items():
            yield option
        for option in cls._specific_options.items():
            yield option

    @classmethod
    def get_general_options(cls):
        """
        Iterate over options common to all PoseEstimator classes.
        """
        for option in cls._general_options.items():
            yield option

    @classmethod
    def get_specific_options(cls):
        """
        Iterate over options specific to the PoseEstimator class at hand.
        """
        for option in cls._specific_options.items():
            yield option

    def set_option(self, option, value):
        """
        Set the given option to the given value.

        Values are specified in integer domain as this is what the OpenCV
        trackbars support. For float value the divider specified in the option
        is applied before assigning the value.
        Options not applying to the model are ignored.

        Parameters
        ----------
        option : str
            identifier of the option in the options dict
        value : int
            Value to set the option to, as int before divider was applied.
        """
        for option_dict in ("_general_options", "_specific_options"):
            option_dict = getattr(self, option_dict)
            if option in option_dict:
                if option_dict[option]["divider"] != 1:
                    value /= float(option_dict[option]["divider"])
                setattr(self, "_" + option, value)
                break

    def get_input_frame(self, frame):
        """
        Pads and rescales the frame to fit the network input size.

        Takes an arbitrary picture and scales it to fit into the networks
        receptive field. Pads either horizontally or vertically to fit the
        exact size required as input by the network.

        Parameters
        ----------
        frame : numpy array
            Frame to be send to device for processing

        Returns
        -------
        nn_frame : numpy array
            Frame suitable to be passed into the pose estimation network.
        """
        self._scale_factor = min(self._input_shape[1] / frame.shape[0],
                                 self._input_shape[0] / frame.shape[1])
        scaled = cv2.resize(frame, (int(frame.shape[1] * self._scale_factor),
                                    int(frame.shape[0] * self._scale_factor)))
        pad_width = (self._input_shape[0] - scaled.shape[1]) / 2
        pad_height = (self._input_shape[1] - scaled.shape[0]) / 2
        # floor&ceil values to account for possibly odd amount of padding
        self._pad_top = int(np.floor(pad_height))
        self._pad_left = int(np.floor(pad_width))
        self._pad_bottom = int(np.ceil(pad_height))
        self._pad_right = int(np.ceil(pad_width))
        nn_frame = cv2.copyMakeBorder(scaled, self._pad_top, self._pad_bottom,
                                      self._pad_left, self._pad_right,
                                      cv2.BORDER_CONSTANT)
        return nn_frame.transpose(2, 0, 1)

    def get_original_frame(self, frame):
        """
        Transforms a frame from NN input shape back into original shape.

        Removes any padding and scaling applied in get_input_frame.

        Parameters
        ----------
        frame : numpy array
            Frame to be transformed

        Returns
        -------
        frame : numpy array
            Frame with transformations removed
        """
        if self._pad_top is not None:
            frame = frame[self._pad_top:frame.shape[0] - self._pad_bottom,
                          self._pad_left:frame.shape[1] - self._pad_right]
        if self._scale_factor is not None and self._scale_factor != 1:
            frame = cv2.resize(frame,
                               (int(frame.shape[1] / self._scale_factor),
                                int(frame.shape[0] / self._scale_factor)))
        return frame

    def get_pose_data(self, raw_output):
        """
        Decodes raw outputs into pose data.

        Retrieves network outputs from raw data packet and decodes results into
        a personwise keypoint array. Decoding is done by calling the abstract
        method decode_results which needs to be implemented by the individual
        classes.
        After decoding any padding and scaling that may have been applied in
        get_input_frame is removed from the keypoints to fit the original
        frame.

        Parameters
        ----------
        raw_output : depthai.NNData object
            Raw output from the neural network retrieved from depthai pipeline.

        Returns
        -------
        personwise_keypoints : numpy array
            Numpy array of shape (n, self._num_keypoints, 3) where n is the
            number of detected people. For each person contains each
            landmark as (x,y,confidence), if a landmark was not detected all 3
            values will be zero.
        """
        outputs = self._convert_raw_outputs(raw_output)
        personwise_keypoints = self.decode_results(outputs)

        # If the frame got scaled and padded before sending the frame to the NN
        # (in get_input_frame in case of  sending a local file) we need to
        # remove this padding and scaling from the keypoints
        if personwise_keypoints.shape[0] > 0:
            if self._pad_top is not None:
                keypoint_ids = np.nonzero(personwise_keypoints[:, :, -1])
                personwise_keypoints[keypoint_ids[0], keypoint_ids[1], :2] -= [
                    self._pad_left, self._pad_top
                ]
            if self._scale_factor is not None and self._scale_factor != 1:
                personwise_keypoints[:, :, :2] /= self._scale_factor

        return personwise_keypoints

    def draw_results(self, personwise_keypoints, frame):
        """
        Draws the detected keypoints onto the given frame.

        Skips keypoints that were not detected (by assuming those have a
        confidence value of zero).

        Parameters
        ----------
        personwise_keypoints : numpy array
            Keypoint array of the form [n,self._num_keypoints, 3] with
            (x,y,confidence) information for each keypoint as returned by
            get_pose_data.
        frame : numpy array
            The frame to draw on. The frame is modified in place.
        """
        for person_id, person in enumerate(personwise_keypoints):
            # Draw keypoints
            if person_id % 3 == 0:
                point_colour = (0, 0, 255)
                line_colour = (255, 0, 0)
            elif person_id % 3 == 1:
                point_colour = (0, 255, 0)
                line_colour = (0, 0, 255)
            else:
                point_colour = (255, 0, 0)
                line_colour = (0, 255, 0)
            for i in range(len(person)):
                # Confidence = 0 means not detected
                if person[i][2] == 0:
                    continue
                cv2.circle(frame, tuple(person[i][0:2].astype(int)), 2,
                           point_colour, -1, cv2.LINE_AA)
            # Draw connections
            for connection in self.connections:
                # Confidence = 0 means not detected
                confidences = person[connection, 2]
                if 0 in confidences:
                    continue
                pt1, pt2 = person[connection, :2].astype(int)
                cv2.line(frame, tuple(pt1), tuple(pt2), line_colour, 1,
                         cv2.LINE_AA)

    def _convert_raw_outputs(self, raw_output):
        """
        Extract the raw outputs from a depthai.NNData object.

        Parameters
        ----------
        raw_output : depthai.NNData object
            Raw output from the neural network retrieved from depthai pipeline.

        Returns
        -------
        outputs : list
            numpy arrays containing the various model outputs as float32 and
            reshaped.
        """
        outputs = [
            np.array(raw_output.getLayerFp16(self._output_layers[i]),
                     dtype=np.float32).reshape((1, -1) + self._output_shape)
            for i in range(len(self._output_layers))
        ]
        return outputs

    @abstractmethod
    def decode_results(self, outputs):
        """
        Decode network outputs into keypoint data.

        After retrieving and reshaping the outputs from the network the
        decoding process can be very different between architectures and should
        be implemented in each specific class. This function should take the
        extracted network outputs and parse them to obtain an array of
        keypoints grouped by persons.

        Parameters
        ----------
        outputs : list of numpy arrays
            Output arrays retrieved from the network. As returned by
            PoseEstimator._convert_raw_outputs. The order of the outputs is
            exactly as the order of output layers in the models.json
            configuration file.

        Returns
        -------
        personwise_keypoints : numpy array
            Numpy array of shape (n, self._num_keypoints, 3) where n is the
            number of detected people. For each person contains each
            landmark as (x,y,confidence), if a landmark was not detected all 3
            values will be zero.
        """
        ...
