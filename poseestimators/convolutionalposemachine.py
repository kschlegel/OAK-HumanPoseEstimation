"""
Decode convolutional pose machine type networks, as found in the OpenVino model
zoo.
"""

import numpy as np

from .poseestimator import PoseEstimator


class ConvolutionalPoseMachine(PoseEstimator):
    """
    Decode convolutional pose machine type networks.
    """
    # The order of the landmarks in the models output
    landmarks = [
        'nose', 'right shoulder', 'right elbow', 'right wrist',
        'left shoulder', 'left elbow', 'left wrist', 'right hip', 'right knee',
        'right ankle', 'left hip', 'left knee', 'left ankle', 'right eye',
        'left eye', 'right ear', 'left ear'
    ]

    # The connections to draw, each element is a pair of indices into above
    # landmarks array
    connections = [[0, 1], [1, 3], [0, 2], [2, 4], [5, 6], [5, 7], [7, 9],
                   [6, 8], [8, 10], [5, 11], [6, 12], [11, 12], [11, 13],
                   [13, 15], [12, 14], [14, 16]]

    # Possible options for tuning the performance of the estimator
    # See poseestimator.PoseEstimator._general_options for a description
    # All models automatically have options defined in
    # poseestimator._general_options options (check there for which these are)
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
        super().__init__(model_config, **kwargs)
        # heatmaps are downsampled by a factor of 8 from the input
        self._output_shape = (model_config["input_size"][1] // 8,
                              model_config["input_size"][0] // 8)

    def _get_keypoints(self, heatmaps):
        """
        Parse the heatmaps to obtain all detected keypoints.

        This is the single person model so returns "personwise" keypoints
        straightaway.

        Parameters
        ----------
        heatmaps : numpy array
            heatmaps as retrieved from the network output

        Returns
        -------
        keypoints
            keypoints is a numpy array of shape (1,self._num_keypoints,3). Each
            keypoint consist of (x,y,confidence).
        """
        keypoints = np.zeros((1, self._num_keypoints, 3), dtype=np.float32)
        flat_hm = heatmaps.reshape((1, heatmaps.shape[1], -1))
        positions = np.argmax(flat_hm[0], axis=1)
        positions = np.unravel_index(positions, heatmaps.shape[-2:])
        for i, (x, y) in enumerate(zip(*positions)):
            if heatmaps[0, i, x, y] >= self._detection_threshold:
                keypoints[0, i] = (y * 8, x * 8, heatmaps[0, i, x, y])

        return keypoints

    def decode_results(self, outputs):
        """
        Decode network outputs into keypoint data.

        Takes the extracted network outputs and parses them to obtain an array
        of keypoints.

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
            Numpy array of shape (1, self._num_keypoints, 3). This is a single
            person model, for compatibility downstream keypoints are returned
            as a personwise array with the first axis always of length 1.
        """
        keypoints = self._get_keypoints(heatmaps=outputs[0])
        return keypoints
