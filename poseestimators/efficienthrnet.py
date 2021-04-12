"""
Decode EfficientHRNet-type networks, as found in the OpenVino model zoo.
"""

import numpy as np

from .poseestimator import PoseEstimator


class EfficientHRNet(PoseEstimator):
    """
    Decode EfficientHRNet type networks.
    """
    # The order of the landmarks in the models output
    landmarks = [
        'nose', 'left eye', 'right eye', 'left ear', 'right ear',
        'left shoulder', 'right shoulder', 'left elbow', 'right elbow',
        'left wrist', 'right wrist', 'left hip', 'right hip', 'left knee',
        'right knee', 'left ankle', 'right ankle'
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
        self._output_shape = (model_config["input_size"][1] // 2,
                              model_config["input_size"][0] // 2)

    def _get_keypoints(self, nms_heatmaps, embeddings):
        """
        Parse the nms-heatmaps to obtain all detected keypoints.

        Also returns the type of each landmark and its tag from the embeddings
        map for grouping into people.

        Parameters
        ----------
        nms_heatmaps : numpy array
            nms-heatmaps as retrieved from the network output
        embeddings : numpy array
            embedding maps as retrieved from the network oputput

        Returns
        -------
        keypoints, keypoint_types, tags
            keypoints is a numpy array of shape (k,3), where k is the number of
            detected keypoints. Each keypoint consist of (x,y,confidence).
            keypoint_types is a list of length k, containing for each detected
            keypoint the id of which landmark it is.
            tags is a list of length k, containing for each detected keypoint
            its tag from the embeddings map.
        """
        nms_heatmaps = nms_heatmaps[0]
        nms_heatmaps[nms_heatmaps < self._detection_threshold] = 0

        # Thanks to nms we already have isolated nonzero points, so each
        # corresponds to a detected keypoint
        positions = np.nonzero(nms_heatmaps)

        keypoints = np.zeros((0, 3), dtype=np.float32)
        keypoint_types = []
        tags = []
        for keypoint_id, y, x in zip(*positions):
            conf = nms_heatmaps[keypoint_id, y, x]
            keypoints = np.append(keypoints, [(x * 2, y * 2, conf)], axis=0)
            keypoint_types.append(keypoint_id)
            tags.append(embeddings[0, keypoint_id, y, x])

        return keypoints, keypoint_types, tags

    def _get_personwise_keypoints(self, keypoints, keypoint_types, tags):
        """
        Groups the keypoints into people using the detected tags.

        Tags will be roughly similar numeric values for keypoints belonging to
        the same person. Grouping is done based on minimising pairwise tag
        distances.

        Parameters
        ----------
        keypoints : numpy array
            Array of detected keypoints of shape (k,3) as output by
            _get_keypoints.
        keypoint_types : list
            List of landmark type of each keypoint as output by _get_keypoints
        tags : list
            List of tag of each keypoint as output by _get_keypoints

        Returns
        -------
        personwise_keypoints : numpy array
            Numpy array of shape (n, self._num_keypoints, 3) where n is the
            number of detected people. For each person contains each
            landmark as (x,y,confidence), if a landmark was not detected all 3
            values will be zero.
        """
        # Catch trivial cases
        if keypoints.shape[0] == 0:
            return np.array([])
        elif keypoints.shape[0] == 1:
            person = np.zeros((1, self._num_keypoints, 3))
            person[0][keypoint_types[0]] = keypoints[0]
            return person

        # Create a distance matrix containing the pairwise differences of tags
        # Then flatten and argsort so we can easily iterate pairs in order of
        # distance to group keypoints
        tags = np.array(tags)
        tag_distances = tags[:, None] - tags[None, :]
        tag_distances[tag_distances <= 0] = np.inf
        tag_distances_shape = tag_distances.shape
        tag_distances = tag_distances.flatten()
        sorted_distances = np.argsort(tag_distances)
        # Create a list containing the group number for each keypoint. Set to
        # -1 for ungrouped keypoints. Group_id will be continuously updated as
        # new keypoints get added to groups or groups are merged
        keypoint_groups = [-1] * len(keypoints)
        grouped = 0
        # group_id will count up each time a new grouplet is created.
        group_id = 0
        for next_shortest in sorted_distances:
            i, j = np.unravel_index(next_shortest, tag_distances_shape)
            # Skip if same landmark type as they have to be different people
            if keypoint_types[i] != keypoint_types[j]:
                g1 = keypoint_groups[i]
                g2 = keypoint_groups[j]
                if g1 == g2:
                    # both keypoints have same group value => only need to
                    # update if not yet grouped
                    if g1 == -1:
                        keypoint_groups[i] = group_id
                        keypoint_groups[j] = group_id
                        group_id += 1
                        grouped += 2
                else:
                    # keypoints have different group values. If one is
                    # ungrouped add it to the other ones group. If both are
                    # grouped, merge the two groups.
                    if g1 == -1:
                        keypoint_groups[i] = g2
                        grouped += 1
                    elif g2 == -1:
                        keypoint_groups[j] = g1
                        grouped += 1
                    else:
                        for k in range(len(keypoint_groups)):
                            if keypoint_groups[k] == g2:
                                keypoint_groups[k] = g1
            if grouped == len(keypoints):
                break
        # Create a person for each unique grouplet id in the groups array.
        # Merging can kill previous group ids so use dict to assign arrays to
        # group ids
        persons = {
            person_id: np.zeros((self._num_keypoints, 3))
            for person_id in set(keypoint_groups)
        }
        # copy the keypoints into the right arrays one by one
        for keypoint_id, person_id in enumerate(keypoint_groups):
            persons[person_id][
                keypoint_types[keypoint_id]] = keypoints[keypoint_id]
        return np.stack(list(persons.values()), axis=0)

    def decode_results(self, outputs):
        """
        Decode network outputs into keypoint data.

        Takes the extracted network outputs and parses them to obtain an array
        of keypoints grouped by persons.

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
        keypoints, keypoint_types, tags = self._get_keypoints(
            nms_heatmaps=outputs[1], embeddings=outputs[2])
        personwise_keypoints = self._get_personwise_keypoints(
            keypoints, keypoint_types, tags)
        return personwise_keypoints
