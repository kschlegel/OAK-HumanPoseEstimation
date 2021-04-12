"""
Decode OpenPose type networks, both from the OpenVino model zoo and PINTOs
model zoo.

This code is heavily based on the original depthai_demo openpose parser:
https://github.com/luxonis/depthai/blob/main/depthai_helpers/openpose_handler.py
"""

import numpy as np
import cv2

from .poseestimator import PoseEstimator


class OpenPose(PoseEstimator):
    """
    Decode OpenPose type networks.
    """
    # The order of the landmarks in the models output
    landmarks = [
        'nose', 'neck', 'right shoulder', 'right elbow', 'right wrist',
        'left shoulder', 'left elbow', 'left wrist', 'right hip', 'right knee',
        'right ankle', 'left hip', 'left knee', 'left ankle', 'right eye',
        'left eye', 'right ear', 'left ear'
    ]

    # The connections to draw, each element is a pair of indices into above
    # landmarks array
    connections = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8],
                   [8, 9], [9, 10], [1, 11], [11, 12], [12, 13], [1, 0],
                   [0, 14], [14, 16], [0, 15], [15, 17]]

    # _paf_keys encodes the structure of the part affinity fields (pafs)
    # each element is a 2-tuple of pairs, where the first pair is a pair of
    # landmarks which have a connection in the pafs and the second pair denotes
    # the channels in the paf array containing the vector field for this
    # connection
    _paf_keys = [((1, 2), (12, 13)), ((1, 5), (20, 21)), ((2, 3), (14, 15)),
                 ((3, 4), (16, 17)), ((5, 6), (22, 23)), ((6, 7), (24, 25)),
                 ((1, 8), (0, 1)), ((8, 9), (2, 3)), ((9, 10), (4, 5)),
                 ((1, 11), (6, 7)), ((11, 12), (8, 9)), ((12, 13), (10, 11)),
                 ((1, 0), (28, 29)), ((0, 14), (30, 31)), ((14, 16), (34, 35)),
                 ((0, 15), (32, 33)), ((15, 17), (36, 37)),
                 ((2, 17), (18, 19)), ((5, 16), (26, 27))]

    # Possible options for tuning the performance of the estimator
    # See poseestimator.PoseEstimator._general_options for a description
    # All models automatically have options defined in
    # poseestimator._general_options options (check there for which these are)
    _specific_options = {
        "num_paf_samples": {
            "max_val":
            20,
            "divider":
            1,
            "default":
            10,
            "description":
            "Number of samples to take from the paf along a potential "
            "connection."
        },
        "min_paf_score_th": {
            "max_val":
            10,
            "divider":
            10,
            "default":
            2,
            "description":
            "Minimal paf value for a paf sample point to be considered 'good'."
        },
        "paf_sample_th": {
            "max_val":
            10,
            "divider":
            10,
            "default":
            4,
            "description":
            "Percentage of paf samples that need to be good for the "
            "connection to be accepted."
        }
    }

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
        # heatmaps and pafs are downsampled by a factor of 8 from the input
        self._output_shape = (model_config["input_size"][1] // 8,
                              model_config["input_size"][0] // 8)

    def _get_keypoints(self, heatmaps):
        """
        Parse the heatmaps to obtain all detected keypoints.

        Also groups detected keypoints by which landmark they are.

        Parameters
        ----------
        heatmaps : numpy array
            heatmaps as retrieved from the network output

        Returns
        -------
        keypoints, landmarkwise_keypoints
            keypoints is a numpy array of shape (k,3), where k is the number of
            detected keypoints. Each keypoint consist of (x,y,confidence).
            landmarkwise_keypoints is a list of n=self._num_keypoints lists.
            Each of the n lists contains the ids into the keypoint array that
            are keypoints of this landmark type.
        """
        keypoints = np.zeros((0, 3), dtype=np.float32)
        landmarkwise_keypoints = [[] for i in range(self._num_keypoints)]
        for keypoint_id in range(self._num_keypoints):
            # scale heatmap back up to frame size for heatmap subpixel accuracy
            scaled_hm = cv2.resize(heatmaps[0, keypoint_id], self._input_shape)
            smooth_hm = cv2.GaussianBlur(scaled_hm, (3, 3), 0, 0)
            mask = np.uint8(smooth_hm > self._detection_threshold)

            contours = None
            try:  # OpenCV4.x
                contours, _ = cv2.findContours(mask, cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)
            except ValueError:  # OpenCV3.x
                _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE,
                                                  cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                blob_mask = np.zeros(mask.shape)
                blob_mask = cv2.fillConvexPoly(blob_mask, cnt, 1)
                masked_hm = smooth_hm * blob_mask
                _, maxVal, _, maxLoc = cv2.minMaxLoc(masked_hm)
                keypoints = np.append(
                    keypoints, [maxLoc + (scaled_hm[maxLoc[1], maxLoc[0]], )],
                    axis=0)
                landmarkwise_keypoints[keypoint_id].append(len(keypoints) - 1)

        return keypoints, landmarkwise_keypoints

    def _get_pairs(self, pafs, keypoints, landmarkwise_keypoints):
        """
        Parses the pafs for detected connections between keypoints.

        Parameters
        ----------
        pafs : numpy array
            Part affinity fields as output by the network.
        keypoints : numpy array
            Array of detected keypoints of shape (k,3) as output by
            _get_keypoints.
        landmarkwise_keypoints: list
            List of list of ids into the keypoints array, grouped by landmark
            type, as output by _get_keypoints

        Returns
        -------
        pairs: list of lists of 3 tuples
            The outer list contains one list for each connection that is
            encoded in the pafs. Each of those lists contains 3-tuples with the
            first two elements the indices into the keypoint array of the
            connected keypoints and the third element the average paf
            confidence score.
        """
        pairs = []
        for connection, paf_ids in self._paf_keys:
            # connection and paf_ids are tuples containing the landmarks and
            # the corresponding paf coordinates for a connection between
            # landmarks
            paf_a = pafs[0, paf_ids[0], :, :]
            paf_b = pafs[0, paf_ids[1], :, :]
            paf_a = cv2.resize(paf_a, self._input_shape)
            paf_b = cv2.resize(paf_b, self._input_shape)
            cand_a = landmarkwise_keypoints[connection[0]]
            cand_b = landmarkwise_keypoints[connection[1]]

            # only search for connections with candidates for both ends
            if (len(cand_a) != 0 and len(cand_b) != 0):
                pairs_found = np.zeros((0, 3))
                for i in range(len(cand_a)):
                    max_j = -1
                    max_score = -1
                    for j in range(len(cand_b)):
                        # paf values are scored by dotting with the unit vector
                        # from A towards B
                        unit_a2b = np.subtract(keypoints[cand_b[j]][:2],
                                               keypoints[cand_a[i]][:2])
                        norm = np.linalg.norm(unit_a2b)
                        if norm:
                            unit_a2b = unit_a2b / norm
                        else:
                            # points were identical, try next candidate for B
                            continue

                        # generate self._num_paf_samples (x,y) pairs along the
                        # line from A to B and sample the paf at those points
                        sample_coords = list(
                            zip(
                                np.linspace(keypoints[cand_a[i]][0],
                                            keypoints[cand_b[j]][0],
                                            num=self._num_paf_samples),
                                np.linspace(keypoints[cand_a[i]][1],
                                            keypoints[cand_b[j]][1],
                                            num=self._num_paf_samples)))
                        paf_samples = []
                        for k in range(len(sample_coords)):
                            paf_samples.append([
                                paf_a[int(round(sample_coords[k][1])),
                                      int(round(sample_coords[k][0]))],
                                paf_b[int(round(sample_coords[k][1])),
                                      int(round(sample_coords[k][0]))]
                            ])
                        # score all paf samples against the vector connecting
                        # the current landmarks
                        paf_scores = np.dot(paf_samples, unit_a2b)

                        # only accept a connection if the number of good paf
                        # scores (scores above self._min_paf_score_th) is
                        # bigger than self._paf_sample_th and for all Bs only
                        # keep the connection with the highest average score
                        good_paf_scores = len(
                            np.where(paf_scores > self._min_paf_score_th)[0])
                        if (good_paf_scores / self._num_paf_samples >
                                self._paf_sample_th):
                            avg_paf_score = sum(paf_scores) / len(paf_scores)
                            if avg_paf_score > max_score:
                                max_j = j
                                max_score = avg_paf_score
                    if max_j >= 0:
                        # pairs to contain connections for instances of A
                        # to their respective corrsponding B (if exists)
                        # each connection is of the form
                        # (keypoint_id_A, keypoint_id_B, avg_paf_score)
                        pairs_found = np.append(
                            pairs_found,
                            [[cand_a[i], cand_b[max_j], max_score]],
                            axis=0)

                pairs.append(pairs_found)
            else:
                pairs.append(None)
        return pairs

    def _get_personwise_keypoints(self, pairs, keypoints):
        """
        Groups the keypoints into people using the detected connections.

        Takes all connections extracted from the pafs and pieces them together
        to form individual persons.

        Parameters
        ----------
        pairs : list
            Pairs detected with the pafs, as output by _get_pairs
        keypoints : numpy array
            Array of detected keypoints of shape (k,3) as output by
            _get_keypoints.

        Returns
        -------
        personwise_keypoints : numpy array
            Numpy array of shape (n, self._num_keypoints, 3) where n is the
            number of detected people. For each person contains each
            landmark as (x,y,confidence), if a landmark was not detected all 3
            values will be zero.
        """
        # Array for final results of shape [person_id, landmark_id, (x,y,conf)]
        personwise_keypoints = np.zeros((0, self._num_keypoints, 3))
        # Array to just hold detection ids for each person for easy matching
        personwise_keypoint_ids = []
        for connection_id in range(len(self._paf_keys)):
            if pairs[connection_id] is not None:
                # Grab the types of landmarks we are connecting here
                start_landmark, end_landmark = self._paf_keys[connection_id][0]
                for i in range(len(pairs[connection_id])):
                    # Grab ids into the array of detected keypoints for the
                    # current connection
                    start_keypoint, end_keypoint = \
                        pairs[connection_id][i, :2].astype(int)

                    # Find if the startpoint already belongs to a person
                    person_id = -1
                    for j in range(len(personwise_keypoint_ids)):
                        if (personwise_keypoint_ids[j][start_landmark] ==
                                start_keypoint):
                            person_id = j
                            break

                    # Create a new person with the start point if no match
                    # found as long as we are not within the last two potential
                    # connections (they are invisible so should not create a
                    # new person)
                    if (person_id == -1
                            and connection_id < len(self._paf_keys) - 2):
                        # temp array for matching
                        personwise_keypoint_ids.append([-1] *
                                                       self._num_keypoints)
                        personwise_keypoint_ids[-1][start_landmark] = \
                            start_keypoint
                        # results array
                        personwise_keypoints = np.append(
                            personwise_keypoints,
                            np.zeros((1, personwise_keypoints.shape[1], 3)),
                            axis=0)
                        personwise_keypoints[-1][start_landmark] \
                            = keypoints[start_keypoint]
                        # set id to new persons id
                        person_id = len(personwise_keypoints) - 1
                    # If a person was found or created update array with the
                    # endpoint of the connection
                    if person_id >= 0:
                        # temp array for matching
                        personwise_keypoint_ids[person_id][end_landmark] \
                            = end_keypoint
                        # results array
                        personwise_keypoints[person_id][end_landmark] \
                            = keypoints[end_keypoint]
        return personwise_keypoints

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
        if len(outputs) == 1:
            heatmaps = outputs[0][:, :self._num_keypoints + 1]
            pafs = outputs[0][:, self._num_keypoints + 1:]
        else:
            heatmaps = outputs[0]
            pafs = outputs[1]

        keypoints, landmarkwise_keypoints = self._get_keypoints(heatmaps)
        # Get lists of detected keypoint connections from pafs
        pairs = self._get_pairs(pafs, keypoints, landmarkwise_keypoints)
        # Assemble the keypoints and connections into people
        personwise_keypoints = self._get_personwise_keypoints(pairs, keypoints)
        return personwise_keypoints
