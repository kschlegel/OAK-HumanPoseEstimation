"""
Decode PoseNet type networks. Models come with MobileNetV1 or ResNet50
backbones.
Output stride can be 8, 16 or 32 for MobileNet or 16, 32 for ResNet. The stride
information is parsed from the blob filename, e.g. str16 indicates a stride of
16.
The input layer size is a default of 641x481. Models with smaller input sizes
are indicated in the blob filename, e.g. x07 indicates approx 0.7 of the
original resolution -> 449x337.

For more information on the models, see
    https://github.com/tensorflow/tfjs-models/blob/master/posenet/README.md
Most of the decoding code came from
    https://github.com/atomicbits/posenet-python

This file was originally written by @rychardeh
Mostly cosmetic edits by @kschlegel
"""

import numpy as np
import scipy.ndimage as ndi

from .poseestimator import PoseEstimator


class PoseNet(PoseEstimator):
    """
    Decode PoseNet type networks.
    """
    # The order of the landmarks in the models output
    landmarks = [
        "nose", "left eye", "right eye", "left ear", "right ear",
        "left shoulder", "right shoulder", "left elbow", "right elbow",
        "left wrist", "right wrist", "left hip", "right hip", "left knee",
        "right knee", "left ankle", "right ankle"
    ]

    # The connections to draw, each element is a pair of indices into above
    # landmarks array
    connections = [[0, 1], [1, 3], [0, 2], [2, 4], [5, 6], [5, 7], [7, 9],
                   [6, 8], [8, 10], [5, 11], [6, 12], [11, 12], [11, 13],
                   [13, 15], [12, 14], [14, 16]]

    _LOCAL_MAXIMUM_RADIUS = 1
    # Complete set of connected parts including those in head.
    _parent_child_tuples = [(0, 1), (1, 3), (0, 2), (2, 4), (0, 5), (5, 7),
                            (7, 9), (5, 11), (11, 13), (13, 15), (0, 6),
                            (6, 8), (8, 10), (6, 12), (12, 14), (14, 16)]

    # Possible options for tuning the performance of the estimator
    # See poseestimator.PoseEstimator._general_options for a description
    # All models automatically have options defined in
    # poseestimator._general_options options (check there for which these are)
    _specific_options = {
        "min_part_score": {
            "max_val": 100,
            "divider": 100,
            "default": 10,
            "description": "Minimal score for valid part location.",
        },
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

        stride_details = model_config["blob"].split("_")[-1]
        if stride_details.startswith("str"):
            self._stride = int(stride_details[3:])
        else:
            self._stride = 16

        self._output_shape = (
            (model_config["input_size"][1] // self._stride) + 1,
            (model_config["input_size"][0] // self._stride) + 1,
        )

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
            number of detected_keypoints. For each person contains each
            landmark as (x,y,confidence), if a landmark was not detected all 3
            values will be zero.
        """
        # Reorder output channels, drop batch channel
        outputs = [o[0].transpose(1, 2, 0) for o in outputs]

        # The models we are currently using do not have the Sigmoid operation
        # at the end of the graph for the heatmap, so we have to calculate that
        # manually.
        outputs[0] = 1 / (1 + np.exp(-outputs[0]))

        (pose_scores, keypoint_scores,
         keypoint_coords) = self._decode_multiple_poses(
             outputs[0],  # heatmap
             outputs[1],  # offsets
             outputs[2],  # displacement_fwd
             outputs[3],  # displacement_bwd,
             output_stride=self._stride,
             max_pose_detections=10,
             min_pose_score=self._detection_threshold,
         )

        # Array for final results of shape [person_id, landmark_id, (x,y,conf)]
        personwise_keypoints = np.zeros((0, self._num_keypoints, 3))

        for pose_index, score in enumerate(pose_scores):
            if score < self._detection_threshold:
                continue

            personwise_keypoints = np.append(
                personwise_keypoints,
                np.zeros((1, personwise_keypoints.shape[1], 3)),
                axis=0,
            )

            part_index = 0
            for ks, kc in zip(keypoint_scores[pose_index],
                              keypoint_coords[pose_index, :]):
                if ks >= self._min_part_score:
                    personwise_keypoints[-1][part_index] = [kc[1], kc[0], ks]
                part_index += 1

        return personwise_keypoints

    # ###############################################
    # The code below this point was taken from
    # https://github.com/atomicbits/posenet-python
    # ###############################################

    def _decode_multiple_poses(
        self,
        scores,
        offsets,
        displacements_fwd,
        displacements_bwd,
        output_stride,
        max_pose_detections=10,
        score_threshold=0.5,
        nms_radius=20,
        min_pose_score=0.5,
    ):
        pose_count = 0
        pose_scores = np.zeros(max_pose_detections)
        pose_keypoint_scores = np.zeros(
            (max_pose_detections, self._num_keypoints))
        pose_keypoint_coords = np.zeros(
            (max_pose_detections, self._num_keypoints, 2))

        squared_nms_radius = nms_radius**2

        scored_parts = self._build_part_with_score_fast(
            score_threshold, self._LOCAL_MAXIMUM_RADIUS, scores)
        scored_parts = sorted(scored_parts, key=lambda x: x[0], reverse=True)

        # change dimensions from (h, w, x) to (h, w, x//2, 2) to allow return
        # of complete coord array
        height = scores.shape[0]
        width = scores.shape[1]
        offsets = offsets.reshape(height, width, 2, -1).swapaxes(2, 3)
        displacements_fwd = displacements_fwd.reshape(height, width, 2,
                                                      -1).swapaxes(2, 3)
        displacements_bwd = displacements_bwd.reshape(height, width, 2,
                                                      -1).swapaxes(2, 3)

        for root_score, root_id, root_coord in scored_parts:
            root_image_coords = (
                root_coord * output_stride +
                offsets[root_coord[0], root_coord[1], root_id])

            if self._within_nms_radius_fast(
                    pose_keypoint_coords[:pose_count, root_id, :],
                    squared_nms_radius,
                    root_image_coords,
            ):
                continue

            keypoint_scores, keypoint_coords = self._decode_pose(
                root_score,
                root_id,
                root_image_coords,
                scores,
                offsets,
                output_stride,
                displacements_fwd,
                displacements_bwd,
            )

            pose_score = self._get_instance_score_fast(
                pose_keypoint_coords[:pose_count, :, :],
                squared_nms_radius,
                keypoint_scores,
                keypoint_coords,
            )

            # NOTE this isn't in the original implementation, but it appears
            # that by initially ordering by part scores, and having a max # of
            # detections, we can end up populating the returned poses with
            # lower scored poses than if we discard 'bad' ones and continue
            # (higher pose scores can still come later).
            # Set min_pose_score to 0. to revert to original behaviour
            if min_pose_score == 0.0 or pose_score >= min_pose_score:
                pose_scores[pose_count] = pose_score
                pose_keypoint_scores[pose_count, :] = keypoint_scores
                pose_keypoint_coords[pose_count, :, :] = keypoint_coords
                pose_count += 1

            if pose_count >= max_pose_detections:
                break

        return pose_scores, pose_keypoint_scores, pose_keypoint_coords

    def _build_part_with_score_fast(self, score_threshold, local_max_radius,
                                    scores):
        parts = []
        num_keypoints = scores.shape[2]
        lmd = 2 * local_max_radius + 1

        # NOTE it seems faster to iterate over the keypoints and perform
        # maximum_filter on each subarray vs doing the op on the full score
        # array with size=(lmd, lmd, 1)
        for keypoint_id in range(num_keypoints):
            kp_scores = scores[:, :, keypoint_id].copy()
            kp_scores[kp_scores < score_threshold] = 0.0
            max_vals = ndi.maximum_filter(kp_scores, size=lmd, mode="constant")
            max_loc = np.logical_and(kp_scores == max_vals, kp_scores > 0)
            max_loc_idx = max_loc.nonzero()
            for y, x in zip(*max_loc_idx):
                parts.append(
                    (scores[y, x, keypoint_id], keypoint_id, np.array((y, x))))

        return parts

    def _get_instance_score_fast(self, exist_pose_coords, squared_nms_radius,
                                 keypoint_scores, keypoint_coords):

        if exist_pose_coords.shape[0]:
            s = (np.sum((exist_pose_coords - keypoint_coords)**2, axis=2) >
                 squared_nms_radius)
            not_overlapped_scores = np.sum(keypoint_scores[np.all(s, axis=0)])
        else:
            not_overlapped_scores = np.sum(keypoint_scores)
        return not_overlapped_scores / len(keypoint_scores)

    def _within_nms_radius_fast(self, pose_coords, squared_nms_radius, point):
        if not pose_coords.shape[0]:
            return False
        return np.any(
            np.sum((pose_coords - point)**2, axis=1) <= squared_nms_radius)

    def _decode_pose(
        self,
        root_score,
        root_id,
        root_image_coord,
        scores,
        offsets,
        output_stride,
        displacements_fwd,
        displacements_bwd,
    ):
        num_parts = scores.shape[2]
        num_edges = len(self._parent_child_tuples)

        instance_keypoint_scores = np.zeros(num_parts)
        instance_keypoint_coords = np.zeros((num_parts, 2))
        instance_keypoint_scores[root_id] = root_score
        instance_keypoint_coords[root_id] = root_image_coord

        for edge in reversed(range(num_edges)):
            target_keypoint_id, source_keypoint_id = self._parent_child_tuples[
                edge]
            if (instance_keypoint_scores[source_keypoint_id] > 0.0
                    and instance_keypoint_scores[target_keypoint_id] == 0.0):
                score, coords = self._traverse_to_targ_keypoint(
                    edge,
                    instance_keypoint_coords[source_keypoint_id],
                    target_keypoint_id,
                    scores,
                    offsets,
                    output_stride,
                    displacements_bwd,
                )
                instance_keypoint_scores[target_keypoint_id] = score
                instance_keypoint_coords[target_keypoint_id] = coords

        for edge in range(num_edges):
            source_keypoint_id, target_keypoint_id = self._parent_child_tuples[
                edge]
            if (instance_keypoint_scores[source_keypoint_id] > 0.0
                    and instance_keypoint_scores[target_keypoint_id] == 0.0):
                score, coords = self._traverse_to_targ_keypoint(
                    edge,
                    instance_keypoint_coords[source_keypoint_id],
                    target_keypoint_id,
                    scores,
                    offsets,
                    output_stride,
                    displacements_fwd,
                )
                instance_keypoint_scores[target_keypoint_id] = score
                instance_keypoint_coords[target_keypoint_id] = coords

        return instance_keypoint_scores, instance_keypoint_coords

    def _traverse_to_targ_keypoint(
        self,
        edge_id,
        source_keypoint,
        target_keypoint_id,
        scores,
        offsets,
        output_stride,
        displacements,
    ):
        height = scores.shape[0]
        width = scores.shape[1]

        source_keypoint_indices = np.clip(
            np.round(source_keypoint / output_stride),
            a_min=0,
            a_max=[height - 1, width - 1],
        ).astype(np.int32)

        displaced_point = (source_keypoint +
                           displacements[source_keypoint_indices[0],
                                         source_keypoint_indices[1], edge_id])

        displaced_point_indices = np.clip(
            np.round(displaced_point / output_stride),
            a_min=0,
            a_max=[height - 1, width - 1],
        ).astype(np.int32)

        score = scores[displaced_point_indices[0], displaced_point_indices[1],
                       target_keypoint_id]

        image_coord = (
            displaced_point_indices * output_stride +
            offsets[displaced_point_indices[0], displaced_point_indices[1],
                    target_keypoint_id, ])

        return score, image_coord
