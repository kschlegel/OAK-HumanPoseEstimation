import os

import numpy as np
import cv2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def add_coco_args(parser):
    """
    Adds command line args for configuring the COCO dataset.

    Parameters
    ----------
    parser : ArgumentParser object
    """
    parser.add_argument('-ds',
                        '--dataset_path',
                        type=str,
                        default=".",
                        help="Path to COCO dataset (default is current dir)")
    parser.add_argument('--min_keypoints',
                        type=int,
                        default=0,
                        help="Minimal required amount of keypoints in the "
                        "ground truth annotations (some annotations have very "
                        "few keypoints and might want to be excluded)")


class COCOData:
    """
    Wraps the COCO api to prune bad examples (e.g. images of people without
    keypoint annotations) and provide easy access to the data.
    """
    # The order of the landmarks in the annotations
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

    def __init__(self, dataset_path, min_keypoints=0, **kwargs):
        """
        Loads and cleans the dataset structure.

        Parameters
        ----------
        dataset_path : str
            Path to the COCO dataset.
        min_keypoints : int, optional (default is 0)
            Minimal number of keypoints an annotation needs to contain for it
            to be considered a valid person worth including in the data. This
            is because some images in the dataset only have annotations of very
            few keypoints, e.g. a single wrist.
        kwargs:
            Dummy argument. Allows to directly pass the entire command line
            args without having to select the relevant ones.
        """
        if not os.path.exists(
                os.path.join(dataset_path, "annotations",
                             "person_keypoints_val2017.json")):
            raise ValueError("COCO dataset directory does not exist!")
        self._coco = COCO(
            os.path.join(dataset_path, "annotations",
                         "person_keypoints_val2017.json"))
        self._img_path = os.path.join(dataset_path, "images", "val2017")
        # Load only images of people (catId=1)
        img_ids = self._coco.getImgIds(catIds=1)

        self._min_keypoints = min_keypoints
        self._img_ids = []
        self._annotations = []
        # Some images contain people but no keypoints, drop those from the list
        for img_id in img_ids:
            annotation_ids = self._coco.getAnnIds(imgIds=img_id, iscrowd=False)
            annotations = self._coco.loadAnns(annotation_ids)

            num_persons = 0
            for i in range(len(annotations)):
                if annotations[i]["num_keypoints"] > self._min_keypoints:
                    num_persons += 1
                    break
            if num_persons > 0:
                self._img_ids.append(img_id)
        print("Images with more than", self._min_keypoints,
              "keypoint annotations:", len(self._img_ids))

    def __len__(self):
        """
        Returns the number of images with keypoint annotations.
        """
        return len(self._img_ids)

    def __getitem__(self, index):
        """
        Get the image data and personwise keypoint annotations.

        Parameters
        ----------
        index : int
            Local image id in [0, lenm(self)], not COCO img id

        Returns
        -------
        tuple : img, keypoints
            img is numpy array containing the RGB data (shape
            [height,width,channels])
            keypoints is numpy array containing personwise keypoints (shape
            [num_persons,landmark,(x,y,visibility)] )
        """
        return self.get_image(index), self.get_keypoints(index)

    def __iter__(self):
        """
        Iterate over the whole dataset.
        """
        for i in range(len(self)):
            yield self[i]

    def get_image(self, index):
        """
        Load the given image.

        Parameters
        ----------
        index : int
            Local image id in [0, lenm(self)], not COCO img id

        Returns
        -------
        img : numpy array of shape [height,width,channels]
              containing the RGB data
        """
        img_info = self._coco.loadImgs(self._img_ids[index])[0]
        return cv2.imread(os.path.join(self._img_path, img_info["file_name"]))

    def get_keypoints(self, index):
        """
        Get personwise keypoints for the given image.

        Parameters
        ----------
        index : int
            Local image id in [0, lenm(self)], not COCO img id

        Returns
        -------
        keypoints : numpy array of shape
                    [num_persons,landmark,(x,y,visibility)]
                    containing personwise keypoints
        """
        annotation_ids = self._coco.getAnnIds(imgIds=self._img_ids[index],
                                              iscrowd=False)
        annotations = self._coco.loadAnns(annotation_ids)
        # Stack all annotations which actually contain keypoints in a np.array
        keypoints = np.zeros((0, 17, 3))
        for i in range(len(annotations)):
            if annotations[i]["num_keypoints"] > self._min_keypoints:
                keypoints = np.append(
                    keypoints,
                    np.array(annotations[i]["keypoints"]).reshape(1, 17, 3),
                    axis=0)
        return keypoints

    def get_coco_imageid(self, index):
        """
        Get the original COCO image id for the given image.

        Parameters
        ----------
        index : int
            Local image id in [0, lenm(self)], not COCO img id
        """
        return self._img_ids[index]

    def get_keypoint_selector(self, landmarks):
        """
        Get the subset/reordering of keypoints to reformat estimator results.

        Different pose estimators may return different sets of landmarks and in
        different orders. This returns a list that can be used to index the
        numpy array of results to take the right subset and reorder it so that
        the results align with the COCO annotations to evaluate the results.

        Parameters
        ----------
        landmarks : list of strings
            List of landmarks output by the pose estimator
        """
        return [landmarks.index(landmark) for landmark in COCOData.landmarks]

    def evaluate_results(self, results_filename):
        """
        Run the COCO evaluation tool on the given results file.

        Parameters
        ----------
        results_filename : str
            Name of the file containing the pose estimation results.
        """
        coco_results = self._coco.loadRes(results_filename)

        coco_eval = COCOeval(self._coco, coco_results, "keypoints")
        coco_eval.params.imgIds = self._img_ids
        coco_eval.params.catIds = [1]
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    def draw(self, personwise_keypoints, frame):
        """
        Draw the COCO keypoint annotations onto the given frame.

        Skips keypoints that were not labelled (visibility_flag=0)

        Parameters
        ----------
        personwise_keypoints : numpy array
            Keypoint array of the form [n,self._num_keypoints, 3] with
            (x,y,visibility) information for each keypoint as returned by
            get_keypoints.
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
                # Confidence = 0 means not labelled
                if person[i][2] == 0:
                    continue
                cv2.circle(frame, tuple(person[i][0:2].astype(int)), 2,
                           point_colour, -1, cv2.LINE_AA)
            # Draw connections
            for connection in COCOData.connections:
                # Confidence = 0 means not labelled
                confidences = person[connection, 2]
                if 0 in confidences:
                    continue
                pt1, pt2 = person[connection, :2].astype(int)
                cv2.line(frame, tuple(pt1), tuple(pt2), line_colour, 1,
                         cv2.LINE_AA)
