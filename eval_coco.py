import argparse
import os
import json

from tqdm import trange
import numpy as np
import depthai as dai

from utils import Timer
from utils.coco import COCOData, add_coco_args
from daipipeline import get_model_list, add_pipeline_args, create_pipeline
from poseestimators import get_poseestimator, add_poseestimator_args

model_list = get_model_list()


def parse_arguments():
    """
    Define the command line arguments for choosing a model etc.

    Returns
    -------
    args object
    """
    parser = argparse.ArgumentParser(description='')
    add_pipeline_args(parser, model_list)
    add_poseestimator_args(parser)
    add_coco_args(parser)
    return vars(parser.parse_args())


def main(args):
    """
    Main programm loop.

    Parameters
    ----------
    args : command line arguments parsed by parse_arguments
    """
    # Set up PoseEstimator and pipeline  and load the dataset
    if args["model"] not in model_list:
        raise ValueError("Unknown model '{}'".format(args["model"]))
    model_config = model_list[args["model"]]
    pose_estimator = get_poseestimator(model_config, **args)

    with dai.Device(
            create_pipeline(model_config, camera=False, sync=True,
                            **args)) as device:
        device.startPipeline()

        pose_in_queue = device.getInputQueue("pose_in")
        pose_queue = device.getOutputQueue("pose")

        # Load coco keypoint annotations
        coco_data = COCOData(**args)
        # The keypoint selector allows to subset and re-order the predicted
        # keypoints to align with the annotation format of the COCO dataset
        keypoint_selector = coco_data.get_keypoint_selector(
            pose_estimator.landmarks)
        results_filename = "results_{model}_{conf}.json".format(
            model=args["model"], conf=args["detection_threshold"])

        # Re-use saved results if available. Iterate over dataset if not.
        if not os.path.exists(results_filename):
            timer = Timer("inference")
            results = []
            for img_id in trange(len(coco_data)):
                img = coco_data.get_image(img_id)

                input_frame = pose_estimator.get_input_frame(img)
                frame_nn = dai.ImgFrame()
                frame_nn.setSequenceNum(img_id)
                frame_nn.setWidth(input_frame.shape[2])
                frame_nn.setHeight(input_frame.shape[1])
                frame_nn.setFrame(input_frame)
                timer.start_timer("inference")
                pose_in_queue.send(frame_nn)

                raw_output = pose_queue.get()
                timer.stop_timer("inference")
                pred_keypoints = pose_estimator.get_pose_data(raw_output)

                # Convert each individual person into output format expected by
                # COCO evaluation tools
                for i in range(pred_keypoints.shape[0]):
                    score = pred_keypoints[i, :, 2]
                    score = np.sum(score) / np.count_nonzero(score)
                    pred_keypoints[i, :, 2] = 1
                    keypoints = np.around(pred_keypoints[i])
                    keypoints = keypoints[keypoint_selector]
                    results.append({
                        "image_id":
                        coco_data.get_coco_imageid(img_id),
                        "category_id":
                        1,
                        "keypoints":
                        keypoints.flatten().tolist(),
                        "score":
                        score
                    })

            with open(results_filename, "w") as results_file:
                json.dump(results, results_file)
            timer.print_times()

        coco_data.evaluate_results(results_filename)


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
