import argparse

import depthai as dai
import cv2

from utils import SliderWindow
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
    # Set up PoseEstimator, pipeline, window with sliders for PoseEstimator
    # options and load the dataset
    if args["model"] not in model_list:
        raise ValueError("Unknown model '{}'".format(args["model"]))
    model_config = model_list[args["model"]]
    pose_estimator = get_poseestimator(model_config, **args)

    with dai.Device(create_pipeline(model_config, camera=False,
                                    **args)) as device:
        device.startPipeline()

        pose_in_queue = device.getInputQueue("pose_in")
        pose_queue = device.getOutputQueue("pose")

        result_window = SliderWindow("Result")
        result_window.add_poseestimator_options(pose_estimator, args)
        cv2.namedWindow("Original")

        # load coco keypoint annotations
        coco_data = COCOData(**args)

        # Start main loop
        last_img_id = -1
        cur_img_id = 0
        original_frame = None
        results_frame = None
        raw_output = None
        redraw = True
        while True:
            # Check for and handle slider changes, redraw if there was a change
            slider_changes = result_window.get_changes()
            for option_name, value in slider_changes.items():
                pose_estimator.set_option(option_name, value)
                redraw = True

            # When new image was selected process it
            if last_img_id != cur_img_id:
                img, gt_keypoints = coco_data[cur_img_id]

                # Send image off for inference
                input_frame = pose_estimator.get_input_frame(img)
                frame_nn = dai.ImgFrame()
                frame_nn.setSequenceNum(0)
                frame_nn.setWidth(input_frame.shape[2])
                frame_nn.setHeight(input_frame.shape[1])
                frame_nn.setFrame(input_frame)
                pose_in_queue.send(frame_nn)

                # Draw ground truth annotations
                original_frame = img.copy()
                coco_data.draw(gt_keypoints, original_frame)
                cv2.rectangle(original_frame, (0, 0), (150, 25), (0, 0, 0), -1)
                cv2.putText(original_frame,
                            "#people: {}".format(len(gt_keypoints)), (2, 15),
                            cv2.FONT_HERSHEY_TRIPLEX,
                            0.4,
                            color=(255, 255, 255))

                # Discard previous results
                results_frame = None
                raw_output = None
                last_img_id = cur_img_id

            # Store the raw results once available
            if pose_queue.has():
                raw_output = pose_queue.get()

            # Once we've got the raw output and again whenever an option
            # changes we need to decode and draw
            if redraw and raw_output is not None:
                results_frame = img.copy()
                pred_keypoints = pose_estimator.get_pose_data(raw_output)
                pose_estimator.draw_results(pred_keypoints, results_frame)
                cv2.rectangle(results_frame, (0, 0), (150, 25), (0, 0, 0), -1)
                cv2.putText(results_frame,
                            "#people: {}".format(len(pred_keypoints)), (2, 15),
                            cv2.FONT_HERSHEY_TRIPLEX,
                            0.4,
                            color=(255, 255, 255))

            cv2.imshow("Original", original_frame)
            if results_frame is not None:
                cv2.imshow("Result", results_frame)

            c = cv2.waitKey(1)
            if c == ord("q"):
                break
            elif c == ord("n"):
                if cur_img_id < len(coco_data) - 1:
                    cur_img_id += 1
            elif c == ord("p"):
                if cur_img_id > 0:
                    cur_img_id -= 1


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
