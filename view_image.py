import argparse
import os

import cv2
import depthai as dai

from utils import Timer, SliderWindow
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

    parser.add_argument('-i', '--image', type=str, help="The image to process")
    return vars(parser.parse_args())


def main(args):
    """
    Main programm loop.

    Parameters
    ----------
    args : command line arguments parsed by parse_arguments
    """
    # Set up PoseEstimator, pipeline, window with sliders for PoseEstimator
    # options and load image
    if args["model"] not in model_list:
        raise ValueError("Unknown model '{}'".format(args["model"]))
    model_config = model_list[args["model"]]
    pose_estimator = get_poseestimator(model_config, **args)

    with dai.Device(create_pipeline(model_config, camera=False,
                                    **args)) as device:
        device.startPipeline()

        pose_in_queue = device.getInputQueue("pose_in")
        pose_queue = device.getOutputQueue("pose")

        if not os.path.exists(args["image"]):
            raise ValueError("Image '{}' does not exist.".format(
                args["image"]))
        print("Loading image", args["image"])
        image = cv2.imread(args["image"])

        window = SliderWindow("preview")
        window.add_poseestimator_options(pose_estimator, args)

        # Start main loop
        frame = None
        keypoints = None
        raw_output = None
        redraw = True
        timer = Timer("inference", "decode")
        while True:
            # Check for and handle slider changes, redraw if there was a change
            slider_changes = window.get_changes()
            for option_name, value in slider_changes.items():
                pose_estimator.set_option(option_name, value)
                redraw = True

            # On the first iteration pass the image to the NN for inference
            # Raw results are kept after so changes in PoseEstimator options
            # only require decoding the results again, not another inference
            if frame is None:
                frame = image.copy()
                # Create DepthAI ImgFrame object to pass to the camera
                input_frame = pose_estimator.get_input_frame(frame)
                frame_nn = dai.ImgFrame()
                frame_nn.setSequenceNum(0)
                frame_nn.setWidth(input_frame.shape[2])
                frame_nn.setHeight(input_frame.shape[1])
                frame_nn.setFrame(input_frame)
                timer.start_timer("inference")
                pose_in_queue.send(frame_nn)

            # Store the raw results once available
            if pose_queue.has():
                raw_output = pose_queue.get()
                timer.stop_timer("inference")

            # Once we've got the raw output and again whenever an option
            # changes we need to decode and draw
            if redraw and raw_output is not None:
                # keep a clean copy of the image for redrawing
                frame = image.copy()
                timer.start_timer("decode")
                keypoints = pose_estimator.get_pose_data(raw_output)
                timer.stop_timer("decode")
                pose_estimator.draw_results(keypoints, frame)
                redraw = False

            cv2.imshow("preview", frame)

            if cv2.waitKey(1) == ord("q"):
                break
        cv2.destroyAllWindows()
        timer.print_times()


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
