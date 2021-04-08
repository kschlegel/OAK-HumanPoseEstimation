import argparse
import os
import time

import cv2
import depthai as dai

from utils import FPS, Timer, SliderWindow
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
    parser.add_argument('-v',
                        '--video',
                        type=str,
                        help="If given, run on the video rather than oak "
                        "camera.")
    return vars(parser.parse_args())


def main(args):
    """
    Main programm loop.

    Parameters
    ----------
    args : command line arguments parsed by parse_arguments
    """
    # Setup PoseEstimator, pipeline, windows with sliders for PoseEstimator
    # options and load video if running on local video file
    camera = args["video"] is None
    if args["model"] not in model_list:
        raise ValueError("Unknown model '{}'".format(args["model"]))
    model_config = model_list[args["model"]]
    pose_estimator = get_poseestimator(model_config, **args)

    with dai.Device(
            create_pipeline(model_config, camera, passthrough=True,
                            **args)) as device:
        device.startPipeline()

        if camera:
            preview_queue = device.getOutputQueue("preview",
                                                  maxSize=4,
                                                  blocking=False)
        else:
            pose_in_queue = device.getInputQueue("pose_in")
        pose_queue = device.getOutputQueue("pose")
        passthrough_queue = device.getOutputQueue("passthrough")

        # Load video if given in command line and set the variables used below
        # to control FPS and looping of the video
        if not camera:
            if not os.path.exists(args["video"]):
                raise ValueError("Video '{}' does not exist.".format(
                    args["video"]))
            print("Loading video", args["video"])
            video = cv2.VideoCapture(args["video"])
            frame_interval = 1 / video.get(cv2.CAP_PROP_FPS)
            last_frame_time = 0
            frame_id = 0
        else:
            print("Running on OAK camera preview stream")

        # Create windows for the original video and the video of frames from
        # the NN passthrough. The window for the original video gets all the
        # option sliders to change pose estimator config
        video_window_name = "Original Video"
        passthrough_window_name = "Processed Video"
        video_window = SliderWindow(video_window_name)
        cv2.namedWindow(passthrough_window_name)
        video_window.add_poseestimator_options(pose_estimator, args)

        # Start main loop
        frame = None
        keypoints = None
        fps = FPS("Video", "NN", interval=0.1)
        timer = Timer("inference", "decode")
        while True:
            # Check for and handle slider changes
            slider_changes = video_window.get_changes()
            for option_name, value in slider_changes.items():
                pose_estimator.set_option(option_name, value)

            fps.start_frame()
            # Get next video frame (and submit for processing if local video)
            if camera:
                frame = preview_queue.get().getCvFrame()
                fps.count("Video")
            else:
                frame_time = time.perf_counter()
                # Only grab next frame from file at certain intervals to
                # roughly preserve its original FPS
                if frame_time - last_frame_time > frame_interval:
                    if video.grab():
                        __, frame = video.retrieve()
                        fps.count("Video")
                        last_frame_time = frame_time
                        # Create DepthAI ImgFrame object to pass to the
                        # camera
                        input_frame = pose_estimator.get_input_frame(frame)
                        frame_nn = dai.ImgFrame()
                        frame_nn.setSequenceNum(frame_id)
                        frame_nn.setWidth(input_frame.shape[2])
                        frame_nn.setHeight(input_frame.shape[1])
                        frame_nn.setType(dai.RawImgFrame.Type.BGR888p)
                        frame_nn.setFrame(input_frame)
                        pose_in_queue.send(frame_nn)
                        frame_id += 1
                    else:
                        frame_id = 0
                        video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)

            # Process pose data whenever a new packet arrives
            if pose_queue.has():
                raw_output = pose_queue.get()
                timer.start_timer("decode")
                keypoints = pose_estimator.get_pose_data(raw_output)
                timer.stop_timer("decode")
                fps.count("NN")
                # When keypoints are available we should also have a
                # passthrough frame to process and display. Make sure it is
                # availabe to avoid suprises.
                if passthrough_queue.has():
                    passthrough = passthrough_queue.get()
                    timer.frame_time("inference", passthrough)
                    passthrough_frame = passthrough.getCvFrame()
                    passthrough_frame = pose_estimator.get_original_frame(
                        passthrough_frame)
                    pose_estimator.draw_results(keypoints, passthrough_frame)
                    cv2.imshow(passthrough_window_name, passthrough_frame)

            # Annotate current video frame with keypoints and FPS
            if keypoints is not None:
                pose_estimator.draw_results(keypoints, frame)
            fps.update()
            fps.display(frame)

            cv2.imshow(video_window_name, frame)

            if cv2.waitKey(1) == ord("q"):
                break
        fps.print_totals()
        timer.print_times()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
