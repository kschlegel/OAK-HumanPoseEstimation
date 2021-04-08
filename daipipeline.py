import os
import json
import re

import depthai as dai

MODELS_FOLDER = "models"


def add_pipeline_args(parser, model_list):
    """
    Adds command line args for configuring the pipeline.

    Adds options for selecting the model, shaves and inference threads.

    Parameters
    ----------
    parser : ArgumentParser object
    model_list : list
        The list of all models obtained from get_model_list to populate the
        choices list for the model parameter.
    """
    parser.add_argument('-m',
                        '--model',
                        type=str,
                        choices=list(model_list.keys()),
                        help="The model to run")
    parser.add_argument('-sh',
                        '--shaves',
                        type=int,
                        help="Number of shaves to use (model must have been "
                        "compiled for this number) Defaults to 8 when running "
                        "on an image/video, 6 when running on camera.")
    parser.add_argument('--num_inference_threads',
                        type=int,
                        help="Optionally set the number of inference threads "
                        "for the NN.")
    parser.add_argument('--nce_per_inference_thread',
                        type=int,
                        help="Optionally set the number of NCEs to use per "
                        "inference thread.")


def get_model_list():
    """
    Reads the configurations of all available models from file.

    The 'models.json' file contains the config dictionaries for all included
    models.

    Returns
    -------
    model_list : dict
        Dictionary of all model configurations
    """
    with open(os.path.join(MODELS_FOLDER, "models.json"), "r") as model_file:
        model_list = json.load(model_file)
    return model_list


def create_pipeline(model_config,
                    camera,
                    sync=False,
                    passthrough=False,
                    **kwargs):
    """
    Create the depthai pipeline.

    Creates a pipeline feeding either the oak rgb camera preview stream or an
    xlink input stream into a neural net running a chosen pose estimation
    model and returns the results via a 'pose' xlink output stream.
    If the camera is used a 'preview' output stream is provided to display the
    results, if an xlink input is used no frame is returned as the host already
    has it.

    Parameters
    ----------
    model_config : dict
        dictionary with model parameters such as .blob-file and input size.
    camera : bool
        Boolean flag controlling whether to use the camera or an xlink input
        stream.
    sync : bool, optional (default is False)
        If False pose NN input queue is set to non-blocking so that frames can
        flow to the host at full rate. If True que is set to blocking so that
        any frame arriving at the host has been run through the NN.
    passthrough : bool, optional (default is False)
        If True create an output queue for the passthrough channel of the pose
        NN to receive the processed frames.

    Returns
    -------
    depthai.pipeline object
    """
    if "shaves" not in kwargs or kwargs["shaves"] is None:
        if camera:
            kwargs["shaves"] = 6
        else:
            kwargs["shaves"] = 8
    model_config["shaves"] = kwargs["shaves"]

    pipeline = dai.Pipeline()

    # Extract OpenVino version from .blob filename and set it in DepthAI
    # This is to be on the save side as sometime OpenVino updates can break
    # compatibility between versions and depthai will keep bumping to use new
    # versions. Reading OpenVino version from blob is currently not possible
    # from Intel/OpenVino side
    ov_version = re.search(r"ov(\d{4}\.\d)", model_config["blob"])
    if ov_version is not None:
        ov_version = ov_version.group(1)
        print("Selecting OpenVino version", ov_version)
        ov_version = ov_version.replace(".", "_")
        pipeline.setOpenVINOVersion(
            version=getattr(dai.OpenVINO.Version, "VERSION_" + ov_version))
    # Create camera with frame type and preview size matching the NN input
    if camera:
        cam_rgb = pipeline.createColorCamera()
        cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
        cam_rgb.setResolution(
            dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        if model_config["color_order"] == "BGR":
            cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        elif model_config["color_order"] == "RGB":
            cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        else:
            raise ValueError("Invalid color order: {}".format(
                model_config["color_order"]))
        cam_rgb.setPreviewSize(*model_config["input_size"])
        cam_rgb.setInterleaved(False)
        cam_rgb.setFps(30)

        preview_out = pipeline.createXLinkOut()
        preview_out.setStreamName("preview")
        cam_rgb.preview.link(preview_out.input)

    # Create pose estimation network
    pose_nn = pipeline.createNeuralNetwork()
    print("Loading model", model_config["model"], "from",
          model_config["source"], "for", model_config["shaves"], "shaves")
    print("(Link: {})".format(model_config["url"]))
    print("This model is from the", model_config["decoder"],
          "family and takes an input of size", model_config["input_size"])
    if model_config["feature_extractor"] is not None:
        print("A", model_config["feature_extractor"],
              "is used as feature extractor")

    model_blob = (model_config["blob"] + "_sh" + str(model_config["shaves"]) +
                  ".blob")
    path = os.path.join(MODELS_FOLDER, model_blob)
    if not os.path.exists(path):
        raise ValueError("Blob file '{}' does not exist.".format(path))
    print("Blob file:", path)
    pose_nn.setBlobPath(path)
    # Allow optional changing of inference thread settings to see if there is
    # potential for improving performance
    if ("num_inference_threads" in kwargs
            and kwargs["num_inference_threads"] is not None):
        pose_nn.setNumInferenceThreads(kwargs["num_inference_threads"])
    if ("nce_per_inference_thread" in kwargs
            and kwargs["nce_per_inference_thread"] is not None):
        pose_nn.setNumNCEPerInferenceThread(kwargs["nce_per_inference_thread"])

    if not sync:
        # Set NN input to not blocking to allow the video stream to flow at
        # full fps to host
        pose_nn.input.setQueueSize(1)
        pose_nn.input.setBlocking(False)
    if camera:
        cam_rgb.preview.link(pose_nn.input)
    else:
        pose_in = pipeline.createXLinkIn()
        pose_in.setStreamName("pose_in")
        pose_in.out.link(pose_nn.input)

    if passthrough:
        # Include passthrough to show 'what the NN sees'
        passthrough_out = pipeline.createXLinkOut()
        passthrough_out.setStreamName("passthrough")
        pose_nn.passthrough.link(passthrough_out.input)

    pose_out = pipeline.createXLinkOut()
    pose_out.setStreamName("pose")
    pose_nn.out.link(pose_out.input)

    return pipeline
