import importlib
import sys

from .poseestimator import PoseEstimator
from .openpose import OpenPose
from .efficienthrnet import EfficientHRNet
from .convolutionalposemachine import ConvolutionalPoseMachine
from .posenet import PoseNet


def add_poseestimator_args(parser):
    """
    Adds command line args for configuring the pose estimator object.

    Iterates over the _general_options array in PoseEstimator for options
    common to all PoseEstimator objects and the _specific_options arrays of the
    subclasses which have been added (Note that this needs to be added manually
    when a new PoseEstimator subclass is added).

    Parameters
    ----------
    parser : ArgumentParser object
    """
    parser.add_argument('-d',
                        '--decoder',
                        type=str,
                        help="Optionally select a decoder other different "
                        "from the models default decoder.")

    help_str = "{text} (default is {default})"
    for option_name, option in PoseEstimator.get_general_options():
        parser.add_argument('--' + option_name,
                            type=int,
                            default=option["default"],
                            help=help_str.format(text=option["description"],
                                                 default=option["default"]))
    # When adding a new PoseEstimator subclass add it to this list to
    # automatically register its hyperparameters as command line arguments
    for estimator in (OpenPose, PoseNet):
        estimator_name = str(estimator)
        estimator_name = estimator_name[estimator_name.rindex(".") + 1:-2]
        for option_name, option in estimator.get_specific_options():
            parser.add_argument(
                '--' + option_name,
                type=int,
                default=option["default"],
                help=help_str.format(text=option["description"],
                                     default=option["default"]) +
                " (only used for {})".format(estimator_name))


def get_poseestimator(model_config, **kwargs):
    """
    Get the pose estimator object to decode the results.

    If a decoder has been selected by command line argument that one will be
    loaded, otherwise the default decoder specified in the model config will be
    used.

    Default decoders are registered, i.e. imported above, but other decoders
    can also be handled. If a decoder is requested which has not been imported
    (of name DecoderName say) it tries to import the class DecoderName from the
    package poseestimators.decodername. If the class does not exist an error is
    thrown.

    Command line args will be passed on to the pose estimator object,
    regardless whether it had been registered before or gets imported here.
    Only registered classes have the command line arguments registered
    automatically (if also manually in the add_poseestimator_args function).

    Parameters
    ----------
    model_config : dict
        Dictionary containing the configuration of the selected model.
    kwargs:
        Command line arguments to determine selected decoder and
        hyperparameters.

    Returns
    -------
    pose_estimator : PoseEstimator object
        Class object to handle the inouts and outputs for the chosen model
    """
    if kwargs["decoder"] is not None:
        decoder_name = kwargs["decoder"]
    else:
        decoder_name = model_config["decoder"]

    module_name = "poseestimators." + decoder_name.lower()
    if module_name in sys.modules:
        estimator_module = sys.modules[module_name]
    else:
        if importlib.util.find_spec(module_name) is not None:
            estimator_module = importlib.import_module(module_name)
        else:
            raise ValueError("Invalid decoder: {}".format(
                model_config["decoder"]))

    PoseEstimatorClass = getattr(estimator_module, decoder_name)
    pose_estimator = PoseEstimatorClass(model_config, **kwargs)

    return pose_estimator
