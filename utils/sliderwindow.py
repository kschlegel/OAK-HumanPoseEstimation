import cv2


def _on_trackbar(val):
    """
    I'm handling trackbar changes manually by checking their current values, so
    just define a dummy function as callback.
    """
    pass


class SliderWindow:
    """
    Simple wrapper for creating windows with various trackbars.

    Creates a named OpenCV window and allows adding various trackbars to change
    various configuration parameters at runtime.
    """
    def __init__(self, window_name):
        """
        Parameters
        ----------
        window_name : str
            Identifier of the window to create
        """
        self._window_name = window_name
        cv2.namedWindow(self._window_name)
        self._slider_values = {}

    def add_slider(self, slider_name, initial_value, max_value):
        """
        Adds a single trackbar to the window.

        Parameters
        ----------
        slider_name : str
            Identifier of the trackbar to create
        initial_value : int
            Initial value of the trackbar
        max_value : int
            Maximal value of the trackbar
        """
        self._slider_values[slider_name] = initial_value
        cv2.createTrackbar(slider_name, self._window_name,
                           self._slider_values[slider_name], max_value,
                           _on_trackbar)

    def add_poseestimator_options(self, pose_estimator, args):
        """
        Add several trackbars, one for each option of the given pose estimator.

        Parameters
        ----------
        pose_estimator : PoseEstimator object
            Pose estimator object that will be used to decode results
        args : dict
            Dictionary of the command line args to determine initial values
        """
        for option_name, option in pose_estimator.get_options():
            if option_name in args:
                # command line args take priority
                initial_value = args[option_name]
            else:
                initial_value = option["default"]
            self.add_slider(slider_name=option_name,
                            initial_value=initial_value,
                            max_value=option["max_val"])

    def get_changes(self):
        """
        Get a dict of all changes to trackbars since the last call.

        Checks all trackbars for changes, updates its memory of previous
        trackbar values and returns as dict of all changes.

        Returns
        -------
        dict : Dictionary of all trackbar changes of the form
               {trackbar_name: value}
        """
        changes = {}
        for slider_name, val in self._slider_values.items():
            if val != cv2.getTrackbarPos(slider_name, self._window_name):
                self._slider_values[slider_name] = cv2.getTrackbarPos(
                    slider_name, self._window_name)
                changes[slider_name] = self._slider_values[slider_name]
        return changes
