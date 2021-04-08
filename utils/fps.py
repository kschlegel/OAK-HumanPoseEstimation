import time

import cv2


class FPS:
    """
    Helper class to handle multiple FPS counters at once.
    """
    def __init__(self, *counters, interval=10):
        """
        Calculate various FPS counters simultaniously.

        Parameters
        ----------
        counters : string
            Sequence of strings as positional arguments, identifiers for the
            fps counters.
        interval : float
            Calculate the display FPS only roughly every interval seconds for
            a smoother display
        """
        # Individual FPS counters
        self._counters = {c: 0 for c in counters}
        self._totals = {c: 0 for c in counters}
        self._fps = {c: 0 for c in counters}
        self._print_text = ", ".join([c + " FPS: {:.1f}" for c in counters])

        # Count update cycles to only update every interval frames for a
        # smoother counter
        self._interval = interval
        self._update_time = 0
        self._frame_time = 0

        # Times for overall runtime for statistics at the end
        self._start_time = time.perf_counter()

    def start_frame(self):
        """
        Saves the time at the start of a new loop.

        Each individual fps counter will calculate against this time.
        """
        self._frame_time = time.perf_counter()

    def count(self, counter):
        """
        Increments the specified counter.

        Parameters
        ----------
        counter : string
            Identifier of the counter to update
        """
        self._counters[counter] += 1
        self._totals[counter] += 1

    def update(self):
        """
        Preiodically update the display for all fps counter.

        If at least self._interval seconds have passed calculates the display
        FPS for all counters.
        """
        if self._frame_time - self._update_time > self._interval:
            for counter in self._counters.keys():
                self._fps[counter] = (self._counters[counter] /
                                      (self._frame_time - self._update_time))
                self._counters[counter] = 0
            self._update_time = self._frame_time

    def display(self, frame):
        """
        Prints all FPS counter onto the frame.

        Parameters
        ----------
        frame : numpy array
            The frame to print on. The frame is modified in place.
        """
        cv2.rectangle(frame, (0, 0), (110 * len(self._fps), 20), (0, 0, 0), -1)
        cv2.putText(frame,
                    self._print_text.format(*self._fps.values()), (2, 15),
                    cv2.FONT_HERSHEY_TRIPLEX,
                    0.4,
                    color=(255, 255, 255))

    def print_totals(self):
        """
        Print number of frames processed and avg fps for each counter.

        Output is written to command line.
        """
        run_time = time.perf_counter() - self._start_time
        print("Totals:")
        for counter, val in self._totals.items():
            print(counter, ":", val, " - Avg FPS:", val / run_time)
