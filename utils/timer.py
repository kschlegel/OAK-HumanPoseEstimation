import time

import numpy as np


class Timer:
    """
    Helper class to time several processes concurrently and repeatedly.

    Times several processes such asd inference and result decoding at the same
    time. Keeps a log of each iteration to provide runtime statistics for each
    tracked process.
    """
    def __init__(self, *processes):
        """
        Time several processes simultaniously.

        Identifiers for timed processed are passed in as sequence of strings.
        Times each process individually and stores results to produce runtime
        statistics at the end.

        Parameters
        ----------
        processes : string
            Sequence of strings as positional arguments, identifiers for the
            processes to be timed.
        """
        self._start_times = {p: None for p in processes}
        self._run_times = {p: [] for p in processes}

    def start_timer(self, process):
        """
        Starts the timer for the specified process.

        Parameters
        ----------
        process : str
            Identifier of the process to be started
        """
        self._start_times[process] = time.perf_counter()

    def stop_timer(self, process):
        """
        Stops the timer for the specified process.

        Parameters
        ----------
        process : str
            Identifier of the process to be stopped
        """
        if self._start_times[process] is not None:
            self._run_times[process].append(time.perf_counter() -
                                            self._start_times[process])

    def frame_time(self, process, frame):
        """
        Determines the processing time of the given frame.

        Pass in a DepthAI ImgFrame object to determine the processing time of
        the frame using the frames timestamp.

        Parameters
        ----------
        process : str
            Identifier of the process that produced the frame
        frame : depthai ImgFrame object
            The frame as obtained from the NN passthrough
        """
        self._run_times[process].append(time.monotonic() -
                                        frame.getTimestamp().total_seconds())

    def print_times(self):
        """
        Prints runtime statistics for each timed process.

        Output is written to command line.
        """
        output = "{process} (#{cnt}) runtimes: avg: {avg}; min: {min};"
        output += " max: {max}"
        for process, times in self._run_times.items():
            if len(times) > 1:
                print(
                    output.format(process=process,
                                  cnt=len(times),
                                  avg=np.mean(times),
                                  min=np.amin(times),
                                  max=np.amax(times)))
            elif len(times) == 1:
                print(process, "runtime:", times[0])
            else:
                print("Process", process, "did not run.")
