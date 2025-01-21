import csv
import numpy as np
from object_tracking.kalman_filter.schema_n_adaptive_q import (
    KalmanNDTrackerAdaptiveQ,
    KalmanStateVectorNDAdaptiveQ,
)

STATE = KalmanStateVectorNDAdaptiveQ
TRACKER = KalmanNDTrackerAdaptiveQ


def read_csv(file_path: str):
    with open(file_path, "r") as file:
        reader = csv.reader(file)
        data = list(reader)
    final_data = []
    for row in data:
        final_data.append([float(x) for x in row])
    return final_data


if __name__ == "__main__":
    data = read_csv("build/data.csv")
    h = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

    initial_state = STATE(np.array([0, 0, 0, 0]))
    tracker = TRACKER(initial_state, 15, 2.5, h)

    output_kalman_state = []
    for i in range(len(data)):
        measurement = np.array(data[i][1:])
        tracker.update(measurement)
        output_kalman_state.append(
            [tracker.state.state_matrix[0], tracker.state.state_matrix[1]]
        )


    with open("py_kalman_output.csv", "w") as file:
        writer = csv.writer(file)
        for i, (state, original) in enumerate(zip(output_kalman_state, data)):
            writer.writerow([i] + original[1:] + state)
