class CoordinateManager:
    def __init__(self):
        # Store the known NED-frame offset of the stereo camera relative to the
        # NED origin (mm-wave radar position). The radar itself has zero offset.
        self.camera_offset = None

    def update_gnss_fixes(self, gnss_fixes):
        # Accept GNSS fixes from the vessel and maintain a running estimate of
        # the vessel NED position v(t), used as the effective position offset
        # for AIS measurements.
        return

    def v(self, t):
        # t: time

        # Accept GNSS fixes from the vessel and maintain a running estimate of
        # the vessel NED position v(t), used as the effective position offset
        # for AIS measurements.
        return

    def h(self, i, x, t):
        # i: sensor
        # x: target state
        # t: time

        # Compute h_i(x, t) and H_i for any target state x and sensor i, using the
        # appropriate position offset. For AIS, which outputs NED position directly,
        # compute the implied (range, bearing) relative to the vessel and provide
        # the corresponding Jacobian.
        match i:
            case "radar":
                return
            case "camera":
                return
            case "ais":
                return
            case "gnss":
                return

    def H(self, i):
        # i: sensor

        # Compute h_i(x, t) and H_i for any target state x and sensor i, using the
        # appropriate position offset. For AIS, which outputs NED position directly,
        # compute the implied (range, bearing) relative to the vessel and provide
        # the corresponding Jacobian.
        match i:
            case "radar":
                return
            case "camera":
                return
            case "ais":
                return
            case "gnss":
                return

    def R(self, i):
        # i: sensor

        # Provide the measurement noise covariance R_i for each sensor,
        # configured from the sensor specifications table.
        match i:
            case "radar":
                return
            case "camera":
                return
            case "ais":
                return
            case "gnss":
                return


if __name__ == "__main__":
    cm = CoordinateManager()

    # Write unit tests verifying that:
    #   (a) a known NED-frame target position generates the expected (range, bearing) for each sensor;
    #   (b) the measurement function produces the expected output for a known input; and
    #   (c) the AIS position-to-observation conversion is consistent with the radar measurement at the same target location.
