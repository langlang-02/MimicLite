"""Read-only, per-control-step tracking diagnostics in reference heading axes."""
import numpy as np


def yaw_wxyz(q):
    w, x, y, z = q
    return float(np.arctan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z)))


def wrap_angle(angle):
    return float((angle + np.pi) % (2*np.pi) - np.pi)


def heading_xy(vector, yaw):
    x, y = vector[:2]
    return np.array([np.cos(yaw)*x + np.sin(yaw)*y,
                     -np.sin(yaw)*x + np.cos(yaw)*y])


class TrackingDiagnostics:
    def __init__(self, names, ori_indices, threshold):
        self.names = list(names)
        self.indices = list(ori_indices)
        self.threshold = threshold
        self.previous = None
        self.consecutive = 0

    def record(self, time_s, robot_pos, ref_pos, robot_quat, ref_quat,
               body_ori_local, body_pos_world, body_pos_local, torso_index):
        ry, fy = yaw_wxyz(robot_quat), yaw_wxyz(ref_quat)
        delta = robot_pos - ref_pos
        xy = heading_xy(delta, fy)
        errors = np.asarray(body_ori_local)[self.indices]
        worst = self.indices[int(np.argmax(errors))]
        self.consecutive = self.consecutive + 1 if max(errors) >= self.threshold else 0
        row = dict(time_s=time_s, root_forward_error_m=float(xy[0]), root_left_error_m=float(xy[1]),
                   root_height_error_m=float(delta[2]), robot_yaw_rad=ry, reference_yaw_rad=fy,
                   yaw_error_rad=wrap_angle(ry-fy), torso_position_error_m=float(body_pos_world[torso_index]),
                   body_ori_max_rad=float(max(errors)), body_ori_worst_link=self.names[worst],
                   body_ori_exceeded_consecutive_steps=self.consecutive)
        # Root-link position derivatives, not COM velocities. Blank first step avoids fabricated zero.
        for name in ['robot_forward_velocity_mps','robot_left_velocity_mps','reference_forward_velocity_mps',
                     'reference_left_velocity_mps','forward_velocity_error_mps','left_velocity_error_mps',
                     'robot_yaw_rate_radps','reference_yaw_rate_radps','yaw_rate_error_radps']:
            row[name] = None
        if self.previous is not None:
            t, rp, fp, pry, pfy = self.previous
            dt = time_s-t
            rv = heading_xy((robot_pos-rp)/dt, fy); fv = heading_xy((ref_pos-fp)/dt, fy)
            rr, fr = wrap_angle(ry-pry)/dt, wrap_angle(fy-pfy)/dt
            for name, value in zip(['robot_forward_velocity_mps','robot_left_velocity_mps','reference_forward_velocity_mps',
                                   'reference_left_velocity_mps','forward_velocity_error_mps','left_velocity_error_mps',
                                   'robot_yaw_rate_radps','reference_yaw_rate_radps','yaw_rate_error_radps'],
                                  [*rv,*fv,*(rv-fv),rr,fr,rr-fr]):
                row[name] = float(value)
        for i, name in enumerate(self.names):
            row[f'body_ori_local_rad/{name}'] = float(body_ori_local[i])
            row[f'body_pos_local_m/{name}'] = float(body_pos_local[i])
        self.previous = (time_s, robot_pos.copy(), ref_pos.copy(), ry, fy)
        return row
