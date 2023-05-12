import pytensor
import pytensor.tensor as pt
from pytensor.tensor.nlinalg import matrix_dot


class KalmanSmoother:
    def build_graph(self, T, R, Q, filtered_states, filtered_covariances):
        a_last = filtered_states[-1]
        P_last = filtered_covariances[-1]

        smoother_result, updates = pytensor.scan(
            self.smoother_step,
            sequences=[filtered_states[:-1], filtered_covariances[:-1]],
            outputs_info=[a_last, P_last],
            non_sequences=[T, R, Q],
            go_backwards=True,
            name="kalman_smoother",
        )

        smoothed_states, smoothed_covariances = smoother_result
        smoothed_states = pt.concatenate([smoothed_states[::-1], pt.atleast_3d(a_last)], axis=0)
        smoothed_covariances = pt.concatenate(
            [smoothed_covariances[::-1], pt.atleast_3d(P_last)], axis=0
        )

        return smoothed_states, smoothed_covariances

    def smoother_step(self, a, P, a_smooth, P_smooth, T, R, Q):
        a_hat, P_hat = self.predict(a, P, T, R, Q)

        # Use pinv, otherwise P_hat is singular when there is missing data
        smoother_gain = matrix_dot(pt.linalg.pinv(P_hat), T, P).T
        a_smooth_next = a + smoother_gain @ (a_smooth - a_hat)

        P_smooth_next = P + matrix_dot(smoother_gain, P_smooth - P_hat, smoother_gain.T)

        return a_smooth_next, P_smooth_next  #

    @staticmethod
    def predict(a, P, T, R, Q):
        a_hat = T.dot(a)
        P_hat = matrix_dot(T, P, T.T) + matrix_dot(R, Q, R.T)

        return a_hat, P_hat
