import pytensor
import pytensor.tensor as at
from pytensor.tensor.nlinalg import matrix_dot


class KalmanSmoother:
    def build_graph(self, T, R, Q, filtered_states, filtered_covariances):
        a_last = filtered_states[-1]
        P_last = filtered_covariances[-1]

        smoother_result, updates = pytensor.scan(self.smoother_step,
                                               sequences=[filtered_states[:-1],
                                                          filtered_covariances[:-1]],
                                               outputs_info=[a_last, P_last],
                                               non_sequences=[T, R, Q],
                                               go_backwards=True,
                                               name='kalman_smoother')

        smoothed_states, smoothed_covariances = smoother_result
        smoothed_states = at.concatenate([smoothed_states[::-1], a_last[None]], axis=0)
        smoothed_covariances = at.concatenate([smoothed_covariances[::-1],
                                               P_last[None]], axis=0)

        return smoothed_states, smoothed_covariances

    def smoother_step(self, a, P, a_smooth, P_smooth, T, R, Q):
        a_hat, P_hat = self.predict(a, P, T, R, Q)

        # Use pinv, otherwise P_hat is singular when there is missing data
        smoother_gain = matrix_dot(at.linalg.pinv(P_hat), T, P).T

        a_smooth_next = a + smoother_gain @ (a_smooth - a_hat)
        P_smooth_next = P + matrix_dot(smoother_gain, P_smooth - P_hat, smoother_gain.T)

        return a_smooth_next, P_smooth_next#

    @staticmethod
    def predict(a, P, T, R, Q):
        a_hat = T.dot(a)
        P_hat = matrix_dot(T, P, T.T) + matrix_dot(R, Q, R.T)

        return a_hat, P_hat


# class UnivariateKalmanSmoother:
#
#     def build_graph(self, T, R, Q, predicted_states, predicted_covariances):
#         smoother_result, updates = aesara.scan(self.smoother_step,
#                                                sequences=[predicted_states[:-1],
#                                                           predicted_covariances[:-1],
#                                                           v_history, F_history, K_history],
#                                                outputs_info=[None, at.zeros_like(a0), None, at.zeros_like(P0)],
#                                                non_sequences=[T, Z],
#                                                go_backwards=True,
#                                                name='backward_kalman_pass')
#
#         smoothed_states, _, smoothed_covariances, _ = smoother_result
#         smoothed_states = at.concatenate([smoothed_states[::-1], predicted_states[-1][None]], axis=0)[:-1]
#         smoothed_covariances = at.concatenate([smoothed_covariances[::-1],
#                                                predicted_covariances[-1][None]], axis=0)[:-1]
#
#         return smoothed_states, smoothed_covariances
#
#     @staticmethod
#     def _univariate_inner_smoother_step(v, F, K, Z, r_p, N_p, T):
#         Z_row = Z[None, :]
#         L = at.eye(T.shape[0]) - K @ Z_row
#         ZTF_inv = Z_row.T / F
#
#         r_next = ZTF_inv * v + L.T @ r_p
#         N_next = ZTF_inv.dot(Z_row) + matrix_dot(L.T, N_p, L)
#
#         return r_next, N_next
#
#     def smoother_step(self, a, P, v, F, K, r_t, N_t, T, Z):
#         results, updates = aesara.scan(self._univariate_inner_smoother_step,
#                                        sequences=[v, F, K, Z],
#                                        outputs_info=[r_t, N_t],
#                                        non_sequences=[T])
#
#         r_next, N_next = results
#         r_next = r_next[-1]
#         N_next = N_next[-1]
#
#         a_smooth = a + P.dot(r_next)
#         P_smooth = P - matrix_dot(P, N_next, P)
#
#         new_r = T.T.dot(r_next)
#         new_N = matrix_dot(T.T, N_next, T)
#
#         return a_smooth, new_r, P_smooth, new_N
