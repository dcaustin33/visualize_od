#include "kalman_filter.h"
#include <cmath>
#include <Eigen/Dense>

KalmanStateVectorNDAdaptiveQ::KalmanStateVectorNDAdaptiveQ(const Eigen::VectorXd &states)
{
    state_matrix = states;
    int state_size = state_matrix.size();
    q = Eigen::MatrixXd::Identity(state_size, state_size);
    cov = Eigen::MatrixXd::Zero(state_size, state_size);
    f = Eigen::MatrixXd::Identity(state_size, state_size);

    int index = state_size / 2;
    f.block(0, index, index, index) = Eigen::MatrixXd::Identity(index, index);
}

void KalmanStateVectorNDAdaptiveQ::initialize_covariance(double noise_std)
{
    int state_size = state_matrix.size();
    cov = Eigen::MatrixXd::Identity(state_size, state_size) * noise_std * noise_std;
}

void KalmanStateVectorNDAdaptiveQ::predict_next_state(double dt)
{
    state_matrix = f * state_matrix;
    predict_next_covariance(dt);
}

void KalmanStateVectorNDAdaptiveQ::predict_next_covariance(double dt)
{
    cov = f * cov * f.transpose() + q;
}

void KalmanStateVectorNDAdaptiveQ::update_q(const Eigen::VectorXd &innovation, const Eigen::MatrixXd &kalman_gain, double alpha)
{
    q = alpha * q + (1 - alpha) * kalman_gain * innovation * innovation.transpose() * kalman_gain.transpose();
}

// KalmanNDTrackerAdaptiveQ Implementation

KalmanNDTrackerAdaptiveQ::KalmanNDTrackerAdaptiveQ(
    const KalmanStateVectorNDAdaptiveQ &initial_state,
    double R_val,
    double Q_val,
    const Eigen::MatrixXd &h_matrix) : state(initial_state)
{
    state.initialize_covariance(Q_val);
    if (h_matrix.size() == 0)
    {
        h = Eigen::MatrixXd::Identity(state.state_matrix.size(), state.state_matrix.size());
    }
    else
    {
        h = h_matrix;
    }
    R = Eigen::MatrixXd::Identity(h.rows(), h.rows()) * R_val * R_val;
    previous_measurements.push_back(h * state.state_matrix);
}

void KalmanNDTrackerAdaptiveQ::predict(double dt)
{
    state.predict_next_state(dt);
}

void KalmanNDTrackerAdaptiveQ::update_covariance(const Eigen::MatrixXd &gain)
{
    state.cov -= gain * h * state.cov;
}

void KalmanNDTrackerAdaptiveQ::update(const Eigen::VectorXd &measurement, double dt, bool predict, int max_measurement_stores)
{
    previous_measurements.push_back(measurement);
    if (previous_measurements.size() > max_measurement_stores)
    {
        previous_measurements.erase(previous_measurements.begin());
    }
    if (predict)
    {
        this->predict(dt);
    }
    Eigen::VectorXd innovation = measurement - h * state.state_matrix;
    Eigen::MatrixXd gain_invertible = h * state.cov * h.transpose() + R;
    Eigen::MatrixXd gain = state.cov * h.transpose() * gain_invertible.inverse();
    Eigen::VectorXd new_state = state.state_matrix + gain * innovation;
    update_covariance(gain);
    state.update_q(innovation, gain);
    state.state_matrix = new_state;
}

double KalmanNDTrackerAdaptiveQ::compute_mahalanobis_distance(const Eigen::VectorXd &measurement)
{
    Eigen::VectorXd innovation = measurement - h * state.state_matrix;
    Eigen::MatrixXd S = h * state.cov * h.transpose() + R;
    double distance = std::sqrt(innovation.transpose() * S.inverse() * innovation);
    return distance;
}

double KalmanNDTrackerAdaptiveQ::compute_p_value(double distance)
{
    int df = h.rows();
    // Assuming standard normal distribution for simplicity, chat gpt note was need a statistical library to compute this accurately
    double p_value = 1.0 - std::erf(distance / std::sqrt(2));
    return p_value;
}

double KalmanNDTrackerAdaptiveQ::compute_p_value_from_measurement(const Eigen::VectorXd &measurement)
{
    double distance = compute_mahalanobis_distance(measurement);
    return compute_p_value(distance);
}