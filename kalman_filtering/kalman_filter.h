#ifndef KALMAN_H
#define KALMAN_H

#include <Eigen/Dense>
#include <vector>

class KalmanStateVectorNDAdaptiveQ {
public:
    Eigen::VectorXd state_matrix;
    Eigen::MatrixXd cov;
    Eigen::MatrixXd q;
    Eigen::MatrixXd f;

    KalmanStateVectorNDAdaptiveQ(const Eigen::VectorXd& states);

    void initialize_covariance(double noise_std);

    void predict_next_state(double dt);

    void predict_next_covariance(double dt);

    void update_q(const Eigen::VectorXd& innovation, const Eigen::MatrixXd& kalman_gain, double alpha = 0.98);
};

class KalmanNDTrackerAdaptiveQ {
public:
    KalmanStateVectorNDAdaptiveQ state;
    Eigen::MatrixXd h;
    Eigen::MatrixXd R;
    std::vector<Eigen::VectorXd> previous_measurements;

    KalmanNDTrackerAdaptiveQ(
        const KalmanStateVectorNDAdaptiveQ& initial_state,
        double R_val,
        double Q_val,
        const Eigen::MatrixXd& h_matrix = Eigen::MatrixXd()
    );

    void predict(double dt);

    void update_covariance(const Eigen::MatrixXd& gain);

    void update(const Eigen::VectorXd& measurement, double dt = 1.0, bool predict = true, int max_measurement_stores = 10);

    double compute_mahalanobis_distance(const Eigen::VectorXd& measurement);

    double compute_p_value(double distance);

    double compute_p_value_from_measurement(const Eigen::VectorXd& measurement);
};

#endif // KALMAN_H