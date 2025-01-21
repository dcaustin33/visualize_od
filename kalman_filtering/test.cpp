#include "kalman_filter.h"
#include <fstream>
#include <random>

int main() {
    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> noise(0, 1.0);

    // Create output file
    std::ofstream outFile("kalman_output.csv");
    std::ofstream dataFile("data.csv");
    
    // Initial state (x, y, vx, vy)
    Eigen::VectorXd initial_state(4);
    initial_state << 0, 0, 0, 0;  // Starting at origin with velocity (1,1)
    
    // Create Kalman filter
    KalmanStateVectorNDAdaptiveQ kf_state(initial_state);
    
    // Measurement matrix (we only measure position, not velocity)
    Eigen::MatrixXd H(2, 4);
    H << 1, 0, 0, 0,
         0, 1, 0, 0;
    
    KalmanNDTrackerAdaptiveQ tracker(kf_state, 15, 2.5, H);
    
    // Simulate measurements and filter
    double dt = 1.0;
    double t = 0;
    
    for (int i = 0; i < 100; i++) {
        // True position (circular motion for this example)
        double true_x = 1 * t;
        double true_y = 1 * t;
        
        // Add noise to create measurement
        Eigen::VectorXd measurement(2);
        measurement << true_x + noise(gen), true_y + noise(gen);
        dataFile << t << "," << measurement(0) << "," << measurement(1) << "\n";
        
        // Update filter
        tracker.update(measurement, dt);
        
        // Write to file: time, measured_x, measured_y, filtered_x, filtered_y
        outFile << t << ","
                << measurement(0) << "," 
                << measurement(1) << ","
                << tracker.state.state_matrix(0) << ","
                << tracker.state.state_matrix(1) << "\n";
        
        t += dt;
    }
    
    outFile.close();
    dataFile.close();
    return 0;
}