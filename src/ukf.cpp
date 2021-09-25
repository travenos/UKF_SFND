#include "ukf.h"
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;

constexpr double NEAR_ZERO_VALUE = 1e-20;

static double NormalizeAngle(double angle)
{
    while (angle > M_PI)
    {
        angle -= 2. * M_PI;
    }
    while (angle < -M_PI)
    {
        angle += 2. * M_PI;
    }
    return angle;
}

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF()
{
  // initially set to false, set to true in first call of ProcessMeasurement
  is_initialized_ = false;

  time_us_ = -1; // not initialized

  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd::Zero(5);

  // initial covariance matrix
  P_ = MatrixXd::Identity(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 3;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 2 * M_PI / 3 ;
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  
  // State dimension
  n_x_ = static_cast<int>(x_.size());

  // Augmented state dimension
  n_aug_ = n_x_ + 2;

  // Sigma point spreading parameter
  lambda_ = 3 - n_aug_;

  // Weights of sigma points
  weights_ = VectorXd(2 * n_aug_ + 1);
  weights_(0) = lambda_ / (lambda_ + n_aug_);
  weights_.tail(weights_.size() - 1) = 0.5 / (lambda_ + n_aug_) * VectorXd::Ones(weights_.size() - 1);

  Xsig_pred_ = MatrixXd::Zero(n_x_, 2 * n_aug_ + 1);
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(const MeasurementPackage& meas_package) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */
    const double delta_t = (meas_package.timestamp_ - time_us_) * 1e-6;
    Prediction(delta_t);
    switch (meas_package.sensor_type_)
    {
        case MeasurementPackage::LASER:
            UpdateLidar(meas_package);
            break;
        case MeasurementPackage::RADAR:
            UpdateRadar(meas_package);
            break;
    }
    time_us_ = meas_package.timestamp_;
}

void UKF::Prediction(double delta_t)
{
  /**
   * Estimate the object's location.
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */
    if (!is_initialized_)
    {
        return;
    }
    // Creation of augmented mean state
    VectorXd x_aug = VectorXd::Zero(n_aug_);
    x_aug.head(x_.size()) = x_;

    // Creation of augmented covariance matrix
    MatrixXd P_aug = MatrixXd::Zero(n_aug_, n_aug_);
    P_aug.topLeftCorner(P_.rows(), P_.cols()) = P_;
    P_aug(P_.rows(), P_.cols()) = std_a_ * std_a_;
    P_aug(P_.rows() + 1, P_.cols() + 1) = std_yawdd_ * std_yawdd_;
    MatrixXd P_aug_sqrt = P_aug.llt().matrixL();

    // Creation of augmented sigma points
    MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
    for (int i = 0; i < Xsig_aug.cols(); ++i)
    {
        Xsig_aug.col(i) = x_aug;
    }
    Xsig_aug.block(0, 1, Xsig_aug.rows(), n_aug_) += std::sqrt(lambda_ + n_aug_) * P_aug_sqrt;
    Xsig_aug.block(0, n_aug_ + 1, Xsig_aug.rows(), n_aug_) -= std::sqrt(lambda_ + n_aug_) * P_aug_sqrt;

    // Prediction of sigma points
    const double dt2 = delta_t * delta_t;
    for (int i = 0; i < Xsig_pred_.cols(); ++i)
    {
        const double v = Xsig_aug(2, i);
        const double psi = Xsig_aug(3, i);
        const double psi_dot = Xsig_aug(4, i);
        const double sin_psi = std::sin(psi);
        const double cos_psi = std::cos(psi);
        Xsig_pred_.col(i) = Xsig_aug.col(i).head(n_x_);
        if (std::fabs(psi_dot) > NEAR_ZERO_VALUE)
        {
            Xsig_pred_(0, i) += v / psi_dot * (std::sin(psi + psi_dot * delta_t) - sin_psi);
            Xsig_pred_(1, i) += v / psi_dot * (-std::cos(psi + psi_dot * delta_t) + cos_psi);
            Xsig_pred_(3, i) += psi_dot * delta_t;
        }
        else
        {
            Xsig_pred_(0, i) += v * cos_psi * delta_t;
            Xsig_pred_(1, i) += v * sin_psi * delta_t;
        }

        const double nu_a = Xsig_aug(5, i);
        const double nu_psi_dot_dot = Xsig_aug(6, i);

        Xsig_pred_(0, i) += 0.5 * dt2 * cos_psi * nu_a;
        Xsig_pred_(1, i) += 0.5 * dt2 * sin_psi * nu_a;
        Xsig_pred_(2, i) += delta_t * nu_a;
        Xsig_pred_(3, i) += 0.5 * dt2 * nu_psi_dot_dot;
        Xsig_pred_(4, i) += delta_t * nu_psi_dot_dot;
    }

    // State mean prediction
    x_ = Xsig_pred_ * weights_;

    // State covariance matrix prediction
    P_.fill(0);
    for (int i = 0; i < Xsig_pred_.cols(); ++i)
    {
        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        // Angle normalization
        x_diff(3) = NormalizeAngle(x_diff(3));
        P_ += x_diff * x_diff.transpose() * weights_(i);
    }
}


void UKF::UpdateLidar(const MeasurementPackage& meas_package)
{
    /**
     * TODO: Complete this function! Use lidar data to update the belief
     * about the object's position. Modify the state vector, x_, and
     * covariance, P_.
     * You can also calculate the lidar NIS, if desired.
     */
    if (!is_initialized_)
    {
        Initialize(meas_package);
        return;
    }

    const long n_z = meas_package.raw_measurements_.size();
    // Matrix for sigma points in measurement space
    MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
    // Transformation of sigma points into measurement space
    Zsig = Xsig_pred_.topLeftCorner(n_z, Zsig.cols());

    // Measurement noise matrix
    MatrixXd R = MatrixXd::Zero(n_z, n_z);
    R(0, 0) = std_laspx_ * std_laspx_;
    R(1, 1) = std_laspy_ * std_laspy_;

    UpdateCommon(meas_package.raw_measurements_, Zsig, R);
}

void UKF::UpdateRadar(const MeasurementPackage& meas_package)
{
    /**
     * TODO: Complete this function! Use radar data to update the belief
     * about the object's position. Modify the state vector, x_, and
     * covariance, P_.
     * You can also calculate the radar NIS, if desired.
     */
    if (!is_initialized_)
    {
        // Initialize only with the first lidar measurement
        return;
    }

    const long n_z = meas_package.raw_measurements_.size();
    // Matrix for sigma points in measurement space
    MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

    // Transformation of sigma points into measurement space
    for (int i = 0; i < Xsig_pred_.cols(); ++i)
    {
        const double px = Xsig_pred_(0, i);
        const double py = Xsig_pred_(1, i);
        const double v = Xsig_pred_(2, i);
        const double psi = Xsig_pred_(3, i);

        Zsig(0, i) = std::sqrt(px * px + py * py);
        Zsig(1, i) = std::atan2(py, px);
        Zsig(2, i) = std::fabs(Zsig(0, i)) > NEAR_ZERO_VALUE ? (px * std::cos(psi) * v + py * std::sin(psi) * v) / Zsig(0, i) : 0;
    }

    // Measurement noise matrix
    MatrixXd R = MatrixXd::Zero(n_z, n_z);
    R(0, 0) = std_radr_ * std_radr_;
    R(1, 1) = std_radphi_ * std_radphi_;
    R(2, 2) = std_radrd_ * std_radrd_;

    UpdateCommon(meas_package.raw_measurements_, Zsig, R);
}

void UKF::Initialize(const MeasurementPackage& meas_package)
{
    x_ = VectorXd::Zero(n_x_);
    x_(0) = meas_package.raw_measurements_(0);
    x_(1) = meas_package.raw_measurements_(1);
    x_(2) = 6; // Assume average velocity in m/s
    x_(3) = 0; // Assume average angle in rad
    x_(4) = 0; // Assume average angle velocity in rad / s

    P_ = MatrixXd::Zero(n_x_, n_x_);
    P_(0, 0) = std_laspx_ * std_laspx_;
    P_(1, 1) = std_laspy_ * std_laspy_;
    P_(2, 2) = 6 / 2 * 6 / 2; // Assume initial uncertaity of the velocity
    P_(3, 3) = M_PI / 6 / 2 * M_PI / 6 / 2; // Assume initial uncertaity of the yaw angle
    P_(4, 4) = M_PI / 4 / 2 * M_PI / 4 / 2; // Assume initial uncertaity of the angle velocity

    is_initialized_ = true;
}

void UKF::UpdateCommon(const VectorXd& z, const MatrixXd& Zsig, const MatrixXd& R)
{
    const long n_z = z.size();
    VectorXd z_pred = Zsig * weights_;

    // Measurement covariance matrix S
    MatrixXd S = MatrixXd::Zero(n_z, n_z);
    for (int i = 0; i < Zsig.cols(); ++i)
    {
        VectorXd z_diff = Zsig.col(i) - z_pred;
        // Angle normalization
        z_diff(1) = NormalizeAngle(z_diff(1));
        S += z_diff * z_diff.transpose() * weights_(i);
    }
    S += R;

    // Calculation of cross correlation matrix
    MatrixXd Tc = MatrixXd::Zero(n_x_, n_z);
    for (int i = 0; i < Zsig.cols(); ++i)
    {
        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        // Angle normalization
        x_diff(3) = NormalizeAngle(x_diff(3));

        VectorXd z_diff = Zsig.col(i) - z_pred;
        // Angle normalization
        z_diff(1) = NormalizeAngle(z_diff(1));
        Tc += x_diff * z_diff.transpose() * weights_(i);
    }

    // Calculation of Kalman gain
    MatrixXd K = Tc * S.inverse();

    // Update of state mean and covariance matrix
    VectorXd z_res = z - z_pred;
    z_res(1) = NormalizeAngle(z_res(1));

    x_ += K * z_res;
    P_ -= K * S * K.transpose();
}
