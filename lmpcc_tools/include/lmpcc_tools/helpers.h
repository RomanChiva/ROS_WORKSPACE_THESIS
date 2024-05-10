
#ifndef HELPERS_H
#define HELPERS_H

#include <random>
#include <Eigen/Eigen>
#include <chrono>
#include <string>
#include <tf/tf.h>
#include <tf/transform_listener.h>
#include <thread>
#include <mutex>
#include <std_msgs/Float64MultiArray.h>
#include <ros/package.h>
#include <fstream>

#include <lmpcc_msgs/halfspace_array.h>
#include <lmpcc_msgs/halfspace.h>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/math/distributions/laplace.hpp>

#include <assert.h>

#include <thread>
#include <mutex>

#include <fstream>
#include <ros/package.h>

#include <geometry_msgs/Pose.h>

#include "lmpcc_tools/ros_visuals.h"
#include "lmpcc_tools/data_saver.h"

/** Logging Pragmas */
#define LMPCC_INFO(msg)                      \
	if (config_->debug_output_)              \
	{                                        \
		ROS_INFO_STREAM("[LMPCC]: " << msg); \
	}

#define LMPCC_WARN(msg)                      \
	if (config_->debug_output_)              \
	{                                        \
		ROS_WARN_STREAM("[LMPCC]: " << msg); \
	}

#define LMPCC_ERROR(msg) ROS_ERROR_STREAM("[LMPCC]: " << msg)

#define LMPCC_INFO_STREAM(msg)               \
	if (config_->debug_output_)              \
	{                                        \
		ROS_INFO_STREAM("[LMPCC]: " << msg); \
	}

#define LMPCC_WARN_STREAM(msg)               \
	if (config_->debug_output_)              \
	{                                        \
		ROS_WARN_STREAM("[LMPCC]: " << msg); \
	}

#define LMPCC_ERROR_STREAM(msg) ROS_ERROR_STREAM("[LMPCC]: " << msg)

#define LMPCC_INFO_ALWAYS(msg) ROS_INFO_STREAM("[LMPCC]: " << msg)
#define LMPCC_WARN_ALWAYS(msg) ROS_WARN_STREAM("[LMPCC]: " << msg)

// Copied from decomp_util!
typedef double decimal_t;

/// Pre-allocated std::vector for Eigen using vec_E
template <typename T>
using vec_E = std::vector<T, Eigen::aligned_allocator<T>>;
/// Eigen 1D float vector
template <int N>
using Vecf = Eigen::Matrix<decimal_t, N, 1>;
/// Eigen 1D int vector
template <int N>
using Veci = Eigen::Matrix<int, N, 1>;
/// MxN Eigen matrix
template <int M, int N>
using Matf = Eigen::Matrix<decimal_t, M, N>;
/// MxN Eigen matrix with M unknown
template <int N>
using MatDNf = Eigen::Matrix<decimal_t, Eigen::Dynamic, N>;
/// Vector of Eigen 1D float vector
template <int N>
using vec_Vecf = vec_E<Vecf<N>>;
/// Vector of Eigen 1D int vector
template <int N>
using vec_Veci = vec_E<Veci<N>>;

/// Eigen 1D float vector of size 2
typedef Vecf<2> Vec2f;
/// Eigen 1D int vector of size 2
typedef Veci<2> Vec2i;
/// Eigen 1D float vector of size 3
typedef Vecf<3> Vec3f;
/// Eigen 1D int vector of size 3
typedef Veci<3> Vec3i;
/// Eigen 1D float vector of size 4
typedef Vecf<4> Vec4f;
/// Column vector in float of size 6
typedef Vecf<6> Vec6f;

/// Vector of type Vec2f.
typedef vec_E<Vec2f> vec_Vec2f;
/// Vector of type Vec2i.
typedef vec_E<Vec2i> vec_Vec2i;
/// Vector of type Vec3f.
typedef vec_E<Vec3f> vec_Vec3f;
/// Vector of type Vec3i.
typedef vec_E<Vec3i> vec_Vec3i;

/// 2x2 Matrix in float
typedef Matf<2, 2> Mat2f;
/// 3x3 Matrix in float
typedef Matf<3, 3> Mat3f;
/// 4x4 Matrix in float
typedef Matf<4, 4> Mat4f;
/// 6x6 Matrix in float
typedef Matf<6, 6> Mat6f;

/// Dynamic Nx1 Eigen float vector
typedef Vecf<Eigen::Dynamic> VecDf;
/// Nx2 Eigen float matrix
typedef MatDNf<2> MatD2f;
/// Nx3 Eigen float matrix
typedef MatDNf<3> MatD3f;
/// Dynamic MxN Eigen float matrix
typedef Matf<Eigen::Dynamic, Eigen::Dynamic> MatDf;

/// Hyperplane class
template <int Dim>
struct Hyperplane
{
	Hyperplane()
	{
	}
	Hyperplane(const Vecf<Dim> &p, const Vecf<Dim> &n) : p_(p), n_(n)
	{
	}

	/// Calculate the signed distance from point
	decimal_t signed_dist(const Vecf<Dim> &pt) const
	{
		return n_.dot(pt - p_);
	}

	/// Calculate the distance from point
	decimal_t dist(const Vecf<Dim> &pt) const
	{
		return std::abs(signed_dist(pt));
	}

	/// Point on the plane
	Vecf<Dim> p_;
	/// Normal of the plane, directional
	Vecf<Dim> n_;
};

/// Hyperplane2D: first is the point on the hyperplane, second is the normal
typedef Hyperplane<2> Hyperplane2D;
/// Hyperplane3D: first is the point on the hyperplane, second is the normal
typedef Hyperplane<3> Hyperplane3D;

// COPIED FROM DECOMP_UTIL!
// /[A, b] for \f$Ax < b\f$
template <int Dim>
struct LinearConstraint
{
	/// Null constructor
	LinearConstraint()
	{
	}
	/// Construct from \f$A, b\f$ directly, s.t \f$Ax < b\f$
	LinearConstraint(const MatDNf<Dim> &A, const VecDf &b) : A_(A), b_(b)
	{
	}
	/**
	 * @brief Construct from a inside point and hyperplane array
	 * @param p0 point that is inside
	 * @param vs hyperplane array, normal should go outside
	 */
	LinearConstraint(const Vecf<Dim> p0, const vec_E<Hyperplane<Dim>> &vs)
	{
		const unsigned int size = vs.size();
		MatDNf<Dim> A(size, Dim);
		VecDf b(size);

		for (unsigned int i = 0; i < size; i++)
		{
			auto n = vs[i].n_;
			decimal_t c = vs[i].p_.dot(n);
			if (n.dot(p0) - c > 0)
			{
				n = -n;
				c = -c;
			}
			A.row(i) = n;
			b(i) = c;
		}

		A_ = A;
		b_ = b;
	}

	/// Check if the point is inside polyhedron using linear constraint
	bool inside(const Vecf<Dim> &pt)
	{
		VecDf d = A_ * pt - b_;
		for (unsigned int i = 0; i < d.rows(); i++)
		{
			if (d(i) > 0)
				return false;
		}
		return true;
	}

	/// Get \f$A\f$ matrix
	MatDNf<Dim> A() const
	{
		return A_;
	}

	/// Get \f$b\f$ matrix
	VecDf b() const
	{
		return b_;
	}

	MatDNf<Dim> A_;
	VecDf b_;
};

// / LinearConstraint 2D
typedef LinearConstraint<2> LinearConstraint2D;
/// LinearConstraint 3D
typedef LinearConstraint<3> LinearConstraint3D;

enum class ObstacleType
{
	STATIC,
	DYNAMIC,
	RANGE
};
enum class ConstraintSide
{
	BOTTOM,
	TOP,
	UNDEFINED
};

struct Scenario
{
	int idx_;
	int obstacle_idx_;
};

struct ScenarioConstraint
{
	// LinearConstraint2D constraint_; // Improve later

	Scenario *scenario_;

	ObstacleType type_;
	ConstraintSide side_;

	ScenarioConstraint(){};

	ScenarioConstraint(Scenario *scenario, const ObstacleType &type, const ConstraintSide &side)
	{
		scenario_ = scenario;
		type_ = type;
		side_ = side;
	}

	bool isActive(const Eigen::Vector2d &x)
	{
		// double val = constraint_.A_(0, 0) * x(0) + constraint_.A_(0, 1) * x(1) - constraint_.b_(0);

		// return std::fabs(val) < 1e-4;
		return false;
	}

	double Value(const Eigen::Vector2d &x)
	{
		return 0;
		// double val = constraint_.A_(0, 0) * x(0) + constraint_.A_(0, 1) * x(1) - constraint_.b_(0);
		// return -val; // needs to be less than 0, so 0 - val
	}

	int GetHalfspaceIndex(int sample_size)
	{
		return type_ == ObstacleType::DYNAMIC ? sample_size * scenario_->obstacle_idx_ + scenario_->idx_ : scenario_->idx_;
	}
};

struct SupportSubsample
{
	std::vector<int> support_indices_;
	std::vector<Scenario *> scenarios_;

	int support_subsample_size_;

	SupportSubsample(int initial_size = 150)
	{
		support_subsample_size_ = 0;
		support_indices_.reserve(initial_size);
		scenarios_.reserve(initial_size);
	}

	void Add(Scenario &scenario)
	{
		// No duplicates
		if (ContainsScenario(scenario))
			return;

		// Note: will allocate if above size!
		support_indices_.push_back(scenario.idx_);
		scenarios_.push_back(&scenario);
		support_subsample_size_++;
	}

	void Reset()
	{
		support_subsample_size_ = 0;
		support_indices_.clear();
		scenarios_.clear();
	}

	bool ContainsScenario(const Scenario &scenario)
	{
		return (std::find(support_indices_.begin(), support_indices_.begin() + support_subsample_size_, scenario.idx_) != support_indices_.begin() + support_subsample_size_);
	}

	// Aggregate vector 2 into vector 1
	void MergeWith(const SupportSubsample &other)
	{
		for (int i = 0; i < other.support_subsample_size_; i++)
		{
			if (!ContainsScenario(*other.scenarios_[i]))
			{
				Add(*other.scenarios_[i]);
			}
		}
	}

	void Print()
	{

		std::cout << "Support Subsample:\n---------------\n";
		for (int i = 0; i < support_subsample_size_; i++)
		{
			std::cout << "Scenario " << scenarios_[i]->idx_ << ", Obstacle: " << scenarios_[i]->obstacle_idx_ << std::endl;
		}
		std::cout << "---------------\n";
	}

	void PrintUpdate(int bound, const SupportSubsample &removed, int removed_bound, int iterations)
	{
		ROS_INFO_STREAM("SQP (" << iterations << "): Support = " << support_subsample_size_ << "/" << bound << " - Removed: " << removed.support_subsample_size_ << "/" << removed_bound);
	}
};

/*
  We need a functor that can pretend it's const,
  but to be a good random number generator
  it needs mutable state.
*/
namespace Eigen
{
	namespace internal
	{
		template <typename Scalar>
		struct scalar_normal_dist_op
		{
			static boost::mt19937 rng;						 // The uniform pseudo-random algorithm
			mutable boost::normal_distribution<Scalar> norm; // The gaussian combinator

			EIGEN_EMPTY_STRUCT_CTOR(scalar_normal_dist_op)

			template <typename Index>
			inline const Scalar operator()(Index, Index = 0) const { return norm(rng); }
		};

		template <typename Scalar>
		boost::mt19937 scalar_normal_dist_op<Scalar>::rng;

		template <typename Scalar>
		struct functor_traits<scalar_normal_dist_op<Scalar>>
		{
			enum
			{
				Cost = 50 * NumTraits<Scalar>::MulCost,
				PacketAccess = false,
				IsRepeatable = false
			};
		};
	} // end namespace internal
} // end namespace Eigen

/*
  Draw nn samples from a size-dimensional normal distribution
  with a specified mean and covariance
*/
/** Todo: Integrate with data allocation */
inline void SampleMultivariateGaussian(int size, int S, const Eigen::VectorXd &mean, const Eigen::MatrixXd &cov)
{
	Eigen::internal::scalar_normal_dist_op<double> randN;		 // Gaussian functor
	Eigen::internal::scalar_normal_dist_op<double>::rng.seed(1); // Seed the rng

	// Define mean and covariance of the distribution
	// Eigen::VectorXd mean(size);
	// Eigen::MatrixXd covar(size, size);

	// mean << 0, 0;
	// covar << 1, .5,
	// 	.5, 1;

	Eigen::MatrixXd normTransform(size, size);

	Eigen::LLT<Eigen::MatrixXd> cholSolver(cov);

	// We can only use the cholesky decomposition if
	// the covariance matrix is symmetric, pos-definite.
	// But a covariance matrix might be pos-semi-definite.
	// In that case, we'll go to an EigenSolver
	if (cholSolver.info() == Eigen::Success)
	{
		// Use cholesky solver
		normTransform = cholSolver.matrixL();
	}
	else
	{
		// Use eigen solver
		Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(cov);
		normTransform = eigenSolver.eigenvectors() * eigenSolver.eigenvalues().cwiseSqrt().asDiagonal();
	}

	Eigen::MatrixXd samples = (normTransform * Eigen::MatrixXd::NullaryExpr(size, S, randN)).colwise() + mean;

	std::cout << "Mean\n"
			  << mean << std::endl;
	std::cout << "Covariance\n"
			  << cov << std::endl;
	std::cout << "Samples\n"
			  << samples << std::endl;
}
typedef std::vector<std::vector<Eigen::VectorXd>> trajectory_sample; // location per obstacle and time step

namespace Helpers
{

	inline void uniformToGaussian2D(Eigen::Vector2d &uniform_variables)
	{

		// Temporarily safe the first variable
		double temp_u1 = uniform_variables(0);

		// Convert the uniform variables to gaussian via Box-Muller
		uniform_variables(0) = std::sqrt(-2 * std::log(temp_u1)) * std::cos(2 * M_PI * uniform_variables(1));
		uniform_variables(1) = std::sqrt(-2 * std::log(temp_u1)) * std::sin(2 * M_PI * uniform_variables(1));
	}

	inline Eigen::Matrix2d rotationMatrixFromHeading(double heading)
	{

		Eigen::Matrix2d result;
		result << std::cos(heading), std::sin(heading),
			-std::sin(heading), std::cos(heading);

		return result;
	}

	// Class for generating random ints/doubles
	class RandomGenerator
	{
	private:
		std::mt19937 rng_double_;
		std::mt19937 rng_int_;
		std::mt19937 rng_gaussian_;
		std::uniform_real_distribution<> runif_;
		double epsilon_;

	public:
		RandomGenerator(int seed = -1)
		{
			if (seed == -1)
			{
				rng_double_ = std::mt19937(std::random_device{}());	  // Standard mersenne_twister_engine seeded with rd()
				rng_int_ = std::mt19937(std::random_device{}());	  // Standard mersenne_twister_engine seeded with rd()
				rng_gaussian_ = std::mt19937(std::random_device{}()); // Standard mersenne_twister_engine seeded with rd()
			}
			else
			{
				rng_double_ = std::mt19937(seed);	// Standard mersenne_twister_engine seeded with rd()
				rng_int_ = std::mt19937(seed);		// Standard mersenne_twister_engine seeded with rd()
				rng_gaussian_ = std::mt19937(seed); // Standard mersenne_twister_engine seeded with rd()
			}
			runif_ = std::uniform_real_distribution<>(0.0, 1.0);
			epsilon_ = std::numeric_limits<double>::epsilon();
		}

		double Double()
		{
			return (double)runif_(rng_double_); //(double)distribution_(random_engine_) / (double)std::numeric_limits<uint32_t>::max();
		}

		int Int(int max)
		{
			std::uniform_int_distribution<std::mt19937::result_type> new_dist(0, max);
			return new_dist(rng_int_);
		}
		// static std::mt19937 random_engine_;
		// static std::uniform_int_distribution<std::mt19937::result_type> distribution_;

		Eigen::Vector2d BivariateGaussian(const Eigen::Vector2d &mean, const double major_axis, const double minor_axis, double angle)
		{

			Eigen::Matrix<double, 2, 2> A, Sigma, R, SVD;

			// // Get the angle of the path
			// psi = Helpers::quaternionToAngle(path.poses[k].pose);
			R = rotationMatrixFromHeading(angle);

			// Generate uniform random numbers in 2D
			// Eigen::Vector2d uniform_samples = Eigen::Vector2d(Double(), Double());
			double u1, u2;
			do
			{
				u1 = runif_(rng_gaussian_);
			} while (u1 <= epsilon_);
			u2 = runif_(rng_gaussian_);
			Eigen::Vector2d uniform_samples(u1, u2);

			// Convert them to a Gaussian
			uniformToGaussian2D(uniform_samples);

			// Convert the semi axes back to gaussians
			SVD << std::pow(major_axis, 2), 0.0,
				0.0, std::pow(minor_axis, 2);

			// Compute Sigma and cholesky decomposition
			Sigma = R * SVD * R.transpose();
			A = Sigma.llt().matrixL(); // ~sqrt

			return Eigen::Vector2d(
				A(0, 0) * uniform_samples(0) + A(0, 1) * uniform_samples(1) + mean(0),
				A(1, 0) * uniform_samples(0) + A(1, 1) * uniform_samples(1) + mean(1));
		}
	};

	inline double dist(const Eigen::Vector2d &one, const Eigen::Vector2d &two)
	{
		return (two - one).norm();
	}

	inline void ProjectOntoDisc(Eigen::Vector2d &point, const Eigen::Vector2d &disc_origin, const double radius)
	{
		point = disc_origin - (disc_origin - point) / (disc_origin - point).norm() * radius;
	}

	inline double evaluate1DCDF(double value)
	{
		return 0.5 * erfc(-value * M_SQRT1_2);
	}

	// Finds the exponential CDF value at probability p (for a rate of lambda)
	inline double ExponentialQuantile(double lambda, double p)
	{
		return -std::log(1 - p) / lambda;
	}

	template <typename T>
	int sgn(T val)
	{
		return (T(0) < val) - (val < T(0));
	}

	inline bool isTopConstraint(const LinearConstraint2D &constraint, const Eigen::Vector2d &pose)
	{
		// Is y at pose.x > pose.y?
		double y = (constraint.b_(0) - constraint.A_(0, 0) * pose(0)) / constraint.A_(0, 1);
		// std::cout << "y = " << y << ", pose = " << pose(1) << ", is top? " << (y > pose(1)) << std::endl;
		return (y > pose(1));
	}

	inline double quaternionToAngle(const geometry_msgs::Pose &pose)
	{
		double ysqr = pose.orientation.y * pose.orientation.y;
		double t3 = +2.0 * (pose.orientation.w * pose.orientation.z + pose.orientation.x * pose.orientation.y);
		double t4 = +1.0 - 2.0 * (ysqr + pose.orientation.z * pose.orientation.z);

		return atan2(t3, t4);
	}

	inline geometry_msgs::Quaternion angleToQuaternion(double angle)
	{
		tf::Quaternion q = tf::createQuaternionFromRPY(0., 0., angle);
		geometry_msgs::Quaternion result;
		result.x = q.getX();
		result.y = q.getY();
		result.z = q.getZ();
		result.w = q.getW();

		return result;
	}

	inline double quaternionToAngle(geometry_msgs::Quaternion q)
	{

		double ysqr, t3, t4;

		// Convert from quaternion to RPY
		ysqr = q.y * q.y;
		t3 = +2.0 * (q.w * q.z + q.x * q.y);
		t4 = +1.0 - 2.0 * (ysqr + q.z * q.z);
		return std::atan2(t3, t4);
	}

	inline bool transformPose(tf::TransformListener &tf_listener_, const std::string &from, const std::string &to, geometry_msgs::Pose &pose)
	{
		bool transform = false;
		tf::StampedTransform stamped_tf;

		// ROS_DEBUG_STREAM("Transforming from :" << from << " to: " << to);
		geometry_msgs::PoseStamped stampedPose_in, stampedPose_out;
		// std::cout << "from " << from << " to " << to << ", x = " << pose.position.x << ", y = " << pose.position.y << std::endl;
		stampedPose_in.pose = pose;
		// std::cout << " value: " << std::sqrt(std::pow(pose.orientation.x, 2.0) + std::pow(pose.orientation.y, 2.0) + std::pow(pose.orientation.z, 2.0) + std::pow(pose.orientation.w, 2.0)) << std::endl;
		if (std::sqrt(std::pow(pose.orientation.x, 2) + std::pow(pose.orientation.y, 2) + std::pow(pose.orientation.z, 2) + std::pow(pose.orientation.w, 2)) < 1.0 - 1e-9)
		{
			stampedPose_in.pose.orientation.x = 0;
			stampedPose_in.pose.orientation.y = 0;
			stampedPose_in.pose.orientation.z = 0;
			stampedPose_in.pose.orientation.w = 1;
			std::cout << "LMPCC: Quaternion was not normalised properly!" << std::endl;
		}
		//    stampedPose_in.header.stamp = ros::Time::now();
		stampedPose_in.header.frame_id = from;

		// make sure source and target frame exist
		if (tf_listener_.frameExists(to) && tf_listener_.frameExists(from))
		{
			try
			{
				// std::cout << "in transform try " << std::endl;
				// find transforamtion between souce and target frame
				tf_listener_.waitForTransform(from, to, ros::Time(0), ros::Duration(0.02));
				tf_listener_.transformPose(to, stampedPose_in, stampedPose_out);

				transform = true;
			}
			catch (tf::TransformException &ex)
			{
				ROS_ERROR("MPCC::getTransform: %s", ex.what());
			}
		}
		else
		{
			ROS_WARN("MPCC::getTransform: '%s' or '%s' frame doesn't exist, pass existing frame", from.c_str(), to.c_str());
			if (!tf_listener_.frameExists(to))
			{
				ROS_WARN("%s doesn't exist", to.c_str());
			}
			if (!tf_listener_.frameExists(from))
			{
				ROS_WARN("%s doesn't exist", from.c_str());
			}
		}
		pose = stampedPose_out.pose;
		stampedPose_in.pose = stampedPose_out.pose;
		stampedPose_in.header.frame_id = to;

		return transform;
	}

	inline void drawPoint(ROSMarkerPublisher &ros_markers, const Eigen::Vector2d &point)
	{
		// Get a line drawer and set properties
		ROSPointMarker &point_marker = ros_markers.getNewPointMarker("CUBE");
		point_marker.setColor(0., 0., 1.);
		point_marker.setScale(0.2, 0.2, 0.2);

		point_marker.addPointMarker(Eigen::Vector3d(point(0), point(1), 0.2));
	}

	inline void drawLine(ROSMarkerPublisher &ros_markers, const LinearConstraint2D &constraint, int r, int g, int b, double intensity)
	{
		// Get a line drawer and set properties
		ROSLine &line = ros_markers.getNewLine();
		line.setScale(0.1, 0.1);
		double line_length = 100.0;
		// line.setLifetime(1.0 / 20.0);
		line.setColor((double)r * intensity, (double)g * intensity, (double)b * intensity);

		// Loop through the columns of the constraints
		for (int i = 0; i < constraint.b_.rows(); i++)
		{

			// Constraint in z
			if (std::abs(constraint.A_(i, 0)) < 0.01 && std::abs(constraint.A_(i, 1)) < 0.01)
			{
				ROS_WARN("Invalid constraint ignored during visualisation!");
			}

			geometry_msgs::Point p1, p2;
			// Debug!
			double z = 0.2;
			if (r == 1)
				z = 0.3;
			// If we cant draw in one direction, draw in the other
			if (std::abs(constraint.A_(i, 0)) < 0.01)
			{
				p1.x = -line_length;
				p1.y = (constraint.b_(i) + constraint.A_(i, 0) * line_length) / constraint.A_(i, 1);
				p1.z = z;

				p2.x = line_length;
				p2.y = (constraint.b_(i) - constraint.A_(i, 0) * line_length) / constraint.A_(i, 1);
				p2.z = z;
			}
			else
			{

				// Draw the constraint as a line
				p1.y = -line_length;
				p1.x = (constraint.b_(i) + constraint.A_(i, 1) * line_length) / constraint.A_(i, 0);
				p1.z = z;

				p2.y = line_length;
				p2.x = (constraint.b_(i) - constraint.A_(i, 1) * line_length) / constraint.A_(i, 0);
				p2.z = z;
			}

			line.addLine(p1, p2);
		}
	};

	inline void drawLinearConstraints(ROSMarkerPublisher &ros_markers, const std::vector<LinearConstraint2D> &constraints, int r = 0, int g = 1, int b = 0)
	{

		for (u_int k = 0; k < constraints.size(); k++)
		{

			double intensity = std::atan(((double)k + constraints.size() * 0.5) / ((double)constraints.size() * 1.5));

			drawLine(ros_markers, constraints[k], r, g, b, intensity);
		}
	};

	inline void drawLinearConstraints(ROSMarkerPublisher &ros_markers, const lmpcc_msgs::halfspace_array &constraints, int r = 0, int g = 1, int b = 0)
	{

		for (u_int k = 0; k < constraints.halfspaces.size(); k++)
		{

			double intensity = std::atan(((double)k + constraints.halfspaces.size() * 0.5) / ((double)constraints.halfspaces.size() * 1.5));

			LinearConstraint2D mod_constraint;
			mod_constraint.A_ = Eigen::MatrixXd::Zero(1, 2);
			mod_constraint.b_ = Eigen::VectorXd::Zero(1);
			mod_constraint = LinearConstraint2D(mod_constraint.A_, mod_constraint.b_);

			mod_constraint.A_(0) = constraints.halfspaces[k].A[0];
			mod_constraint.A_(1) = constraints.halfspaces[k].A[1];
			mod_constraint.b_(0) = constraints.halfspaces[k].b;
			drawLine(ros_markers, mod_constraint, r, g, b, intensity);
		}
	};

	inline void drawLinearConstraints(ROSMarkerPublisher &ros_markers, const std::vector<LinearConstraint2D> &constraints, const std::vector<int> &indices,
									  int r = 0, int g = 1, int b = 0)
	{

		for (size_t k = 0; k < indices.size(); k++)
		{
			const int &index = indices[k];

			const LinearConstraint2D &constraint = constraints[index];

			// Chance the color for every polygon
			double intensity = std::atan(((double)index + indices.size() * 0.5) / ((double)indices.size() * 1.5));

			drawLine(ros_markers, constraint, r, g, b, intensity);
		}
	};

	inline std::string GetLMPCCDataPath()
	{
		return ros::package::getPath("lmpcc") + "/matlab_exports/data";
	}

	/** Note: blocking! */
	inline void RecordVideo(const std::string &&name)
	{
		const std::string video_command(std::string("bash ~/Videos/util/capture.sh " + name));
		int result = std::system(video_command.c_str()); // Capture video
		assert(result == 1);
	}

	// Use as static to print average run time
	class Benchmarker
	{

	public:
		Benchmarker(const std::string &name, bool record_duration = false)
		{

			name_ = name;
			record_duration_ = record_duration;
			running_ = false;
		}

		// Simpler
		Benchmarker()
		{
		}

		void initialize(const std::string &name, bool record_duration = false)
		{
			name_ = name;
			record_duration_ = record_duration;
		}

		// Print results on destruct
		~Benchmarker()
		{

			double average_run_time = total_duration_ / ((double)total_runs_) * 1000.0;

			std::cout << "Timing Results for [" << name_ << "]\n";
			std::cout << "Average: " << average_run_time << " ms\n";
			std::cout << "Min: " << min_duration_ * 1000.0 << " ms\n";
			std::cout << "Max: " << max_duration_ * 1000.0 << " ms\n";
		}

		void start()
		{
			running_ = true;
			start_time_ = std::chrono::system_clock::now();
		}

		double stop()
		{
			if (!running_)
				return 0.0;

			// Don't time the first 10, there may be some startup behavior
			if (total_runs_ < 10)
			{
				total_runs_++;
				return 0.0;
			}

			auto end_time = std::chrono::system_clock::now();
			std::chrono::duration<double> current_duration = end_time - start_time_;

			if (record_duration_)
				duration_list_.push_back(current_duration.count() * 1000.0); // in ms

			if (current_duration.count() < min_duration_)
				min_duration_ = current_duration.count();

			if (current_duration.count() > max_duration_)
				max_duration_ = current_duration.count();

			total_duration_ += current_duration.count();
			total_runs_++;
			running_ = false;

			last_ = current_duration.count();
			return last_;
		}

		void dataToMessage(std_msgs::Float64MultiArray &msg)
		{

			msg.data.resize(duration_list_.size());

			for (size_t i = 0; i < duration_list_.size(); i++)
				msg.data[i] = duration_list_[i];
		}

		void reset()
		{
			total_runs_ = 0;
			total_duration_ = 0.0;
			max_duration_ = -1.0;
			min_duration_ = 99999.0;
		}

		bool isRunning() { return running_; };

		int getTotalRuns() { return total_runs_; };
		double getLast() { return last_; };

	private:
		std::chrono::system_clock::time_point start_time_;

		double total_duration_ = 0.0;
		double max_duration_ = -1.0;
		double min_duration_ = 99999.0;

		double last_ = -1.0;

		int total_runs_ = 0;

		std::string name_;
		bool record_duration_;
		std::vector<double> duration_list_;
		bool running_ = false;
	};

	class DouglasRachford
	{
	public:
	private:
		Eigen::Vector2d Project(const Eigen::Vector2d &p, const Eigen::Vector2d &delta, const double r, const Eigen::Vector2d &start_pose)
		{
			if (std::sqrt((p - delta).transpose() * (p - delta)) < r)
				return delta - (delta - start_pose) / (std::sqrt((start_pose - delta).transpose() * (start_pose - delta))) * r;
			else
				return p;
		}

		Eigen::Vector2d Reflect(const Eigen::Vector2d &p, const Eigen::Vector2d &delta, const double r, const Eigen::Vector2d &start_pose)
		{
			return 2.0 * Project(p, delta, r, start_pose) - p;
		}

	public:
		Eigen::Vector2d DouglasRachfordProjection(const Eigen::Vector2d &p,
												  const Eigen::Vector2d &delta, const Eigen::Vector2d &anchor,
												  const double r, const Eigen::Vector2d &start_pose)
		{
			return (p + Reflect(Reflect(p, anchor, r, p), delta, r, start_pose)) / 2.0;
		}
	};

	class TriggeredTimer
	{

	public:
		// Duration in s
		TriggeredTimer(const double &duration)
		{
			duration_ = duration;
		}

		void start()
		{
			start_time = std::chrono::system_clock::now();
		}

		double currentDuration()
		{
			auto end_time = std::chrono::system_clock::now();
			std::chrono::duration<double> current_duration = end_time - start_time;

			return current_duration.count();
		}

		bool hasFinished()
		{
			auto end_time = std::chrono::system_clock::now();
			std::chrono::duration<double> current_duration = end_time - start_time;

			return current_duration.count() >= duration_;
		}

	private:
		std::chrono::system_clock::time_point start_time;
		double duration_;
	};

	// Simple class to count the number of simulations
	class SimulationTool
	{
	public:
		SimulationTool(const std::string &topic, double min_time_between, int max_experiments)
			: max_experiments_(max_experiments)
		{
			counter_ = 0;
			finished_ = false;
			reset_sub_ = nh_.subscribe(topic.c_str(), 1, &SimulationTool::ResetCallback, this);
			timer_.reset(new TriggeredTimer(min_time_between));
			timer_->start();
		}

	public:
		void ResetCallback(const std_msgs::Empty &msg)
		{
			// Was this the last simulation (noting that the system is reset initially)
			if (counter_ >= max_experiments_)
			{
				ROS_ERROR_STREAM("Simulation Tool: Done with " << max_experiments_ << " experiments!");
				finished_ = true;
			}

			// Otherwise count
			if (timer_->hasFinished())
			{
				counter_++;
				ROS_WARN_STREAM("Simulation Tool: === Experiment " << counter_ << " / " << max_experiments_ << " ===");
			}
		}

		bool Finished() const { return finished_; };

	private:
		ros::NodeHandle nh_;
		ros::Subscriber reset_sub_;

		int counter_;
		int max_experiments_;

		bool finished_;

		std::unique_ptr<TriggeredTimer> timer_;
	};

	/* T needs to be summable */
	template <class T>
	class StatisticAnalysis
	{
	public:
		StatisticAnalysis()
		{
			data_.reserve(100);
			empty_ = true;
		};

	public:
		bool Empty() { return empty_; };

		void AddData(const T &new_data)
		{
			if (empty_)
				empty_ = false;

			data_.push_back(new_data);
		}

		void Clear()
		{
			data_.clear();
		}

		double Mean()
		{
			if (empty_)
				return 0;

			double sum = 0.;
			for (T &d : data_)
				sum += d;

			return sum / ((double)data_.size());
		}

		double cVaR(double alpha)
		{
			if (empty_)
				return 0;

			// Sort all data (can be slow)
			std::sort(data_.begin(), data_.end());

			// Get the alpha*N highest value
			double threshold = data_[std::floor((1. - alpha) * (double)data_.size())];

			double sum = 0.;
			int n = 0;
			for (T &d : data_)
			{
				// Add if higher than the threshold
				if (d >= threshold)
				{
					sum += d;
					n++;
				}
			}

			return sum / ((double)n);
		}

		/**
		 * @brief How many probability mass is higher than value
		 *
		 * @param value Threshold
		 * @return double (0 - 1)
		 */
		double PAbove(const T &value)
		{
			if (Empty())
				return 0.0;

			int count = 0;
			for (T &d : data_)
			{
				if (d > value)
					count++;
			}

			return ((double)count) / data_.size();
		}

		void Load(const std::string &&file_name)
		{
			DataSaver data_saver;

			std::map<std::string, std::vector<int>> read_data;
			bool success = data_saver.LoadData(file_name, read_data);

			if (!success)
				return;

			data_ = read_data["support"];
			empty_ = false;
		}

		void Save(const std::string &&file_name)
		{
			DataSaver data_saver;

			for (T &data_point : data_)
				data_saver.AddData("support", data_point);

			data_saver.SaveData(file_name);
		}

	private:
		std::vector<T> data_;
		bool empty_;
	};

/********** Some Fancy timing classes for profiling (from TheCherno) ***********/
#define PROFILER 1
#if PROFILER
#define PROFILE_SCOPE(name) Helpers::InstrumentationTimer timer##__LINE__(name)
#define PROFILE_FUNCTION() PROFILE_SCOPE(__FUNCTION__)
#define PROFILE_AND_LOG(name) \
	LMPCC_INFO(name);         \
	Helpers::InstrumentationTimer timer##__LINE__(name)
#else
#define PROFILE_SCOPE(name)
#define PROFILE_FUNCTION()
#endif

	struct ProfileResult
	{
		std::string Name;
		long long Start, End;
		uint32_t ThreadID;
	};

	struct InstrumentationSession
	{
		std::string Name;
	};

	class Instrumentor
	{
	private:
		InstrumentationSession *m_CurrentSession;
		std::ofstream m_OutputStream;
		int m_ProfileCount;
		std::mutex m_lock;

	public:
		Instrumentor()
			: m_CurrentSession(nullptr), m_ProfileCount(0)
		{
		}

		void BeginSession(const std::string &name, const std::string &filepath = "lmpcc_profiler.json")
		{
			std::string full_filepath = ros::package::getPath("lmpcc") + "/profiling/" + filepath;
			m_OutputStream.open(full_filepath);
			WriteHeader();
			m_CurrentSession = new InstrumentationSession{name};
		}

		void EndSession()
		{
			WriteFooter();
			m_OutputStream.close();
			delete m_CurrentSession;
			m_CurrentSession = nullptr;
			m_ProfileCount = 0;
		}

		void WriteProfile(const ProfileResult &result)
		{
			std::lock_guard<std::mutex> lock(m_lock);

			if (m_ProfileCount++ > 0)
				m_OutputStream << ",";

			std::string name = result.Name;
			std::replace(name.begin(), name.end(), '"', '\'');

			m_OutputStream << "{";
			m_OutputStream << "\"cat\":\"function\",";
			m_OutputStream << "\"dur\":" << (result.End - result.Start) << ',';
			m_OutputStream << "\"name\":\"" << name << "\",";
			m_OutputStream << "\"ph\":\"X\",";
			m_OutputStream << "\"pid\":0,";
			m_OutputStream << "\"tid\":" << result.ThreadID << ",";
			m_OutputStream << "\"ts\":" << result.Start;
			m_OutputStream << "}";

			m_OutputStream.flush();
		}

		void WriteHeader()
		{
			m_OutputStream << "{\"otherData\": {},\"traceEvents\":[";
			m_OutputStream.flush();
		}

		void WriteFooter()
		{
			m_OutputStream << "]}";
			m_OutputStream.flush();
		}

		static Instrumentor &Get()
		{
			static Instrumentor *instance = new Instrumentor();
			return *instance;
		}
	};

	class InstrumentationTimer
	{
	public:
		InstrumentationTimer(const char *name)
			: m_Name(name), m_Stopped(false)
		{
			m_StartTimepoint = std::chrono::system_clock::now();
		}

		~InstrumentationTimer()
		{
			if (!m_Stopped)
				Stop();
		}

		void Stop()
		{
			auto endTimepoint = std::chrono::system_clock::now();

			long long start = std::chrono::time_point_cast<std::chrono::microseconds>(m_StartTimepoint).time_since_epoch().count();
			long long end = std::chrono::time_point_cast<std::chrono::microseconds>(endTimepoint).time_since_epoch().count();

			uint32_t threadID = std::hash<std::thread::id>{}(std::this_thread::get_id());
			Instrumentor::Get().WriteProfile({m_Name, start, end, threadID});

			m_Stopped = true;
		}

	private:
		const char *m_Name;
		std::chrono::system_clock::time_point m_StartTimepoint;
		bool m_Stopped;
	};
};

#endif