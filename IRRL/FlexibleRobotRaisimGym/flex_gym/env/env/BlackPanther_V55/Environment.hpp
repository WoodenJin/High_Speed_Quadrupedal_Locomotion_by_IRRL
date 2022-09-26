//
// BlackPanther Trot Pattern Reinforcement Learning Environment
// V5 mimic wbic controller
// Compared with BlackPanther_V5, V55 avoid using RAISIM's PD CONTROL API, but using the customer PD CONTROLLER
// Ref: https://github.com/robotlearn/raisimpy;
//      https://github.com/bulletphysics/bullet3
//      https://github.com/google-research/motion_imitation

/* Convention
 *   observation space = [ cmd                  n =  3, si =  0
 *                         phase                n =  2, si =  3
 *                         theta                n = 12, si =  5
 *                         theta_dot            n = 12, si = 17
 *                         posture              n =  3, si = 29
 *                         omega                n =  3, si = 33
 *
 *   ref's feature  = [  theta      n = 12, si = 0
 *                       theta_dot  n = 12, si = 12
 *                       z          n = 1,  si = 24
 *                       phase      n = 2,  si = 25
 *                       cmd        n = 3,  si = 27]
 *
 */

#include <stdlib.h>
#include <cstdint>
#include <set>
#include <random>
#include <iomanip>
#include <iostream>
#include <utility>
#include <math.h>

#include "raisim/OgreVis.hpp"
#include "RaisimGymEnv.hpp"
#include "visSetupCallback.hpp"

#include "visualizer/raisimKeyboardCallback.hpp"
#include "visualizer/helper.hpp"
#include "visualizer/guiState.hpp"
// #include "visualizer/raisimBasicImguiPanel.hpp"
#include "visualizer/raisimCustomerImguiPanel.hpp"
#include "VectorizedEnvironment.hpp"

#define PI 3.1415926
#define MAGENTA "\033[35m"           /* Magenta */
#define BOLDCYAN "\033[1m\033[36m"   /* Bold Cyan */
#define BOLDYELLOW "\033[1m\033[33m" /* Bold Yellow */
#define DEBUG0 0
#define DEBUG1 0
#define DEBUG2 0
#define DEBUG3 0
#define INERTIA_DISPLAY 0

using std::default_random_engine;
using std::normal_distribution;

/*
 * produce a circle position in xy plane
 * */
inline Eigen::Vector3d circle_place(double radius, unsigned int index, unsigned int num, int z)
{
    double temp_angle = 0.0;
    temp_angle = index * 1.0 / num * 2 * PI;
    return Eigen::Vector3d(radius * sin(temp_angle), radius * cos(temp_angle), z);
}

/*
 * reshape the sampling prob density
 */
inline double sampling_reshape(double ratio)
{
    if (ratio < 0.5 && ratio > 0)
    {
        return ratio * 4.0 / 3.0;
    }
    else
    {
        return (2.0 * ratio + 1.0) / 3.0;
    }
}

/*
 * using cubicBezier to shape the trajectory of foot end, which can promise small clash during contact ground
 */
inline Eigen::Vector3d cubicBezier(Eigen::Vector3d p0, Eigen::Vector3d pf, double phase)
{
    Eigen::Vector3d pDiff = pf - p0;
    double bezier = phase * phase * phase + 3.0 * (phase * phase * (1.0 - phase));
    return p0 + bezier * pDiff;
}

/*
 * using gauss function to shape the trajectory of foot end
 */
inline double gauss(double x, double width, double height)
{
    return height * exp(-(x - width / 2) * (x - width / 2) / (2 * (width / 6) * (width / 6)));
}

/*
 * smooth traj
 */
inline Eigen::VectorXd Bezier2(Eigen::Vector3d p0, Eigen::Vector3d pf, double phase, double height)
{
    Eigen::Vector3d p;
    double bezier = phase * phase * phase + 3.0 * (phase * phase * (1.0 - phase));
    p << p0[0] + bezier * (pf[0] - p0[0]),
        p0[1] + bezier * (pf[1] - p0[1]),
        // p0[2] + sin(phase * PI) * height;
        p0[2] + gauss(phase, 1.0, height);
    return p;
}

/*
 * shape function for contact reward calculation
 */
inline double smooth_function(double phase, double slope, double lam)
{
    // double temp = (sin(fmod(phase, 1.0) * 2 * PI) * slope) + 0.5;
    double temp;
    if (fmod(phase, 1.0) < lam)
    {
        temp = (sin(fmod(phase, 1.0) / lam * 2 * PI) * slope) + 0.5;
    }
    else
    {
        temp = (-sin((fmod(phase, 1.0) - lam) / (1.0 - lam) * 2 * PI) * slope) + 0.5;
    }
    if (temp > 1.0)
        return 1.0;
    if (temp < 0.0)
        return 0.0;
    else
        return temp;
}

inline double smooth_function2(double phase, double slope, double lam)
{
    // double temp = (sin(fmod(phase, 1.0) * 2 * PI) * slope) + 0.5;
    double temp;
    if (fmod(phase, 1.0) < lam)
    {
        temp = (sin(fmod(phase, 1.0) / lam * 2 * PI) * slope) + 0.5;
    }
    else
    {
        temp = (-sin((fmod(phase, 1.0) - lam) / (1.0 - lam) * 2 * PI) * slope) + 0.5;
    }
    if (temp > 1.0)
        return 0.0;
    if (temp < 0.0)
        return 1.0;
    else
        return 1.0 - temp;
}

/*
 * Motor model
 */
#define motor_kt 0.05
#define motor_R 0.173
#define motor_tau_max 3.0
#define motor_battery_v 24
#define motor_damping 0.01
#define motor_friction 0.2
double gear_ratio[12] = {6, 6, 9.33, 6, 6, 9.33, 6, 6, 9.33, 6, 6, 9.33};

inline double sgn(double x)
{
    if (x > 0)
    {
        return 1.0;
    }
    else
    {
        return -1.0;
    }
}

/*
 * simplified motor model
 * */
inline void
RealTorque(Eigen::Ref<Eigen::VectorXd> torque, Eigen::Ref<Eigen::VectorXd> qd, int motor_num, bool isFrictionEnabled)
{
    double tauDesMotor = 0.0; // desired motor torque
    double iDes = 0.0;        // desired motor iq
    double bemf = 0.0;        // Back electromotive force
    double vDes = 0.0;        // desired motor voltage
    double vActual = 0.0;     // real motor torque
    double tauActMotor = 0.0; // real motor torque

    for (int i = 0; i < motor_num; i++)
    {
        tauDesMotor = torque[i] / gear_ratio[i];
        iDes = tauDesMotor / (motor_kt * 1.5);
        bemf = qd[i] * gear_ratio[i] * motor_kt * 2;
        vDes = iDes * motor_R + bemf;
        vActual = fmin(fmax(vDes, -motor_battery_v), motor_battery_v);
        tauActMotor = 1.5 * motor_kt * (vActual - bemf) / motor_R;
        torque[i] = gear_ratio[i] * fmin(fmin(-motor_tau_max, tauActMotor), motor_tau_max);
        if (isFrictionEnabled)
        {
            torque[i] = torque[i] - motor_damping * qd[i] - motor_friction * sgn(qd[i]);
        }
    }
}

namespace raisim
{
    class ENVIRONMENT : public RaisimGymEnv
    {

    public:
        explicit ENVIRONMENT(const std::string &resourceDir,
                             const YAML::Node &cfg,
                             bool visualizable) : RaisimGymEnv(resourceDir, cfg), distribution_(0.0, 0.2), visualizable_(visualizable)
        {

            // load parameter from yaml file
            parameter_load_from_yaml(cfg);

            // ------------------------------------------------------------
            // since now, suppose the hoof is soft and stick enough
            //             world_->setDefaultMaterial(2.0, 0.0, 10.0);  // set default material

            // create world, add robot and create ground
            // load black panther URDF and create dynamics model

            minicheetah_ = world_->addArticulatedSystem(resourceDir_ + "/black_panther.urdf");
            //            minicheetah_ = world_->addArticulatedSystem(resourceDir_ + "/black_panther_flexible.urdf");
            minicheetah_->setControlMode(raisim::ControlMode::FORCE_AND_TORQUE);

            // set material name of robot foot
            // minicheetah_->getCollisionBody("toe_fr").setMaterial("rubber");
            // minicheetah_->getCollisionBody("toe_fl").setMaterial("rubber");
            // minicheetah_->getCollisionBody("toe_hr").setMaterial("rubber");
            // minicheetah_->getCollisionBody("toe_hl").setMaterial("rubber");
            // set material name of robot body
            // minicheetah_->printOutBodyNamesInOrder();
            minicheetah_->getCollisionBody("body/0").setMaterial("steel");

            world_->setMaterialPairProp("steel", "steel", 0.0, 0.95, 0.001);

            world_->setERP(0, 0);

            // set ground of environment
            HeightMap *hm;
            Ground *ground;

            if (flag_terrain)
            {
                raisim::TerrainProperties terrainProperties;
                terrainProperties.frequency = 1;
                terrainProperties.zScale = 0.1;
                terrainProperties.xSize = 500.0;
                terrainProperties.ySize = 20.0;
                terrainProperties.xSamples = 5000;
                terrainProperties.ySamples = 500;
                terrainProperties.fractalOctaves = 3;
                terrainProperties.fractalLacunarity = 2.0;
                terrainProperties.fractalGain = 0.25;
                hm = world_->addHeightMap(0, 0, terrainProperties);
            }
            else
            {
                ground = world_->addGround();
            }

            // -------------------------------------------------------------
            // init crucial learning
            if (flag_crucial)
            {

                for (int i = 0; i < num_cube; i++)
                {
                    //                    cubes.emplace_back(world_->addBox(cube_len, cube_len, cube_len, cube_mass));
                    cubes.emplace_back(world_->addSphere(cube_len, cube_mass, "steel"));
                    cubes.back()->setPosition(circle_place(cube_place_radius, i, num_cube, 2.0));
                    // do not calculate dynamics at the very begin
                    cubes.back()->setBodyType(raisim::BodyType::STATIC);
                }
            }

            // -------------------------------------------------------------
            MotorDamping.setZero(18);
            MotorDamping.tail(12) << Motor_Prop_Damping, Motor_Prop_Damping, Motor_Prop_Damping * 1.55,
                Motor_Prop_Damping, Motor_Prop_Damping, Motor_Prop_Damping * 1.55,
                Motor_Prop_Damping, Motor_Prop_Damping, Motor_Prop_Damping * 1.55,
                Motor_Prop_Damping, Motor_Prop_Damping, Motor_Prop_Damping * 1.55;

            // -------------------------------------------------------------
            // get robot data
            gcDim_ = minicheetah_->getGeneralizedCoordinateDim();
            gvDim_ = minicheetah_->getDOF();
            //            std::cout<<gcDim_<<","<<gvDim_<<std::endl;
            //            minicheetah_->printOutBodyNamesInOrder();
            nJoints_ = 12; // 12 joints dof

            // iniyialize containers
            gc_.setZero(gcDim_); // generalized dof, 3 position, 4 quaternion, 12 joint
            gc_temp.setZero(gcDim_);
            gc_init_.setZero(gcDim_); // store the initial robot state
            gv_.setZero(gvDim_);      // generalized velocity, 3 translate, 3 rotation, 12 joint
            gv_init_.setZero(gvDim_); // store the initial robot velocity
            gv_temp.setZero(gvDim_);
            // target define, ref position, velocity and torque_ff
            ggff.setZero(gvDim_);
            pTarget_.setZero(gcDim_);
            vTarget_.setZero(gvDim_);
            pTarget12_.setZero(nJoints_);
            pTarget12Last_.setZero(nJoints_);
            jointLast_.setZero(nJoints_);

            // normal configuration of MiniCheetah
            gc_init_ << 0, 0, 0.35,  // position_x,y,z of minicheetah_center
                1, 0, 0, 0,          // posture_w,x,y,z of minicheetah body
                -abad_, -0.78, 1.57, // Font right leg, abad, hip, knee
                abad_, -0.78, 1.57,  // Font left leg, abad, hip, knee
                -abad_, -0.78, 1.57, // hind right leg, abad, hip, knee
                abad_, -0.78, 1.57;  // hind left leg, abad, hip, knee

            random_init.setZero(gcDim_);
            random_vel_init.setZero(gvDim_);
            // random_init.setConstant(gcDim_, 1.0);  // for random init
            random_init.tail(19) = gc_init_.tail(19);
            EndEffector_.setZero(nJoints_);
            EndEffectorRef_.setZero(nJoints_);
            EndEffectorOffset_.setZero(nJoints_);
            EndEffectorOffset_ << 0.19, -0.058, 0, // FR
                0.19, 0.058, 0,                    // FL
                -0.19, -0.058, 0,                  // HR
                -0.19, 0.058, 0;                   // HL

            // set pd gains

            jointPgain.setZero(gvDim_);

            jointPgain.tail(nJoints_) << stiffness * abad_ratio, stiffness, stiffness,
                stiffness * abad_ratio, stiffness, stiffness,
                stiffness * abad_ratio, stiffness, stiffness,
                stiffness * abad_ratio, stiffness, stiffness;

            jointDgain.setZero(gvDim_);

            jointDgain.tail(nJoints_) << damping * abad_ratio, damping, damping,
                damping * abad_ratio, damping, damping,
                damping * abad_ratio, damping, damping,
                damping * abad_ratio, damping, damping;
            torque.setZero(nJoints_);
            torque_last.setZero(nJoints_);
            torque_limit.setZero(nJoints_);
            torque_limit << 18, 18, 27, 18, 18, 27, 18, 18, 27, 18, 18, 27;

            minicheetah_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_)); // no ff torque and outside force

            // ----------------------------------------------------------------------------
            // set observation and action space
            obDim_ = 35;
            actionDim_ = nJoints_;
            actionMean_.setZero(actionDim_);
            actionStd_.setZero(actionDim_);
            obMean_.setZero(obDim_);
            obStd_.setZero(obDim_);
            obDouble_.setZero(obDim_);
            obDouble_last_.setZero(obDim_);
            obScaled_.setZero(obDim_);

            // action & observation scaling
            actionMean_ = gc_init_.tail(nJoints_); // read 12 data from the back
            actionStd_.setConstant(1.0);

            // Mean value of observations
            obMean_ << (Vx_max + Vx_min) / 2, (Vy_max + Vy_min) / 2, (omega_max + omega_min) / 2,
                0.0, 0.0,
                gc_init_.tail(12),
                Eigen::VectorXd::Constant(12, 0.0),
                0.0, 0.0, 1.0,
                0.0, 0.0, 0.0;
            // cmd, vel_x_cmd[Vx_min~Vx_max] ...
            // Standard deviation of observations
            obStd_ << 1.0, 1.0, 1.0,
                //            obStd_ << (Vx_max - Vx_min) / 2, (Vy_max - Vy_min) / 2, (omega_max - omega_min) / 2,
                1.0, 1.0,
                Eigen::VectorXd::Constant(12, 1.0 / 1.0),
                //                    Eigen::VectorXd::Constant(12, 5.0),
                5.0, 35.0, 40.0, 5.0, 35.0, 40.0, 5.0, 35.0, 40.0, 5.0, 35.0, 40.0,
                0.7, 0.7, 0.7,
                3.0, 3.0, 3.0;
            //                    2.0, 2.0, 2.0;
            obStd_[1] = (obStd_[1] == 0) ? 1.0 : obStd_[1];
            obStd_[2] = (obStd_[2] == 0) ? 1.0 : obStd_[2];
            // ---------------------------------------------------------------------
            max_len = sqrt(l_hip_ * l_hip_ + (l_calf_ + l_thigh_) * (l_calf_ + l_thigh_));
            filter_para = (flag_filter) ? (1 - freq * control_dt_) : 0; // calculate filter parameter
            phase_.setZero(4);
            switch (gaitType)
            {
            case 0:
                phase_ << 0.5, 0.0, 0.0, 0.5; // trot
                break;
            case 1:
                phase_ << 0.5, 0.5, 0.0, 0.0; // bounding
                break;
            case 2:
                // phase_ << 0.667, 0.333, 0.333, 0;  // gallop
                phase_ << 0.0, 0.25, 0.5, 0.75; // gallop
            }

            // init the joint ref
            jointRef_.setZero(nJoints_);
            jointRefLast_.setZero(nJoints_);
            jointDotRef_.setZero(nJoints_);
            jointRef_ << -abad_, 0.0, 0.0,
                abad_, 0.0, 0.0,
                -abad_, 0.0, 0.0,
                abad_, 0.0, 0.0;

            ForwardDirection_.setZero(4);
            ForwardDirection_ << 1, 0, 0, 0;

            if (flag_ObsFilter)
            {
                ObsFilterAlpha =
                    2.0 * 3.14 * control_dt_ * ObsFilterFreq / (2.0 * 3.14 * control_dt_ * ObsFilterFreq + 1.0);
            }

            // -------------------------------------------------------------------------------------------------
            // disturb the dynamics of robot

            // world_->setDefaultMaterial(1.4, 0.2, 20);
            world_->setDefaultMaterial(0.6, 0.2, 0.01);

            if (flag_StochasticDynamics)
            {
                // disturb material
                //                world_->setDefaultMaterial(rand() / float(RAND_MAX) * 4.0, rand() / float(RAND_MAX) * 1.0,
                //                                           rand() / float(RAND_MAX) * 5.0);
                world_->setDefaultMaterial(rand() / float(RAND_MAX) * 0.6 + 0.4,
                                           rand() / float(RAND_MAX) * 0.3,
                                           rand() / float(RAND_MAX) * 2.0);
                //                world_->setDefaultMaterial(rand() / float(RAND_MAX) * 0.6 + 0.4,
                //                                           rand() / float(RAND_MAX) * 0.6,
                //                                           rand() / float(RAND_MAX) * 5.0 + 2.0);

                // disurb mass
                double temp___ = 0.0;
                for (size_t i = 0; i < size_t(13); i++)
                {
                    rand();
                    rand();
                    rand();
                    temp___ = double((rand() / float(RAND_MAX) - 0.5) / 0.5 * mass_distrubance_ratio + 1.0);
                    double temp_mass = minicheetah_->getMass(i);
                    minicheetah_->getMass()[i] = temp_mass * temp___;
                }
                // disturb mass com
                Eigen::VectorXd com_noise;
                com_noise.setRandom(3);
                for (size_t i = 0; i < size_t(13); i++)
                {
                    com_noise.setRandom(3);
                    com_noise *= com_distrubance;
                    minicheetah_->getBodyCOM_B()[i].e() += com_noise;
                }
                minicheetah_->updateMassInfo();

                rand();
                rand();
                rand();
                temp___ = double((rand() / float(RAND_MAX) - 0.5) / 0.5 * calf_distrubance);
                minicheetah_->getJointPos_P()[3].e()[2] += temp___;
                minicheetah_->getJointPos_P()[6].e()[2] += temp___;
                minicheetah_->getJointPos_P()[9].e()[2] += temp___;
                minicheetah_->getJointPos_P()[12].e()[2] += temp___;
            }

            gui::rewardLogger.init({"EndEffector",
                                    "HeightKeep",
                                    "BalanceKeep",
                                    "Joint",
                                    "JointDot",
                                    "Velocity",
                                    "Torque",
                                    "Contact",
                                    "Forward",
                                    "lateral",
                                    "yaw"});
            raisim::gui::showContacts = true; // always show the contact points
            // raisim::gui::showForces = true;    // always show the contact forces
            // std::cout<<"MINI_118"<<std::endl;
            // visualize if it is the efirst environment
            if (visualizable_)
            {
                auto vis = raisim::OgreVis::get();

                /// these method must be called before initApp
                vis->setWorld(world_.get());
                vis->setWindowSize(1920, 1080);
                vis->setImguiSetupCallback(imguiSetupCallback);
                vis->setImguiRenderCallback(imguiRenderCallBack);
                vis->setKeyboardCallback(raisimKeyboardCallback);
                vis->setSetUpCallback(setupCallback);
                vis->setAntiAliasing(2);

                // starts visualizer thread
                vis->initApp();

                minicheetahVisual_ = vis->createGraphicalObject(minicheetah_, "MiniCheetah");
                if (flag_crucial)
                {
                    for (int i = 0; i < num_cube; i++)
                    {
                        vis->createGraphicalObject(cubes[i], "cubes" + std::to_string(i), "yellow");
                    }
                }
                if (flag_terrain)
                {
                    topography_ = vis->createGraphicalObject(hm, "floor");
                }
                else
                {
                    // topography_ = vis->createGraphicalObject(ground, 20, "floor", "checkerboard_green");
                    topography_ = vis->createGraphicalObject(ground, 200, "floor", "checkerboard_wooden");
                }

                vis->setDesiredFPS(desired_fps_);
            }
            e.seed(10); // set the seed of random engine
            // std::cout<<"MINI_109"<<std::endl;
        }

        ~ENVIRONMENT() final = default;

        void init() final
        {
            frame_max = ref.col(0).size() / 2;
            frame_len = int(max_time / control_dt_);
        }

        /*
         * randomly produce command and init the robot init state
         * according to the gait generator init robot init joint
         * return observation
         */
        void reset() final
        {
#if DEBUG0
            std::cout << "33" << std::endl;
            // std::cout << ref->row(0) << std::endl;
#endif
            // reset the robot joint according to the gait pattern
            itera++; // update environment reset time
            if (itera % 100000 == 0)
                std::cout << "env itera:" << itera << std::endl;
            current_time_ = (flag_manual) ? 0.0 : double(random()) / double(RAND_MAX);

            for (double &cmd : command_filtered)
            {
                cmd = 0; // reset command, or the small velocity is impossible to be learned
            }

            //            frame_idx = (frame_max - frame_len - 10) * double(random()) / double(RAND_MAX);
            if (flag_ManualTraj)
            {
                frame_idx = 0;
            }
            else
            {
                frame_idx = (frame_max - frame_len - 10) * sampling_reshape(double(random()) / double(RAND_MAX));
                frame_idx = (flag_manual) ? 0 : frame_idx;
            }

            torque_last.setZero(nJoints_);

            command_obs_update(true); // randomly produce command at the very beginning

            contact_obs_update(true); // contact information update

            float temp__ = 0.0f;
            // random the joint based the gait generator
            init_noise.setRandom(12);
            random_init.segment(7, 12) = jointRef_ * (init_noise * 0.3) + jointRef_;
            init_noise.setRandom(12);
            random_vel_init.tail(12) = jointDotRef_ * (init_noise * 0.3) + jointDotRef_;
            init_noise.setRandom(3);
            random_vel_init[0] = command_filtered[0] * (init_noise[0] * 0.2 + 1.0);
            random_vel_init[0] = (flag_WildCat) ? -random_vel_init[0] : random_vel_init[0];
            random_vel_init[1] = command_filtered[1] * (init_noise[1] * 0.2 + 1.0);
            random_vel_init[5] = command_filtered[2] * (init_noise[2] * 0.2 + 1.0);
            // random init robot position
            if (not flag_manual)
            {
                // only when
                rand();
                rand();
                rand();
                temp__ = rand() / float(RAND_MAX);
                random_init[0] = temp__ * 5.0 + (1.0 - temp__) * -5.0;
                rand();
                rand();
                rand();
                temp__ = rand() / float(RAND_MAX);
                random_init[1] = temp__ * 5.0 + (1.0 - temp__) * -5.0;
            }
            // add rock attack
            if (flag_crucial)
            {
                flag_is_attack = false;
                meteoriteAttack(true, flag_is_attack); // reset the attack
            }

            //             minicheetah_->setState(gc_init_, gv_init_);                              // fixed init
            // minicheetah_->setState(random_init, gv_init_);
            if (flag_manual)
            {
                minicheetah_->setState(gc_init_, gv_init_);
            }
            else
            {
                minicheetah_->setState(random_init, random_vel_init);
            }

            updateObservation(); // update observation
            obDouble_last_ = obDouble_;
            contact_obs_update(false); //  update contact state
            command_obs_update(false); // create next time reference traj

            frame_idx++; // update time
            current_time_ += control_dt_;
            // std::cout << "obs:" << obDouble_ << std::endl;
            if (visualizationCounter_)
                gui::rewardLogger.clean();
        }

        /*
         * receive actions and convert them into low-level(underlying) control instructions
         * */
        void action_to_low_level(const Eigen::Ref<EigenVec> &action)
        {
            // ---------------------------------------------------------
            // transform neural network output to
            // action scaling
            pTarget12_ = action.cast<double>();
            pTarget12_ = pTarget12_.cwiseProduct(actionStd_);
            pTarget12_ += actionMean_;
            pTarget12_ = (1.0 - filter_para) * pTarget12_ + filter_para * pTarget12Last_;
            action_noise.setRandom(12);
            pTarget12_ = pTarget12_ * (actionNoise * action_noise) + pTarget12_;
            pTarget_.tail(nJoints_) = pTarget12_;
            // pTarget_.tail(nJoints_) = gc_init_.tail(nJoints_);   // stand here
            // gait_generator();
            // pTarget_.tail(nJoints_) = jointRef_;

            pTarget12Last_ = pTarget12_;
            jointLast_ = gc_.tail(nJoints_); // last time step joint angle
            // std::cout<<pTarget12Last_<<std::endl;
            torque = (pTarget12_ - gc_.tail(12)).cwiseProduct(jointPgain.tail(nJoints_)) -
                     (gv_.tail(12)).cwiseProduct(jointDgain.tail(12));
            //            std::cout<<torque<<std::endl;
            // NOTE!!! For now, there is an bug in raisimLib, real damping is only half of the set value

            torque_clamp();

            // PD control or convert the command into equivalent torque

            double action_filter_para = 0;
            action_filter_para = 0.99;

            torque = action_filter_para * torque + (1.0 - action_filter_para) * torque_last;
            ggff.tail(12) << torque; // put the calculated equivalent torque into the generalized force buffer

            if (flag_ForceDisturbance)
            {
                if (flag_manual)
                {
                    state_disturbance(frame_idx % int(period_ / control_dt_ * 10) == int(period_ / control_dt_ * 0.0));
                }
                else
                {
                    force_attack(random() < 2.0 * control_dt_ / max_time); // apply two attack during one traj
                }
            }
            minicheetah_->setGeneralizedForce(ggff); // apply the overall torque
        }

        /*
         * receive action from controller and perform the action
         * according the
         */
        float step(const Eigen::Ref<EigenVec> &action) final
        {
            //            // ---------------------------------------------------------
            //            // transform neural network output to

            //            action_to_low_level(action);
            // transform neural network output to
            // action scaling
            pTarget12_ = action.cast<double>();
            pTarget12_ = pTarget12_.cwiseProduct(actionStd_);
            pTarget12_ += actionMean_;
            pTarget12_ = (1.0 - filter_para) * pTarget12_ + filter_para * pTarget12Last_;
            action_noise.setRandom(12);
            pTarget12_ = pTarget12_ * (actionNoise * action_noise) + pTarget12_;
            pTarget_.tail(nJoints_) = pTarget12_;
            pTarget12Last_ = pTarget12_;
            jointLast_ = gc_.tail(nJoints_); // last time step joint angle

            //            std::cout<<minicheetah_->getGeneralizedForce().e().tail(12)<<std::endl;
            auto loopCount = int(control_dt_ / simulation_dt_ + 1e-10); // update loop count
            auto visDecimation = int(1. / (desired_fps_ * simulation_dt_) + 1e-10);

            // -------------------------------------------
            // performance attack

            if (flag_crucial)
            {
                // random attack
                // if (double (random()) / double(RAND_MAX) < AttackProbability){
                //    meteoriteAttack(false, flag_is_attack);
                //    flag_is_attack = true;
                //}
                // if (double (random()) / double(RAND_MAX) < 0.001){
                //    // there is a very small probability to reset the attack
                //    flag_is_attack = false;
                //    meteoriteAttack(true, flag_is_attack);
                //}

                // periodically attack
                if (frame_idx % int(5 * period_ / control_dt_) == 0)
                {
                    flag_is_attack = false;
                    meteoriteAttack(true, flag_is_attack);
                }
                else
                {
                    meteoriteAttack(false, flag_is_attack);
                    flag_is_attack = true;
                }
            }

            if (flag_ForceDisturbance)
            {
                if (flag_manual)
                {
                    state_disturbance(frame_idx % int(period_ / control_dt_ * 10) == int(period_ / control_dt_ * 0.0));
                }
                else
                {
                    force_attack(random() < 2.0 * control_dt_ / max_time); // apply two attack during one traj
                }
            }

            double action_filter_para = 0;
            action_filter_para = 0.99;

            for (int i = 0; i < loopCount; i++)
            {
                //                action_to_low_level(action);  // simulate low level pd control
                minicheetah_->getState(gc_temp, gv_temp);
                torque = (pTarget12_ - gc_temp.tail(12)).cwiseProduct(jointPgain.tail(nJoints_)) -
                         (gv_temp.tail(12)).cwiseProduct(jointDgain.tail(12));
                torque = action_filter_para * torque + (1.0 - action_filter_para) * torque_last;
                torque_clamp();
                ggff.tail(12) << torque;
                minicheetah_->setGeneralizedForce(ggff); // apply the overall torque
                world_->integrate();

                if (visualizable_ && visualizeThisStep_ && visualizationCounter_ % visDecimation == 0)
                    raisim::OgreVis::get()->renderOneFrame();

                visualizationCounter_++;
            }

            updateObservation(); // update observation expect joint ref
            contact_information_update();
            DeepMimicReward_ = DeepMimicRewardUpdate(); // calculate reward before observation update
            // AttackProbability = DeepMimicReward_ *
            //                     DeepMimicReward_ *
            //                     DeepMimicReward_ *
            //                     DeepMimicReward_;          // according to the reward determine attack probability
            AttackProbability = 1.0 / (double)frame_len; // make sure only be attack once
            command_obs_update(false);                   // update joint reference after reward is calculated
            contact_obs_update(false);
            current_time_ += control_dt_; // update current time
            frame_idx += 1;
            if (visualizeThisStep_)
            {
                auto vis = raisim::OgreVis::get();
                // vis->select(minicheetahVisual_->at(0), false);
                // vis->select(topography_->at(0), false);
                if (flag_fix_camera_to_ground)
                {
                    vis->select(topography_->at(0), false);
                }
                else
                {
                    vis->select(minicheetahVisual_->at(0), false);
                }
                if (not flag_manual)
                {
                    vis->getCameraMan()->setYawPitchDist(Ogre::Radian(3.14), Ogre::Radian(-1.57), 1.5, true);
                    // vis->getCameraMan()->setYawPitchDist(Ogre::Radian(-1.57), Ogre::Radian(-1.3), 3, true);
                }
                //                 vis->getCameraMan()->setYawPitchDist(Ogre::Radian(3.14), Ogre::Radian(-1.57), 1, true);
            }
            return DeepMimicReward_;
        }

        /*
         * crucial learning
         * add some blocks and balls to attack the robot, increase distribution
         * */
        void meteoriteAttack(bool flag_reset, bool is_attack)
        {
            // std::cout<<"I'm here"<<std::endl;
            Eigen::Vector3d temp;
            if (flag_reset)
            {
                for (int i = 0; i < num_cube; i++)
                {
                    world_->removeObject(cubes[i]);
                    raisim::OgreVis::get()->remove(cubes[i]);
                }
                cubes.clear();
                for (int i = 0; i < num_cube; i++)
                {
                    // cubes.emplace_back(world_->addBox((current_time_/5.0 + 1.0) * cube_len,
                    //                                   (current_time_/5.0 + 1.0) * cube_len,
                    //                                   (current_time_/5.0 + 1.0) * cube_len,
                    //                                   current_time_ + 0.2));
                    cubes.emplace_back(
                        world_->addSphere((current_time_ / 5.0 + 1.0) * cube_len, current_time_ / 5 + 0.2,
                                          "steel"));
                    temp = circle_place(cube_place_radius, i, num_cube, 1.0);
                    cubes[i]->setPosition(temp[0] + gc_[0] + 0.05, temp[1] + gc_[1], temp[2] + gc_[2]);
                    cubes[i]->setBodyType(raisim::BodyType::STATIC);
                    raisim::OgreVis::get()->createGraphicalObject(cubes[i], "cubes" + std::to_string(i), "yellow");
                    //                    cubes[i]->setMass(current_time_ + 0.2);
                }
            }
            else
            {
                if (not is_attack)
                {
                    Eigen::Vector3d vel(0, 0, 0);
                    Eigen::Vector3d omega(0, 0, 0);
                    for (int i = 0; i < num_cube; i++)
                    {
                        cubes[i]->setBodyType(raisim::BodyType::DYNAMIC);
                        // vel = minicheetah_->getBasePosition().e() - cubes[i]->getPosition(); // attack towards robot
                        // vel << 5 * vel[0] * (1.0 + (*noise_attack)(e)),
                        //         5 * vel[1] * (1.0 + (*noise_attack)(e)),
                        //         5 * vel[2] * (1.0 + (*noise_attack)(e));
                        vel << gv_[0], gv_[1], -5;
                        cubes[i]->setVelocity(vel, omega);
                    }
                }
            }
        }

        /*
         * random force attack
         */
        void force_attack(bool flag_attack)
        {
            Eigen::VectorXd ff;
            ff.setZero(6);
            Eigen::Vector3d temp_torque;
            // temp_torque.setZero(3);

            if (flag_attack)
            {
                ff.head(6).setRandom();
                ff.head(6).setRandom();
                ff.head(6).setRandom();
                for (int i = 0; i < 3; i++)
                {
                    ff[i] = ff[i] * 2000.0; // 1000.0;
                    //                    ff[i] = -3000;
                    //                    ff[i] = (0 < ff[i] && ff[i] < 100) ? 100 : ff[i];
                    //                    ff[i] = (0 > ff[i] && ff[i] > -100) ? -100 : ff[i];
                }
                for (int i = 3; i < 6; i++)
                {
                    ff[i] = ff[i] * 400.0; // 400.0;
                    //                    ff[i] = (0 < ff[i] && ff[i] < 50) ? 50 : ff[i];
                    //                    ff[i] = (0 > ff[i] && ff[i] > -50) ? -50 : ff[i];
                }
                ff[0] = 0.0;
                ff[1] = 0.0;
                //                ff[3] = 0.0;
                //                ff[4] = 0.0;
                ff[5] = 0.0;

                temp_torque = ff.segment(3, 3);
                temp_torque = bodyFrameMatrix_.e() * temp_torque;
                ff.segment(3, 3) << temp_torque;
                // std::cout << "rotation" << bodyFrameMatrix_ << std::endl;
                // std::cout << "force" << ff.segment(0, 3) << std::endl;
                ggff.head(6) << ff;
                // minicheetah_->setGeneralizedForce(ff);
            }
            else
            {
                // minicheetah_->setGeneralizedForce(ff);
                ggff.head(6) << ff;
            }
        }

        void state_disturbance(bool flag_disturbance)
        {
            // directly add disturbance to robot base state
            float ratio = 0.5;
            if (flag_disturbance)
            {
                Eigen::VectorXd pos_noise;
                Eigen::VectorXd vel_noise;
                pos_noise.setRandom(7);
                vel_noise.setRandom(6);
                // gc_[2] = gc_[2] + 0.02 * pos_noise[2];  // z disturbance
                // gc_.segment(2, 5) << gc_.segment(2, 5) * pos_noise.segment(2, 5) * 0.1 + gc_.segment(2, 5);
                // gv_.segment(2, 4) << gv_.segment(2, 4) * vel_noise.segment(2, 4) * 0.1 + gv_.segment(2, 4);
                gc_[2] = gc_[2] + 0.03 * pos_noise[2] * ratio;
                gc_[3] = gc_[3] + 0.1 * pos_noise[3] * ratio;
                gc_[4] = gc_[4] + 0.1 * pos_noise[4] * ratio;
                gc_[5] = gc_[5] + 0.1 * pos_noise[5] * ratio;
                gc_[6] = gc_[6] + 0.1 * pos_noise[6] * ratio;
                gv_[2] = gv_[2] + 0.1 * vel_noise[2] * ratio;
                gv_[3] = gv_[3] + 0.3 * vel_noise[3] * ratio;
                gv_[4] = gv_[4] + 0.3 * vel_noise[4] * ratio;

                minicheetah_->setState(gc_, gv_);
            }
            else
            {
                ;
            }
        }

        void updateExtraInfo() final
        {
            extraInfo_["EndEffectorReward(0.15)"] = EndEffectorReward;
            extraInfo_["Height_Keep_Reward(0.1)"] = BodyCenterReward;
            extraInfo_["base height"] = gc_[2];
            extraInfo_["Balance_Keep_Reward(0.1)"] = BodyAttitudeReward;
            extraInfo_["JointReward(0.65)"] = JointReward;
            extraInfo_["VelocityReward(0.2)"] = VelocityReward;
        }

        /*
         * this method is used to update observation expect reference joint
         * height X1, axis_z X3, joint X12, vel X3, omega X3, joint_dot X12, contact X4
         */
        void updateObservation()
        {
            minicheetah_->getState(gc_, gv_); // get dynamics state of robot

            obDouble_.setZero(obDim_); // clear observation buffer
            obScaled_.setZero(obDim_);

            // phase update
            if (flag_manual or flag_ManualTraj)
            {
                obDouble_[3] = sin(2 * PI * current_time_ / period_);
                obDouble_[4] = cos(2 * PI * current_time_ / period_);
                // std::cout<<"current time: "<<current_time_<<" phase: "<<obDouble_[3]<<" "<<obDouble_[4]<<std::endl;
            }
            else
            {
                obDouble_.segment(3, 2) << ref.row(frame_idx).transpose().segment(25, 2).cast<double>();
            }
            // obDouble_[4] = cos(2 * PI * current_time_ / period_);
            // joint angle update
            Eigen::VectorXd joint_noise;
            joint_noise.setRandom(12);
            // obDouble_.segment(5, 12) = gc_.tail(12) * (joint_noise * jointNoise * noise_flag) + gc_.tail(12);
            obDouble_.segment(5, 12) = (joint_noise * jointNoise * noise_flag) + gc_.tail(12);

            // joint angle velocity update
            joint_noise.setRandom(12);
            obDouble_.segment(17, 12) = (joint_noise * jointVelocityNoise * noise_flag) + gv_.tail(12);

            // posture update
            raisim::Mat<3, 3> rot;
            raisim::Vec<4> quat;
            quat[0] = gc_[3];
            quat[1] = gc_[4];
            quat[2] = gc_[5];
            quat[3] = gc_[6];
            raisim::quatToRotMat(quat, rot);
            bodyFrameMatrix_ = rot;
            obDouble_.segment(29, 3) << rot.e().row(2)(0) + (*noise_posture)(e)*noise_flag,
                rot.e().row(2)(1) + (*noise_posture)(e)*noise_flag,
                rot.e().row(2)(2) + (*noise_posture)(e)*noise_flag;

            // transform from the global coordinate
            bodyLinearVel_ = rot.e().transpose() * gv_.segment(0, 3);
            bodyAngularVel_ = rot.e().transpose() * gv_.segment(3, 3);
            obDouble_.segment(32, 3) << bodyAngularVel_(0) + noise_flag * (*noise_omega)(e),
                bodyAngularVel_(1) + noise_flag * (*noise_omega)(e),
                bodyAngularVel_(2) + noise_flag * (*noise_omega)(e);
        }

        /*
         * this function is used to update command observation
         * ramdomly produce command, which is used for gait generation to produce reference joint
         */
        void command_obs_update(bool flag_reset)
        {

            if (flag_manual)
            {
                // since now, do nothing
                // in manual control mode, this should be change in controller scripts
                // for AI controller, user can just modify nn input layer directly
                ;
            }
            else
            {

                // obDouble_.head(3) << ref.row(frame_idx).transpose().tail(3).cast<double>();
                if (flag_ManualTraj)
                {
                    float temp__ = 0.0;
                    temp__ = rand() / float(RAND_MAX);
                    if (temp__ < 0.5 / (max_time / control_dt_) or flag_reset)
                    {
                        // randomly modify control command
                        // since this is uniform random distribution, the command update almost T/dt/100 times
                        // update command
                        rand();
                        rand();
                        rand();
                        rand();
                        rand();
                        temp__ = rand() / float(RAND_MAX);
                        if (temp__ < 0.2)
                        {
                            for (auto cmd : command)
                            {
                                cmd = 0.0;
                            }
                        }
                        if (0.2 < temp__ and temp__ <= 0.7)
                        {
                            rand();
                            rand();
                            rand();
                            rand();
                            rand();
                            temp__ = rand() / float(RAND_MAX);
                            command[0] = temp__ * Vx_max + (1.0 - temp__) * Vx_min;
                        }
                        else
                        {
                            if (0.7 < temp__ and temp__ <= 0.85)
                            {
                                rand();
                                rand();
                                rand();
                                rand();
                                rand();
                                temp__ = rand() / float(RAND_MAX);
                                command[1] = temp__ * Vy_max + (1.0 - temp__) * Vy_min;
                            }
                            else
                            {
                                rand();
                                rand();
                                rand();
                                rand();
                                rand();
                                temp__ = rand() / float(RAND_MAX);
                                command[2] = temp__ * omega_max + (1.0 - temp__) * omega_min;
                            }
                        }
                    }
                    if (flag_reset)
                    {
                        command_filtered[0] = command[0];
                        command_filtered[1] = command[1];
                        command_filtered[2] = command[2];
                    }
                    else
                    {
                        for (int i = 0; i < 3; i++)
                        {
                            command_filtered[i] =
                                command_filtered[i] * cmd_update_param + command[i] * (1 - cmd_update_param);
                        }
                    }

                    obDouble_.head(3) << command_filtered[0],
                        command_filtered[1],
                        command_filtered[2];
                    gait_generator_manual(flag_reset);
                }
                else
                {
                    obDouble_.head(3) << ref.row(frame_idx).transpose().segment(27, 3).cast<double>();
                    command_filtered[0] = obDouble_.data()[0];
                    command_filtered[1] = obDouble_.data()[1];
                    command_filtered[2] = obDouble_.data()[2];
                    gait_generator();
                }
            }
        }

        /*
         * contact_obs_update is used to check whether the leg contact with ground
         * since now, if the hoof contact with ground, the value will close to 1
         * using the contact force and shape function to produce the contact signal
         */
        void contact_obs_update(bool flag_reset)
        {
            // note: contact is setted to zero during every update observation
            // shape function: obs = 1 - exp(alpha*contact_force)
            // std::cout << "contact:";
            if (not flag_TimeBasedContact)
            {
                for (double &cc : contact_)
                {
                    cc = 0.0;
                }
                for (auto &contact : minicheetah_->getContacts())
                {
                    // minicheetah_->printOutBodyNamesInOrder();
                    if (minicheetah_->getBodyIdx("shank_fr") == contact.getlocalBodyIndex())
                    {
                        // obDouble_(34) = 1 - exp(contact_para * contact.getImpulse()->e().squaredNorm());
                        // std::cout << "fr:"<<contact_para * contact.getImpulse()->e().squaredNorm();
                        // std::cout << "fr: "<<contact.getImpulse()->e() / world_->getTimeStep()<<std::endl;
                        contact_[0] = 1.0;
                    }
                    else if (minicheetah_->getBodyIdx("shank_fl") == contact.getlocalBodyIndex())
                    {
                        // obDouble_(35) = 1 - exp(contact_para * contact.getImpulse()->e().squaredNorm());
                        // std::cout << "fl:"<<contact_para * contact.getImpulse()->e().squaredNorm();
                        contact_[1] = 1.0;
                    }
                    else if (minicheetah_->getBodyIdx("shank_hr") == contact.getlocalBodyIndex())
                    {
                        // obDouble_(36) = 1 - exp(contact_para * contact.getImpulse()->e().squaredNorm());
                        // std::cout << "hr:"<<contact_para * contact.getImpulse()->e().squaredNorm();
                        contact_[2] = 1.0;
                    }
                    else if (minicheetah_->getBodyIdx("shank_hl") == contact.getlocalBodyIndex())
                    {
                        // obDouble_(37) = 1 - exp(contact_para * contact.getImpulse()->e().squaredNorm());
                        // std::cout << "hl:"<<contact_para * contact.getImpulse()->e().squaredNorm();
                        contact_[3] = 1.0;
                    }
                    else
                    {
                        ; // amazing saturation
                    }
                }
                for (int i = 0; i < 4; i++)
                {
                    //                    contact_filtered[i] = (flag_reset) ? contact_[i] : (contact_filtered[i] * filter_para +
                    //                                                                        (1.0 - filter_para) * contact_[i]);
                    contact_filtered[i] = contact_[i];
                    // obDouble_(34 + i) = contact_filtered[i];
                }
                // std::cout << obDouble_.segment(34, 4).transpose() << std::endl;
            }
            else
            {
                // time based contact
                float real_phase;
                float temp;
                for (double &cc : contact_)
                {
                    cc = 0.0;
                }
                for (int i = 0; i < 4; i++)
                {
                    real_phase = current_time_ + phase_[i] * period_;
                    real_phase = fmod(real_phase, period_) / period_;
                    // temp = (real_phase < lam_) ? 2.0 * real_phase : 0.0;
                    // contact_filtered[i] = ((temp < 1.0) and (temp > 0.01)) ? 1.0 : 0.0;
                    // obDouble_(34 + i) = contact_filtered[i];
                    contact_filtered[i] = (real_phase < lam_) ? 1.0 : 0.0;
                    contact_[i] = contact_filtered[i];
                    // obDouble_(34 + i) = contact_filtered[i];
                }
                // obDouble_[34] = cos(2 * PI * current_time_ / period_);
                // obDouble_[35] = sin(2 * PI * current_time_ / period_);
                // obDouble_[36] = sin(4 * PI * current_time_ / period_);
                // obDouble_[37] = sin(6 * PI * current_time_ / period_);
            }
        }

        /*
         * update contact information
         */
        void contact_information_update()
        {
            // uddate contact force
            for (double &force : contact_force_norm)
                force = 0.0; // clear all contact force
            for (auto &contact : minicheetah_->getContacts())
            {
                if (minicheetah_->getBodyIdx("shank_fr") == contact.getlocalBodyIndex())
                {
                    contact_force_norm[0] = contact.getImpulse().e().norm() / control_dt_;
                }
                if (minicheetah_->getBodyIdx("shank_fl") == contact.getlocalBodyIndex())
                {
                    contact_force_norm[1] = contact.getImpulse().e().norm() / control_dt_;
                }
                if (minicheetah_->getBodyIdx("shank_hr") == contact.getlocalBodyIndex())
                {
                    contact_force_norm[2] = contact.getImpulse().e().norm() / control_dt_;
                }
                if (minicheetah_->getBodyIdx("shank_hl") == contact.getlocalBodyIndex())
                {
                    contact_force_norm[3] = contact.getImpulse().e().norm() / control_dt_;
                }
            }
            raisim::Vec<3> temp_footVelocity;
            minicheetah_->getFrameVelocity(minicheetah_->getFrameByName("toe_fr_joint"), temp_footVelocity);
            contact_vel_norm[0] = temp_footVelocity.norm();
            minicheetah_->getFrameVelocity(minicheetah_->getFrameByName("toe_fl_joint"), temp_footVelocity);
            contact_vel_norm[1] = temp_footVelocity.norm();
            minicheetah_->getFrameVelocity(minicheetah_->getFrameByName("toe_hr_joint"), temp_footVelocity);
            contact_vel_norm[2] = temp_footVelocity.norm();
            minicheetah_->getFrameVelocity(minicheetah_->getFrameByName("toe_hl_joint"), temp_footVelocity);
            contact_vel_norm[3] = temp_footVelocity.norm();
            //            std::cout<<"frame"<<frame_idx<<" force: "
            //                    <<contact_force_norm[0]<<", "
            //                    <<contact_force_norm[1]<<", "
            //                    <<contact_force_norm[2]<<", "
            //                    <<contact_force_norm[3]<<", "
            //                    <<"vel: "
            //                    <<contact_vel_norm[0]<<", "
            //                    <<contact_vel_norm[1]<<", "
            //                    <<contact_vel_norm[2]<<", "
            //                    <<contact_vel_norm[3]<<", "
            //                    <<std::endl;
        }

        /*
         * scale the observation and return to neural network
         */
        void observe(Eigen::Ref<EigenVec> ob) final
        {
            // convert it to float
            if (flag_ObsFilter)
            {
                obDouble_.tail(obDim_ - 5) << obDouble_.tail(obDim_ - 5) * ObsFilterAlpha +
                                                  obDouble_last_.tail(obDim_ - 5) * (1.0 - ObsFilterAlpha);
                obDouble_last_ = obDouble_;
            }
            else
            {
                ;
            }
            obScaled_ = (obDouble_ - obMean_).cwiseQuotient(obStd_); // (obDouble_-obMean_)./obStd_
            ob = obScaled_.cast<float>();
            // std::cout << "-----------------------------" << std::endl;
            // std::cout << "ob_double" << obDouble_.transpose() << std::endl;
            // std::cout << "ob_scale" << obScaled_.transpose() << std::endl;
            // std::cout << "*****************************" << std::endl;
            // std::cout << "raisim_ob_scaled:" << obScaled_.segment(16, 6).transpose() << std::endl;
        }

        /*
         * clamp the torque according to current speed
         * */
        void torque_clamp()
        {
            double up = 0;
            double low = 0;
            double ratio = 1.0;
            double r;
            r = MotorMaxTorque / (MotorMaxSpeed - MotorCriticalSpeed);

            Eigen::VectorXd upper;
            Eigen::VectorXd lower;
            upper.setZero(nJoints_ + 6);
            lower.setZero(nJoints_ + 6);

            upper.head(6).setConstant(20000);
            lower.head(6).setConstant(-20000);

            for (int i = 0; i < nJoints_; i++)
            {
                ratio = ((i + 1) % 3 == 0) ? 1.55f : 1.0f;
                up = (gv_temp[i + 6] * ratio > MotorCriticalSpeed) ? (MotorMaxTorque -
                                                                      (gv_temp[i + 6] * ratio - MotorCriticalSpeed) * r)
                                                                   : MotorMaxTorque;
                up = up * ratio;
                low = (gv_temp[i + 6] * ratio < -MotorCriticalSpeed) ? ((-MotorMaxSpeed - gv_temp[i + 6] * ratio) /
                                                                        (-MotorMaxSpeed + MotorCriticalSpeed) *
                                                                        -MotorMaxTorque)
                                                                     : -MotorMaxTorque;
                low = low * ratio;
                upper[i + 6] = up;
                lower[i + 6] = low;
                torque[i] = fmax(fmin(torque[i], up), low);
                //                std::cout << "joint" << i << ": " << up << " " << torque[i] << " " << low << std::endl;
            }
            minicheetah_->setActuationLimits(upper, lower);
            //            std::cout << "Actuation Limit:" << std::endl;
            //            std::cout<<minicheetah_->getActuationUpperLimits()<<std::endl;
            //            std::cout << upper << std::endl;
            //            std::cout<<minicheetah_->getActuationLowerLimits()<<std::endl;
            //            std::cout << lower << std::endl;
        }

        /*
         * return all original state
         */
        void OriginState(Eigen::Ref<EigenVec> origin_state)
        {
            Eigen::VectorXd ss;
            // ss.setZero(gcDim_ + gvDim_);
            // ss << gc_, gv_;
            ss.setZero(gcDim_ + gvDim_ + 4);
            ss << gc_, gv_, contact_filtered[0], contact_filtered[1], contact_filtered[2], contact_filtered[3];
            origin_state = ss.cast<float>();
        }

        /*
         * return origin state dim
         */
        int GetOriginStateDim()
        {
            // return (gcDim_ + gvDim_);
            return (gcDim_ + gvDim_ + 4);
        }

        /*
         * return the reference trajectory
         */
        void ReferenceState(Eigen::Ref<EigenVec> refer_state)
        {
            Eigen::VectorXd rs;
            rs.setZero(12 + 12);
            rs << jointRef_, jointDotRef_;
            refer_state = rs.cast<float>();
        }

        /*
         * return generalized joint force
         */
        void GetJointEffort(Eigen::Ref<EigenVec> joint_effort)
        {
            Eigen::VectorXd gf;
            gf.setZero(nJoints_);
            gf << minicheetah_->getGeneralizedForce().e().tail(nJoints_);
            //            gf << minicheetah_->getFeedForwardGeneralizedForce().e().tail(nJoints_);
            //            std::cout<<gf.transpose()<<std::endl;
            joint_effort = gf.cast<float>();
        }

        /*
         * return generalized force
         */
        void GetGeneralizedForce(Eigen::Ref<EigenVec> generalized_force)
        {
            Eigen::VectorXd gf;
            gf.setZero(gvDim_);
            gf << minicheetah_->getGeneralizedForce().e();
            generalized_force = gf.cast<float>();
            // Eigen::delete(gf);
        }

        /*
         * Get MassMatrix, shape is gvdim*gvdim,1
         */
        void GetInverseMassMatrix(Eigen::Ref<EigenVec> inverse_mass)
        {
            Eigen::MatrixXd mm;
            Eigen::VectorXd mm_v;
            mm.setZero(gvDim_, gvDim_);
            mm_v.setZero(gvDim_ * gvDim_);
            mm = minicheetah_->getInverseMassMatrix().e();
            // std::cout << "1111" << std::endl;
            for (int i = 0; i < gvDim_; i++)
            {
                // std::cout << i << std::endl;
                mm_v.segment(i * gvDim_, gvDim_) << mm.col(i);
            }
            // Eigen::Map<Eigen::VectorXd> mm(minicheetah_->getInverseMassMatrix().e().data(), minicheetah_->getInverseMassMatrix().e().size());
            inverse_mass = mm_v.cast<float>();
            // std::cout << "2222" << std::endl;
        }

        /*
         * Get nonlinear part
         */
        void GetNonlinear(Eigen::Ref<EigenVec> Nonlinear)
        {
            Eigen::VectorXd nonlinear;
            nonlinear.setZero(gvDim_);
            nonlinear << minicheetah_->getNonlinearities(world_->getGravity()).e();
            Nonlinear = nonlinear.cast<float>();
        }

        /*
         * Set Contact Model Coefficient
         */
        void SetContactCoefficient(const Eigen::Ref<EigenVec> &contact_coeff)
        {
            Eigen::VectorXd cccc;
            cccc.setZero(3);
            cccc = contact_coeff.cast<double>();
            contact_model_friction = cccc[0];
            contact_model_restitution = cccc[1];
            contact_model_resThreshold = cccc[2];
            world_->setDefaultMaterial(contact_model_friction, contact_model_restitution, contact_model_resThreshold);
            std::cout << "Set Default Material to:" << cccc.transpose() << std::endl;
            std::cout << "ATTENTION, Whether [RESET] is needed" << std::endl;
        }

        /*
         * Return the attack ball information
         */
        void GetSphereInfo(Eigen::Ref<EigenVec> sphere_info)
        {
            if (flag_crucial)
            {
                Eigen::VectorXd sphere_kinematics_data;
                sphere_kinematics_data.setZero(4);
                sphere_kinematics_data << cubes[0]->getComPosition_rs().e(), cubes[0]->getRadius();
                sphere_info = sphere_kinematics_data.cast<float>();
            }
            else
            {
                std::cout << "Please make sure the [Flag_Crucial] is True";
            }
        }

        /*
         * according to gait pattern generator & current state update rewards
         * the form of the cost function mainly refer to https://github.com/xbpeng/DeepMimic
         * for more detail, please click this link
         * return DeepMimic reward (double)
         */
        double DeepMimicRewardUpdate()
        {
            // =================================================================
            // <<<<<<<<<<<<Calculate end effector reward>>>>>>>>>>>>>>>>>>>>>>>>
            bodyPos_ << gc_[0], gc_[1], gc_[2];
            Eigen::Vector3d temp;
            for (int i = 0; i < 4; i++)
            {
                minicheetah_->getFramePosition(minicheetah_->getFrameIdxByName(ToeName_[i]), TempPositionHolder_);
                // here, the hoof's radius is ignored, this may be wrong, please check it
                // make sure to check it !!!!!!
                temp << TempPositionHolder_[0], TempPositionHolder_[1], TempPositionHolder_[2];
                EndEffector_.segment(3 * i, 3) << bodyFrameMatrix_.e().transpose() * (temp - bodyPos_);
            }
            // std::cout<<BOLDYELLOW<<EndEffector_.transpose()<<std::endl;
            EndEffectorReward = (EndEffector_ - EndEffectorRef_).squaredNorm();
            EndEffectorReward = EECoeff * exp(-40 * EndEffectorReward);

            // ==================================================================
            // <<<<<<<<<<<<<<<<< Calculate body center error >>>>>>>>>>>>>>>>>>>>
            // the designd trajectory along the x direction, designed speed is gait_step / period
            // bodyPos_(0) = bodyPos_(0) + gait_step_ / period_ * current_time_;
            // std::cout<<BOLDYELLOW<<bodyPos_<<std::endl;
            bodyPos_(0) = 0;
            bodyPos_(1) = 0;
            // std::cout<<BOLDCYAN<<bodyPos_<<std::endl;

            // temp << 0, 0, ref.row(0)[24];
            // temp << 0, 0, ref.row(frame_idx).transpose().segment(24, 1)[0];
            temp << 0, 0, stand_height_;
            // BodyCenterReward = bodyPos_.segment(1, 1).squaredNorm();  // here is a problem, z axis should be a desired constant
            BodyCenterReward = (bodyPos_ - temp).squaredNorm();
            BodyCenterReward = BodyPosCoeff * exp(-80 * BodyCenterReward);

            // posture reward
            // double BodyAttitudeReward = (gv_.segment(3,4) - ForwardDirection_.tail(4)).squaredNorm();  // keep forward & balance
            // temp << ref.row(frame_idx).transpose().segment(30, 3).cast<double>();
            BodyAttitudeReward = obDouble_.segment(29, 2).squaredNorm(); // make the sobot keep balance
            // BodyAttitudeReward = (obDouble_.segment(29, 3) - temp).squaredNorm();
            BodyAttitudeReward = BodyAttiCoeff * exp(-80 * BodyAttitudeReward);

            // DirectionKeepReward = obDouble_.segment(22, 3).squaredNorm();
            // temp << 1, 0, 0;
            // DirectionKeepReward = (obDouble_.tail(3) - temp).squaredNorm();
            // DirectionKeepReward = 0.0 * exp(-2 * DirectionKeepReward);

            // ==================================================================
            // <<<<<<<<<<<<<<<<<< Calculate Joint Mimic Reward>>>>>>>>>>>>>>>>>>>
            JointReward = (jointRef_ - gc_.tail(12)).squaredNorm();
            JointReward = JointMimicCoeff * 0.25 * exp(-2.0 * JointReward);
            JointDotReward = (jointDotRef_ - gv_.tail(12)).squaredNorm();
            JointDotReward = JointMimicCoeff * 0.75 * exp(-control_dt_ * JointDotReward);

            //===================================================================
            // <<<<<<<<<<<<<<<<<<<<<<<<< Velocity Reward >>>>>>>>>>>>>>>>>>>>>>>>
            // this reward is used to follow user command
            bodyLinearVelRef_ << command_filtered[0], command_filtered[1], 0;
            bodyLinearVelRef_[0] = (flag_WildCat) ? -bodyLinearVelRef_[0]
                                                  : bodyLinearVelRef_[0]; // mirror the vx command
            bodyAngularVelRef_ << 0.0, 0.0, command_filtered[2];
            VelocityReward = VelKeepCoeff / 2 * exp(-2 * (bodyLinearVel_ - bodyLinearVelRef_).squaredNorm()) +
                             VelKeepCoeff / 2 * exp(-2 * (bodyAngularVel_ - bodyAngularVelRef_).squaredNorm());

            // ==================================================================
            // <<<<<<<<<<<<<<<<<<<<<<<< Torque Reward >>>>>>>>>>>>>>>>>>>>>>>>>>>
            //            torque = (pTarget12_ - gc_.tail(12)).cwiseProduct(jointPgain.tail(nJoints_)) -
            //                     (gv_.tail(12)).cwiseProduct(jointDgain.tail(12));
            torque = torque.cwiseQuotient(torque_limit);
            // TorqueReward = TorqueCoeff * exp(-1.0 * torque.squaredNorm());
            TorqueReward = TorqueCoeff / 2.0 * exp(-0.1 * torque.squaredNorm()) +
                           TorqueCoeff / 2.0 * exp(-0.1 / control_dt_ * (torque - torque_last).squaredNorm());
            torque_last = torque;

            // ==================================================================
            // <<<<<<<<<<<<<<<<<<<<<<< Contact Reward >>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            ContactReward = 0;
            double real_phase = 0;
            for (int i = 0; i < 4; i++)
            {
                real_phase = current_time_ + phase_[i] * period_;
                real_phase = fmod(real_phase, period_) / period_;
                ContactReward += 4 * contact_vel_norm[i] * contact_vel_norm[i] * smooth_function(real_phase, 2, lam_);
                ContactReward += 2 * (contact_force_norm[i] / 12.5) * (contact_force_norm[i] / 12.5) *
                                 smooth_function2(real_phase, 2, lam_);
            }
            ContactReward = ContactCoeff * exp(-2 * ContactReward);

            // update gui log
            if (visualizeThisStep_)
            {
                gui::rewardLogger.log("EndEffector", EndEffectorReward);
                gui::rewardLogger.log("HeightKeep", BodyCenterReward);
                gui::rewardLogger.log("BalanceKeep", BodyAttitudeReward);
                gui::rewardLogger.log("Joint", JointReward);
                gui::rewardLogger.log("JointDot", JointDotReward);
                gui::rewardLogger.log("Velocity", VelocityReward);
                gui::rewardLogger.log("Torque", TorqueReward);
                gui::rewardLogger.log("Contact", ContactReward);
                gui::rewardLogger.log("Forward", command_filtered[0]);
                gui::rewardLogger.log("lateral", command_filtered[1]);
                gui::rewardLogger.log("yaw", command_filtered[2]);
            }
            return (EndEffectorReward + BodyCenterReward + JointReward + JointDotReward +
                    VelocityReward + BodyAttitudeReward + TorqueReward + ContactReward);
        }

        /*
         * detect whether robot is under terrible situation and terminate this explore
         * */
        bool isTerminalState(float &terminalReward) final
        {
            // used to detect whether to terminate the episode
            terminalReward = float(terminalRewardCoeff_);
            //            std::cout<<obDouble_.segment(0,4).transpose()<<std::endl;
            // if the height of the robot is not in 0.28~0.39 or The body is too inclined
            // if(obDouble_[0]<0.28 or obDouble_[0]>0.39 or obDouble_[3]<0.7){
            if (gc_[2] < 0.15 or gc_[2] > 0.65 or obDouble_[31] < 0.5)
            {
                //                if(obDouble_[0] < 0.25){
                //                    std::cout<<"body too low"<<std::endl;
                //                }
                //                if(obDouble_[0] > 0.4){
                //                    std::cout<<"body too high"<<std::endl;
                //                }
                //                if(obDouble_[3]<0.7){
                //                    std::cout<<"body too declining"<<std::endl;
                //                }
                return true;
            }
            else
            {
                terminalReward = 0.f;
                return false;
            }
        }

        void setSeed(int seed) final
        {
            std::srand(seed);
        }

        void close() final {}

        /*
         * this function is used to load all parameter from the yaml file
         * switch the environment to be manual mode or auto mode
         * switch the ground to be plane or terrain
         * load gait parameter
         * something else
         */
        void parameter_load_from_yaml(const YAML::Node &cfg)
        {
            // ------------------------------------------------------------
            // load gait parameter
            READ_YAML(double, abad_, cfg["abad"]);
            READ_YAML(double, period_, cfg["period"]);
            READ_YAML(double, lam_, cfg["lam"]);
            READ_YAML(double, stand_height_, cfg["stand_height"]);
            READ_YAML(double, up_height_, cfg["up_height"]);
            up_height_max_ = up_height_;
            READ_YAML(double, down_height_, cfg["down_height"]);
            READ_YAML(double, gait_step_, cfg["gait_step"]);
            READ_YAML(double, Vx_max, cfg["Vx"]);
            //            Vx_min = -Vx_max;
            READ_YAML(double, Vy_max, cfg["Vy"]);
            Vy_min = -Vy_max;
            READ_YAML(double, omega_max, cfg["Omega"]);
            omega_min = -omega_max;
            READ_YAML(double, Lean_middle_front, cfg["LeanFront"]);
            READ_YAML(double, Lean_middle_hind, cfg["LeanHind"]);
            // ------------------------------------------------------------
            // load mode parameter
            READ_YAML(bool, flag_terrain, cfg["Terrain"]);
            READ_YAML(bool, flag_manual, cfg["Manual"]);
            READ_YAML(bool, flag_crucial, cfg["Crutial"]);
            READ_YAML(bool, flag_filter, cfg["Filter"]);
            READ_YAML(bool, flag_fix_camera_to_ground, cfg["Camera"]);
            READ_YAML(bool, flag_StochasticDynamics, cfg["StochasticDynamics"]);
            READ_YAML(bool, flag_HeightVariable, cfg["HeightVariable"]);
            READ_YAML(bool, flag_TimeBasedContact, cfg["TimeBasedContact"]);
            READ_YAML(bool, flag_ManualTraj, cfg["ManualTraj"]);
            READ_YAML(bool, flag_MotorDynamics, cfg["MotorDynamics"]);
            READ_YAML(bool, flag_ObsFilter, cfg["ObsFilter"]);
            READ_YAML(bool, flag_WildCat, cfg["WILDCAT"]);
            READ_YAML(bool, flag_ForceDisturbance, cfg["ForceDisturbance"]);
            READ_YAML(bool, flag_convert2torque, cfg["Convert2Torque"]);
            // ------------------------------------------------------------
            // load reward parameter
            READ_YAML(double, terminalRewardCoeff_, cfg["terminalRewardCoeff"]);
            READ_YAML(double, EECoeff, cfg["EndEffectorRewardCoeff"]);
            READ_YAML(double, BodyPosCoeff, cfg["BodyPosRewardCoeff"]);
            READ_YAML(double, BodyAttiCoeff, cfg["BodyAttitudeRewardCoeff"]);
            READ_YAML(double, JointMimicCoeff, cfg["JointRewardCoeff"]);
            READ_YAML(double, VelKeepCoeff, cfg["VelRewardCoeff"]);
            READ_YAML(double, TorqueCoeff, cfg["TorqueCoeff"]);
            READ_YAML(double, ContactCoeff, cfg["ContactCoeff"]);

            // ------------------------------------------------------------
            // load control parameter
            READ_YAML(double, stiffness, cfg["Stiffness"]);
            READ_YAML(double, stiffness_low, cfg["Stiffness_Low"]);
            READ_YAML(double, abad_ratio, cfg["AbadRatio"]);
            READ_YAML(double, damping, cfg["Damping"]);
            READ_YAML(double, freq, cfg["Freq"]);
            READ_YAML(double, max_time, cfg["max_time"]);
            READ_YAML(int, num_cube, cfg["CubeNum"]);
            READ_YAML(double, desired_fps_, cfg["FPS"]);
            READ_YAML(double, actionNoise, cfg["ActionNoise"]);
            READ_YAML(double, noise_flag, cfg["ObsNoise"]);
            READ_YAML(int, gaitType, cfg["GaitType"]);
            // -------------------------------------------------------------
            // load motor work condition parameter
            READ_YAML(double, MotorMaxTorque, cfg["MotorMaxTorque"]);
            READ_YAML(double, MotorCriticalSpeed, cfg["MotorCriticalSpeed"]);
            READ_YAML(double, MotorMaxSpeed, cfg["MotorMaxSpeed"]);
        }

        /*
         * generate reference joint angle
         * */
        void gait_generator()
        {
#if DEBUG0
            std::cout << "4444" << std::endl;
            // std::cout << ref->row(0) << std::endl;
#endif
            jointRef_ << ref.row(frame_idx).transpose().segment(0, 12).cast<double>();
            jointDotRef_ << ref.row(frame_idx).transpose().segment(12, 12).cast<double>();
#if DEBUG1
            std::cout << "frame_id: " << frame_idx << std::endl;
            std::cout << "ref_traj: " << jointRef_.transpose() << std::endl;
            std::cout << "col: " << ref.col(0).segment(0, 12) << std::endl;
            std::cout << "row: " << ref.row(frame_idx).segment(0, 12).cast<double>() << std::endl;
#endif
#if DEBUG0
            std::cout << "444444" << std::endl;
            // std::cout << ref->row(0) << std::endl;
#endif
        }

        /*
         * 3D inverse kinematics solver
         */
        void inverse_kinematics(double x, double y, double z,
                                double l_hip, double l_thigh, double l_calf,
                                double *theta, bool is_right)
        {

            double ll = sqrt(x * x + y * y + z * z);
            if (ll > max_len)
            {
                x = x * (max_len - 1e-5) / ll;
                y = y * (max_len - 1e-5) / ll;
                z = z * (max_len - 1e-5) / ll;
            }
            double temp, temp1, temp2 = 0.0;
            if (is_right)
            {
                temp = (-z * l_hip - sqrt(y * y * (z * z + y * y - l_hip * l_hip))) / (z * z + y * y);
                if (abs(temp) <= 1)
                {
                    theta[0] = asin(temp);
                }
                else
                {
                    std::cout << "error1" << std::endl;
                }
            }
            else
            {
                temp = (z * l_hip + sqrt(y * y * (z * z + y * y - l_hip * l_hip))) / (z * z + y * y);
                if (abs(temp) <= 1)
                {
                    theta[0] = asin(temp);
                }
                else
                {
                    std::cout << "error1" << std::endl;
                }
            }
            double lr = sqrt(x * x + y * y + z * z - l_hip * l_hip);
            lr = (lr > (l_thigh + l_calf)) ? (l_thigh + l_calf - 1e-4) : lr;
            temp = (l_thigh * l_thigh + l_calf * l_calf - lr * lr) / 2 / l_thigh / l_calf + 1e-5;
            if (fabs(temp) <= 1)
            {
                theta[2] = -(PI - acos(temp));
            }
            else
            {
                std::cout << "error2" << std::endl;
                std::cout << "temp" << temp << std::endl;
                std::cout << "lr=" << lr << std::endl;
            }
            //            temp1 = x / sqrt(y * y + z * z) - 1e-10;
            temp1 = x / lr;
            temp2 = (lr * lr + l_thigh * l_thigh - l_calf * l_calf) / 2 / lr / l_thigh - 1e-5;
            if (fabs(temp1) <= 1 and fabs(temp2) <= 1)
            {
                theta[1] = acos(temp2) - asin(temp1);
            }
            else
            {
                std::cout << "error3" << std::endl;
                std::cout << "abs(temp1)" << fabs(temp1) << std::endl;
                std::cout << "abs(temp2)" << fabs(temp2) << std::endl;
                std::cout << "lr=" << lr << std::endl;
            }
        }

        /*
         * manual traj gait generator
         */
        void gait_generator_manual(bool is_first)
        {
            // temp variable
            double real_phase = 0.0;
            double toe_x = 0.0;
            double toe_y = 0.0;
            double toe_z = 0.0;

            double l1 = l_thigh_;
            double l2 = l_calf_;
            double temp[3] = {0.0, 0.0, 0.0};

            double anti_flag = 1.0;
            EndEffectorRef_.setZero(nJoints_);

            // according to the command to update the gait parameter
            gait_step_ = command_filtered[0] * lam_ * period_;      // command speed along x * period
            gait_step_ = (flag_WildCat) ? -gait_step_ : gait_step_; // mirror the trajectory
                                                                    //            side_step_ = command_filtered[1] / lam_ * period_;        // command speed along y * period
            side_step_ = command_filtered[1] * lam_ * period_;      // command speed along y * period
            // rot_step_ = command_filtered[2] * period_ * 0.22;  // command rotation speed * radius * period
            rot_step_ = command_filtered[2] * period_ * 0.4;
            // change the maximum height of the hoof from the ground
            if (flag_HeightVariable)
            {
                double ratio = 0.0;
                ratio = std::abs(command_filtered[0]) / Vx_max;
                if (Vy_max > 0)
                {
                    ratio = fmax(ratio, std::abs(command_filtered[1]) / Vy_max);
                }
                if (omega_max > 0)
                {
                    ratio = fmax(ratio, std::abs(command_filtered[2] / omega_max));
                }
                up_height_ = (ratio > 0.1) ? up_height_max_ : ratio * up_height_max_;
            }
            Eigen::Vector3d p0, pf, toe;
            double temp_offset[4];
            temp_offset[0] = -l_hip_ + Lean_middle_front;
            temp_offset[1] = l_hip_ - Lean_middle_front;
            temp_offset[2] = -l_hip_ + Lean_middle_hind;
            temp_offset[3] = l_hip_ - Lean_middle_hind;
            if (is_first)
            {
                // calculate last step to get the reference joint velocity
                for (int i = 0; i < 4; i++)
                {
                    real_phase =
                        current_time_ + phase_[i] * period_ - control_dt_; // calculate last time joint reference
                    real_phase = fmod(real_phase, period_) / period_;
                    anti_flag = (i < 2) ? 1.0 : -1.0;
                    // calculate the toe position relative to the hip
                    double temp_r = 0;
                    if (real_phase < lam_)
                    {
                        temp_r = real_phase / lam_;
                        p0 << gait_step_ / 2.0, side_step_ / 2.0 + anti_flag * rot_step_ / 2.0, -stand_height_;
                        pf << -gait_step_ / 2.0, -side_step_ / 2.0 + -anti_flag * rot_step_ / 2.0, -stand_height_;
                        toe = cubicBezier(p0, pf, temp_r);
                    }
                    else
                    {
                        temp_r = (real_phase - lam_) / (1.0 - lam_);
                        //                        if (temp_r < 0.5) {
                        //                            p0 << -gait_step_ / 2.0, -side_step_ / 2.0 + -anti_flag * rot_step_ / 2.0, -stand_height_;
                        //                            pf << 0.0, 0.0, up_height_ - stand_height_;
                        //                            toe = cubicBezier(p0, pf, temp_r / 0.5);
                        //                        } else {
                        //                            p0 << 0.0, 0.0, up_height_ - stand_height_;
                        //                            pf << gait_step_ / 2.0, side_step_ / 2.0 + anti_flag * rot_step_ / 2.0, -stand_height_;
                        //                            toe = cubicBezier(p0, pf, (temp_r - 0.5) / 0.5);
                        //                        }
                        pf << gait_step_ / 2.0, side_step_ / 2.0 + anti_flag * rot_step_ / 2.0, -stand_height_;
                        p0 << -gait_step_ / 2.0, -side_step_ / 2.0 + -anti_flag * rot_step_ / 2.0, -stand_height_;
                        toe = Bezier2(p0, pf, temp_r, up_height_);
                    }
                    toe_x = toe(0);
                    toe_y = toe(1);
                    toe_z = toe(2);
                    inverse_kinematics(toe_x, toe_y + temp_offset[i], toe_z,
                                       l_hip_, l_thigh_, l_calf_, temp,
                                       i == 0 or i == 2);
                    jointRefLast_[3 * i + 0] = temp[0];
                    jointRefLast_[3 * i + 1] = -temp[1];
                    jointRefLast_[3 * i + 2] = -temp[2];
                }
            }
            for (int i = 0; i < 4; i++)
            {
                real_phase = current_time_ + phase_[i] * period_;
                real_phase = fmod(real_phase, period_) / period_;
                anti_flag = (i < 2) ? 1.0 : -1.0;
                // calculate the toe position relative to the hip
                double temp_r = 0;
                if (real_phase < lam_)
                {
                    temp_r = real_phase / lam_;
                    p0 << gait_step_ / 2.0, side_step_ / 2.0 + anti_flag * rot_step_ / 2.0, -stand_height_;
                    pf << -gait_step_ / 2.0, -side_step_ / 2.0 + -anti_flag * rot_step_ / 2.0, -stand_height_;
                    toe = cubicBezier(p0, pf, temp_r);
                }
                else
                {
                    temp_r = (real_phase - lam_) / (1.0 - lam_);
                    //                    if (temp_r < 0.5) {
                    //                        p0 << -gait_step_ / 2.0, -side_step_ / 2.0 + -anti_flag * rot_step_ / 2.0, -stand_height_;
                    //                        pf << 0.0, 0.0, up_height_ - stand_height_;
                    //                        toe = cubicBezier(p0, pf, temp_r / 0.5);
                    //                    } else {
                    //                        p0 << 0.0, 0.0, up_height_ - stand_height_;
                    //                        pf << gait_step_ / 2.0, side_step_ / 2.0 + anti_flag * rot_step_ / 2.0, -stand_height_;
                    //                        toe = cubicBezier(p0, pf, (temp_r - 0.5) / 0.5);
                    //                    }
                    pf << gait_step_ / 2.0, side_step_ / 2.0 + anti_flag * rot_step_ / 2.0, -stand_height_;
                    p0 << -gait_step_ / 2.0, -side_step_ / 2.0 + -anti_flag * rot_step_ / 2.0, -stand_height_;
                    toe = Bezier2(p0, pf, temp_r, up_height_);
                }
                toe_x = toe(0);
                toe_y = toe(1);
                toe_z = toe(2);
                inverse_kinematics(toe_x, toe_y + temp_offset[i], toe_z, l_hip_, l_thigh_, l_calf_, temp,
                                   i == 0 or i == 2);
                jointRef_[3 * i + 0] = temp[0];
                jointRef_[3 * i + 1] = -temp[1];
                jointRef_[3 * i + 2] = -temp[2];
                EndEffectorRef_[3 * i + 0] = toe_x;
                EndEffectorRef_[3 * i + 1] = toe_y;
                EndEffectorRef_[3 * i + 2] = toe_z;
            }
            jointDotRef_ = (jointRef_ - jointRefLast_) / control_dt_;
            jointRefLast_ = jointRef_;

            EndEffectorRef_ = EndEffectorRef_ + EndEffectorOffset_;
        }

        /*
         * set reference trajectory database
         */
        void set_ref(Eigen::MatrixXf pr)
        {
            // ref = std::move(pr);
            ref = pr;
        }

        //-----------------------------------------
        // static Eigen::MatrixXf ref;

    private:
        // visualization related definition
        bool visualizable_ = false;
        std::normal_distribution<double> distribution_;
        raisim::ArticulatedSystem *minicheetah_;
        std::vector<GraphicObject> *minicheetahVisual_;
        std::vector<GraphicObject> *topography_;

        double desired_fps_ = 60.;
        int visualizationCounter_ = 0;

        // policy & robot related definition
        int gcDim_, gvDim_, nJoints_;
        Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, gc_temp, gv_temp, pTarget_, pTarget12_, vTarget_, ggff;
        // ggff is the generalized force for PD_PLUS_FORWARD_FORCE
        Eigen::VectorXd random_init;
        Eigen::VectorXd random_vel_init;
        Eigen::VectorXd pTarget12Last_;
        Eigen::VectorXd jointLast_; // store last time joint angle
        Eigen::VectorXd jointPgain, jointDgain;
        Eigen::VectorXd torque;       // estimate joint torque of PD controller
        Eigen::VectorXd torque_limit; // the boundary of torque
        Eigen::VectorXd torque_last;  // store the last time torque, try to smooth the output

        // lost function related definition
        double terminalRewardCoeff_ = -10.;

        Eigen::VectorXd actionMean_, actionStd_, obMean_, obStd_;
        Eigen::VectorXd obDouble_, obScaled_, obDouble_last_;
        Eigen::Vector3d bodyLinearVel_, bodyAngularVel_;
        Eigen::Vector3d bodyLinearVelRef_, bodyAngularVelRef_;

        // gait pattern parameter define, those parameter can be change according to yaml configuration file
        Eigen::VectorXd phase_;
        double current_time_ = 0.0; // current time
        double abad_ = 0.0;         // control the reference abad joint
        double period_ = 0.5;       // default gait period
        double lam_ = 0.65;         // proportion of time the robot touch the ground in each cycle
        double stand_height_ = 0.30;
        double up_height_ = 0.02;     // height of toe raise
        double up_height_max_ = 0.02; // record the maximum height of toe raise
        double down_height_ = 0.02;   // height of toe embedded in the ground
        double gait_step_ = 0.09;     // length of one gait
        double side_step_ = 0.0;      // control the speed along the y axis
        double rot_step_ = 0.0;       // control the rotation speed
        double l_thigh_ = 0.209;      // length of the thigh
        double l_calf_ = 0.2175;      // length of the calf    # 0.19+0.0275
        // double l_calf_ = 0.19;        // ignore hoof size
        double l_hip_ = 0.085; // length of hip along y direction
        double max_len = 0.0;
        double abad_ratio = 1.0;   // scaling factor, for abad joint stiffness and damping
        Eigen::VectorXd jointRef_; // store the reference joint angle at current time, length should be 12
        Eigen::VectorXd jointRefLast_;
        Eigen::VectorXd jointDotRef_; // store the reference joint angle at last time, length should be 12

        raisim::Vec<3> TempPositionHolder_; // tempory position holder to get toe position
        Eigen::VectorXd EndEffector_;       // storage minicheetah storage four toe position relative to body frame
        Eigen::VectorXd EndEffectorRef_;    // reference end effector traj updated by pattern_generator(), relative to body frame
        Eigen::VectorXd EndEffectorOffset_; // offset between hip coordinates and body center
        std::string ToeName_[4] = {"toe_fr_joint", "toe_fl_joint", "toe_hr_joint",
                                   "toe_hl_joint"}; // storage minicheetah toe frame name
        Eigen::Vector3d bodyPos_;
        raisim::Mat<3, 3> bodyFrameMatrix_;
        double DeepMimicReward_ = 0;
        Eigen::VectorXd ForwardDirection_;

        //        std::vector<raisim::Box *> cubes;  // for crucial learning, box will attack the robot
        std::vector<raisim::Sphere *> cubes; // for crucial learning, box will attack the robot

        int num_cube = 10;              // cube number
        double cube_len = 0.08;         // length of cube
        double cube_mass = 0.4;         // default mass of cube
        double cube_place_radius = 0.0; // cube place radius

        double EndEffectorReward = 0;
        double BodyCenterReward = 0;
        double BodyAttitudeReward = 0;
        double JointReward = 0;
        double JointDotReward = 0;
        double VelocityReward = 0;
        double TorqueReward = 0;
        double ContactReward = 0;

        double actionNoise = 0.1;
        double jointNoise = 0.002;       // 0.002;
        double jointVelocityNoise = 0.8; // 0.8;
        // double observer_noise_amplitude = 1.1;

        double noise_flag = 1.0;
        default_random_engine e;
        // double noise_std_z = 0.01  // default height estimate noise standard deviation
        // normal_distribution<float>* noise_z;
        normal_distribution<float> *noise_z = new normal_distribution<float>(0.0, 0.005);
        normal_distribution<float> *noise_posture = new normal_distribution<float>(0.0, 0.02); // 0.02);
                                                                                               //        normal_distribution<float> *noise_posture = new normal_distribution<float>(0.0, 0.0);
        normal_distribution<float> *noise_omega = new normal_distribution<float>(0.0, 0.5);    // 0.5);
                                                                                               //        normal_distribution<float> *noise_omega = new normal_distribution<float>(0.0, 0.0);

        normal_distribution<float> *noise_vel = new normal_distribution<float>(0.0, 0.5);
        normal_distribution<float> *noise_joint_vel = new normal_distribution<float>(0.0, 0.8);
        normal_distribution<float> *noise_attack = new normal_distribution<float>(0.0, 0.15);
        Eigen::VectorXd action_noise;
        Eigen::VectorXd init_noise;

        double filter_para = 0.0; // action = filter_para * action_now + (1-filter_para) * action_last

        bool flag_manual = false;               // control robot automatically or manually
        bool flag_terrain = false;              // terrain ground or flat
        bool flag_crucial = false;              // whether add the difficulty of learning
        bool flag_filter = false;               // whether to add action filter
        bool flag_is_attack = false;            // flag to show whether the robot is under attack
        bool flag_fix_camera_to_ground = false; // flag to control the camera view point, fixed to ground or robot
        bool flag_StochasticDynamics = false;   // flag to control whether disturb robot's dynamics
        bool flag_HeightVariable = false;       // flag to control whether the height of the foot lift is variable
        bool flag_TimeBasedContact = false;     // flag to control whether use the real contact or time based contact
        bool flag_ManualTraj = false;           // flag to control whether use the manual trajectory
        bool flag_MotorDynamics = false;        // flag to enable motor dynamics during simulation
        bool flag_ObsFilter = false;            // flag to control whether filter the observation value
        bool flag_WildCat = false;              // flag to mimic boston dynamics wild cat way
        bool flag_ForceDisturbance = false;     // flag to apply force to robot body
        bool flag_convert2torque = false;

        double ObsFilterFreq = 20;      // observation filter frequency
        double ObsFilterAlpha = 1.0;    // observation filter parameter
        double stiffness = 40.0;        // PD control stiffness
        double stiffness_low = 5.0;     // PD control stiffness during swing phase
        double damping = 1.0;           // PD control damping
        double freq = 0;                // low pass through filter's frequency
        double contact_para = -0.5;     // contact shape function parameter, 1-exp(para*force)
        double EECoeff = 0.0;           // end effector reward coefficient
        double BodyPosCoeff = 0.0;      // keep body height reward coefficient
        double BodyAttiCoeff = 0.0;     // keep body attitude reward coefficient
        double JointMimicCoeff = 0.0;   // mimic the reference joint reward coefficient
        double VelKeepCoeff = 0.0;      // velocity follow coefficient
        double TorqueCoeff = 0.0;       // torque penalty coefficient
        double ContactCoeff = 0.0;      // contact reward coefficient
        double AttackProbability = 0.0; // attack probability, =reward^3
        int gaitType = 0;               // gait type, influence the phase.    0 for trot, 1 for bounding

        double cmd_update_param = 0.995; // about 1s to reach the expected command
        // double cmd_update_param = 0.95;
        double command[3] = {0.0, 0.0, 0.0};               // update new command in reset()
        double command_filtered[3] = {0.0, 0.0, 0.0};      // store the filtered command
        double contact_[4] = {0.0, 0.0, 0.0, 0.0};         // store the contact state
        double contact_filtered[4] = {0.0, 0.0, 0.0, 0.0}; // filtered contacted information

        double contact_force_norm[4] = {0.0, 0.0, 0.0, 0.0}; // store four hoof's contact force
        double contact_vel_norm[4] = {0.0, 0.0, 0.0, 0.0};   // store four hoof's velocity

        double Vx_max = 0.8;           // maximum body velocity along x direction (forward)
        double Vx_min = 0.0;           // minimum body velocity along x direction (forward)
        double Vy_max = 0.3;           // maximum body velocity along x direction (forward)
        double Vy_min = -0.3;          // minimum body velocity along x direction (forward)
        double omega_max = PI / 12.0;  // maximum body velocity along x direction (forward)
        double omega_min = -PI / 12.0; // minimum body velocity along x direction (forward)

        unsigned long int itera = 0;
        int frame_idx = 0; // frame of reference trajectory data
        int frame_max = 0; // total frame number of reference trajectory
        int frame_len = 0; // max frame of single explore
        double max_time = 0.0;
        double Lean_middle_front = 0.04; // the distance of foot point moves symmetrically towards the center of front legs
        double Lean_middle_hind = 0.04;  // the distance of foot point moves symmetrically towards the center of hind legs

        // define disturbance of dynamics
        double mass_distrubance_ratio = 0.15; // add 15% mass disturbance of robot
        double com_distrubance = 0.02;        // 0.02 m uncertainty of center of mass
        double calf_distrubance = 0.01;       // 0.01 m uncertainty of the calf length

        // contact related parameter
        double contact_model_friction = 1.0;
        double contact_model_restitution = 0.2;
        double contact_model_resThreshold = 20.0;

        // Motor work condition parameter definition
        double MotorMaxTorque = 18.0;
        double MotorCriticalSpeed = 14.2;
        double MotorMaxSpeed = 40;
        double Motor_Prop_Damping = 0.00;
        Eigen::VectorXd MotorDamping;

        // Eigen::MatrixXf ref;
        Eigen::MatrixXf ref;
    };
}