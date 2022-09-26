//
// Created by jemin on 3/27/19.
// MIT License
//
// Copyright (c) 2019-2019 Robotic Systems Lab, ETH Zurich
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef SRC_RAISIMGYMVECENV_HPP
#define SRC_RAISIMGYMVECENV_HPP

#include "RaisimGymEnv.hpp"
#include "omp.h"
#include "yaml-cpp/yaml.h"


Eigen::MatrixXf readCSV_m(const std::string &w_file) {

    std::ifstream in(w_file);
    std::string line;
    int row = 0;
    int col = 0;

    Eigen::MatrixXf res(1, 1);
    Eigen::RowVectorXf rowVector(1);
    if (in.is_open()) {

        while (std::getline(in, line, '\n')) {

            // cout << line << endl;

            char *ptr = (char *) line.c_str();
            int len = line.length();

            col = 0;

            char *start = ptr;

            for (int i = 0; i < len; i++) {

                if (ptr[i] == ',') {
                    //res(row, col++) = atof(start);
                    rowVector(col++) = atof(start);
                    start = ptr + i + 1;
                    rowVector.conservativeResize(col + 1);
                }
            }
            //res(row++, col) = atof(start);
            rowVector(col) = atof(start);
            res.conservativeResize(row + 1, col + 1);
            res.row(row++) = rowVector;
        }

        in.close();
    } else {
        std::cout << "Can Not Load Parameter File of " << w_file << std::endl;
    }

    return res;
}

void readCSV_m2(const std::string &w_file, Eigen::MatrixXf *p) {

    std::ifstream in(w_file);
    std::string line;
    int row = 0;
    int col = 0;

    Eigen::MatrixXf res(1, 1);
    Eigen::RowVectorXf rowVector(1);
    if (in.is_open()) {

        while (std::getline(in, line, '\n')) {

            // cout << line << endl;

            char *ptr = (char *) line.c_str();
            int len = line.length();

            col = 0;

            char *start = ptr;

            for (int i = 0; i < len; i++) {

                if (ptr[i] == ',') {
                    //res(row, col++) = atof(start);
                    rowVector(col++) = atof(start);
                    start = ptr + i + 1;
                    rowVector.conservativeResize(col + 1);
                }
            }
            //res(row++, col) = atof(start);
            rowVector(col) = atof(start);
            res.conservativeResize(row + 1, col + 1);
            res.row(row++) = rowVector;
        }

        in.close();
    } else {
        std::cout << "Can Not Load Parameter File of " << w_file << std::endl;
    }

    p = &res;
    // return res;
}


namespace raisim {

    template<class ChildEnvironment>
    class VectorizedEnvironment {

    public:

        explicit VectorizedEnvironment(std::string resourceDir, std::string cfg)
                : resourceDir_(resourceDir) {
            raisim::World::setActivationKey(raisim::Path(resourceDir + "/activation.raisim").getString());
            cfg_ = YAML::Load(cfg);
            if (cfg_["render"])
                render_ = cfg_["render"].template as<bool>();
        }

        ~VectorizedEnvironment() {
            for (auto *ptr: environments_)
                delete ptr;
        }

        void init() {
            omp_set_num_threads(cfg_["num_threads"].template as<int>());
            num_envs_ = cfg_["num_envs"].template as<int>();

            for (int i = 0; i < num_envs_; i++) {
                environments_.push_back(new ChildEnvironment(resourceDir_, cfg_, render_ && i == 0));
                environments_.back()->setSimulationTimeStep(cfg_["simulation_dt"].template as<double>());
                environments_.back()->setControlTimeStep(cfg_["control_dt"].template as<double>());
            }

            if (render_) raisim::OgreVis::get()->hideWindow();

            // ----------------------------------------------------------------------------------
            Eigen::MatrixXf ref;
            bool is_ref = false;
            try {
                const std::string ref_file = cfg_["RefTraj"].template as<std::string>();
                ref = readCSV_m(ref_file);
                // readCSV_m2(ref_file, ref);
                // std::cout<<"****************************"<<std::endl;
                is_ref = true;
            }
            catch (const std::exception &e) {
                is_ref = false;
            }

            setSeed(cfg_["seedd"].template as<double>());  // set seed for each environment
            for (int i = 0; i < num_envs_; i++) {
                // only the first environment is visualized
                try {
                    environments_[i]->set_ref(ref);
                }
                catch (const std::exception &e) {

                }
                environments_[i]->init();
                environments_[i]->reset();
            }

            obDim_ = environments_[0]->getObDim();
            actionDim_ = environments_[0]->getActionDim();
            RSFATAL_IF(obDim_ == 0 || actionDim_ == 0,
                       "Observation/Action dimension must be defined in the constructor of each environment!")

            /// generate reward names
            /// compute it once to get reward names. actual value is not used
            environments_[0]->updateExtraInfo();
            for (auto &re: environments_[0]->extraInfo_)
                extraInfoName_.push_back(re.first);
        }

        std::vector<std::string> &getExtraInfoNames() {
            return extraInfoName_;
        }

        // resets all environments and returns observation
        void reset(Eigen::Ref<EigenRowMajorMat> &ob) {
            
            for (auto env: environments_)
                env->reset();

            observe(ob);
        }

        void observe(Eigen::Ref<EigenRowMajorMat> &ob) {
            for (int i = 0; i < num_envs_; i++)
                environments_[i]->observe(ob.row(i));
        }

        void OriginState(Eigen::Ref<EigenRowMajorMat> &ob) {
            for (int i = 0; i < num_envs_; i++)
                environments_[i]->OriginState(ob.row(i));
        }

        int GetOriginStateDim() {
            return environments_[0]->GetOriginStateDim();
        }

        void ReferenceState(Eigen::Ref<EigenRowMajorMat> &refer) {
            for (int i = 0; i < num_envs_; i++)
                environments_[i]->OriginState(refer.row(i));
        }

        // return the generalized joint effort
        void GetJointEffort(Eigen::Ref<EigenRowMajorMat> &joint_effort) {
            for (int i = 0; i < num_envs_; i++)
                environments_[i]->GetJointEffort(joint_effort.row(i));
        }

        // return the generalized force
        void GetGeneralizedForce(Eigen::Ref<EigenRowMajorMat> &generalized_force) {
            for (int i = 0; i < num_envs_; i++)
                environments_[i]->GetGeneralizedForce(generalized_force.row(i));
        }

        // return the inverse mass matrix
        void GetInverseMassMatrix(Eigen::Ref<EigenRowMajorMat> &inverse_mass) {
            for (int i = 0; i < num_envs_; i++)
                environments_[i]->GetInverseMassMatrix(inverse_mass.row(i));
        }

        // return nonlinear force
        void GetNonlinear(Eigen::Ref<EigenRowMajorMat> &nonlinear) {
            for (int i = 0; i < num_envs_; i++)
                environments_[i]->GetNonlinear(nonlinear.row(i));
        }

        // return the attack sphere's information
        void GetSphereInfo(Eigen::Ref<EigenRowMajorMat> &sphere_info){
            for (int i=0;i<num_envs_;i++)
                environments_[i]->GetSphereInfo(sphere_info.row(i));
        }

//        void SetContactCoefficient(Eigen::Ref<EigenVec> &contact_coeff) {
//            for (int i = 0; i < num_envs_; i++)
//                environments_[i]->SetContactCoefficient(contact_coeff);
//        }

        void SetContactCoefficient(Eigen::Ref<EigenRowMajorMat> &contact_coeff) {
            for (int i = 0; i < num_envs_; i++)
                environments_[i]->SetContactCoefficient(contact_coeff.row(i));
        }

        void step(Eigen::Ref<EigenRowMajorMat> &action,
                  Eigen::Ref<EigenRowMajorMat> &ob,
                  Eigen::Ref<EigenVec> &reward,
                  Eigen::Ref<EigenBoolVec> &done,
                  Eigen::Ref<EigenRowMajorMat> &extraInfo) {
#pragma omp parallel for schedule(dynamic)
            for (int i = 0; i < num_envs_; i++) {
                perAgentStep(i, action, ob, reward, done, extraInfo);
                environments_[i]->observe(ob.row(i));
            }
        }

        void testStep(Eigen::Ref<EigenRowMajorMat> &action,
                      Eigen::Ref<EigenRowMajorMat> &ob,
                      Eigen::Ref<EigenVec> &reward,
                      Eigen::Ref<EigenBoolVec> &done,
                      Eigen::Ref<EigenRowMajorMat> &extraInfo) {
            if (render_) environments_[0]->turnOnVisualization();
            perAgentStep(0, action, ob, reward, done, extraInfo);
            if (render_) environments_[0]->turnOffvisualization();

            environments_[0]->observe(ob.row(0));
        }

        void startRecordingVideo(const std::string &fileName) {
            if (render_) environments_[0]->startRecordingVideo(fileName);
        }

        void stopRecordingVideo() {
            if (render_) environments_[0]->stopRecordingVideo();
        }

        void showWindow() {
            raisim::OgreVis::get()->showWindow();
        }

        void hideWindow() {
            raisim::OgreVis::get()->hideWindow();
        }

        void setSeed(int seed) {
            int seed_inc = seed;
            for (auto *env: environments_)
                env->setSeed(seed_inc++);
        }

        void close() {
            for (auto *env: environments_)
                env->close();
        }

        void isTerminalState(Eigen::Ref<EigenBoolVec> &terminalState) {
            for (int i = 0; i < num_envs_; i++) {
                float terminalReward;
                terminalState[i] = environments_[i]->isTerminalState(terminalReward);
            }
        }

        void setSimulationTimeStep(double dt) {
            for (auto *env: environments_)
                env->setSimulationTimeStep(dt);
        }

        void setControlTimeStep(double dt) {
            for (auto *env: environments_)
                env->setControlTimeStep(dt);
        }

        int getObDim() { return obDim_; }

        int getActionDim() { return actionDim_; }

        int getExtraInfoDim() { return extraInfoName_.size(); }

        int getNumOfEnvs() { return num_envs_; }

        ////// optional methods //////
        void curriculumUpdate() {
            for (auto *env: environments_)
                env->curriculumUpdate();
        };

    private:

        inline void perAgentStep(int agentId,
                                 Eigen::Ref<EigenRowMajorMat> &action,
                                 Eigen::Ref<EigenRowMajorMat> &ob,
                                 Eigen::Ref<EigenVec> &reward,
                                 Eigen::Ref<EigenBoolVec> &done,
                                 Eigen::Ref<EigenRowMajorMat> &extraInfo) {
            reward[agentId] = environments_[agentId]->step(action.row(agentId));

            float terminalReward = 0;
            done[agentId] = environments_[agentId]->isTerminalState(terminalReward);

            environments_[agentId]->updateExtraInfo();

            for (int j = 0; j < extraInfoName_.size(); j++)
                extraInfo(agentId, j) = environments_[agentId]->extraInfo_[extraInfoName_[j]];

            if (done[agentId]) {
                environments_[agentId]->reset();
                reward[agentId] += terminalReward;
            }
        }

        std::vector<ChildEnvironment *> environments_;
        std::vector<std::string> extraInfoName_;

        int num_envs_ = 1;
        int obDim_ = 0, actionDim_ = 0;
        bool recordVideo_ = false, render_ = false;
        std::string resourceDir_;
        YAML::Node cfg_;
    };

}

#endif //SRC_RAISIMGYMVECENV_HPP
