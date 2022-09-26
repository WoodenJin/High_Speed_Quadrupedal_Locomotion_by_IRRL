#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "Environment.hpp"
#include "VectorizedEnvironment.hpp"

namespace py = pybind11;
using namespace raisim;

#ifndef ENVIRONMENT_NAME
#define ENVIRONMENT_NAME FlexibleGymEnv
#endif

PYBIND11_MODULE(_flexible_robot, m) {
    py::class_<VectorizedEnvironment<ENVIRONMENT>>(m, RSG_MAKE_STR(ENVIRONMENT_NAME))
            .def(py::init<std::string, std::string>())
            .def("init", &VectorizedEnvironment<ENVIRONMENT>::init)
            .def("getExtraInfoNames", &VectorizedEnvironment<ENVIRONMENT>::getExtraInfoNames)
            .def("reset", &VectorizedEnvironment<ENVIRONMENT>::reset)
            .def("observe", &VectorizedEnvironment<ENVIRONMENT>::observe)
            .def("step", &VectorizedEnvironment<ENVIRONMENT>::step)
            .def("setSeed", &VectorizedEnvironment<ENVIRONMENT>::setSeed)
            .def("step", &VectorizedEnvironment<ENVIRONMENT>::step)
            .def("testStep", &VectorizedEnvironment<ENVIRONMENT>::testStep)
            .def("close", &VectorizedEnvironment<ENVIRONMENT>::close)
            .def("isTerminalState", &VectorizedEnvironment<ENVIRONMENT>::isTerminalState)
            .def("setSimulationTimeStep", &VectorizedEnvironment<ENVIRONMENT>::setSimulationTimeStep)
            .def("setControlTimeStep", &VectorizedEnvironment<ENVIRONMENT>::setControlTimeStep)
            .def("getObDim", &VectorizedEnvironment<ENVIRONMENT>::getObDim)
            .def("getActionDim", &VectorizedEnvironment<ENVIRONMENT>::getActionDim)
            .def("getExtraInfoDim", &VectorizedEnvironment<ENVIRONMENT>::getExtraInfoDim)
            .def("getNumOfEnvs", &VectorizedEnvironment<ENVIRONMENT>::getNumOfEnvs)
            .def("startRecordingVideo", &VectorizedEnvironment<ENVIRONMENT>::startRecordingVideo)
            .def("stopRecordingVideo", &VectorizedEnvironment<ENVIRONMENT>::stopRecordingVideo)
            .def("showWindow", &VectorizedEnvironment<ENVIRONMENT>::showWindow)
            .def("hideWindow", &VectorizedEnvironment<ENVIRONMENT>::hideWindow)
            .def("curriculumUpdate", &VectorizedEnvironment<ENVIRONMENT>::curriculumUpdate)
            .def("OriginState", &VectorizedEnvironment<ENVIRONMENT>::OriginState)
            .def("GetOriginStateDim", &VectorizedEnvironment<ENVIRONMENT>::GetOriginStateDim)
            .def("ReferenceState", &VectorizedEnvironment<ENVIRONMENT>::ReferenceState)
            .def("GetJointEffort", &VectorizedEnvironment<ENVIRONMENT>::GetJointEffort)
            .def("GetGeneralizedForce", &VectorizedEnvironment<ENVIRONMENT>::GetGeneralizedForce)
            .def("GetInverseMassMatrix", &VectorizedEnvironment<ENVIRONMENT>::GetInverseMassMatrix)
            .def("GetNonlinear", &VectorizedEnvironment<ENVIRONMENT>::GetNonlinear)
            .def("SetContactCoefficient", &VectorizedEnvironment<ENVIRONMENT>::SetContactCoefficient)
            .def("GetSphereInfo", &VectorizedEnvironment<ENVIRONMENT>::GetSphereInfo);
}
