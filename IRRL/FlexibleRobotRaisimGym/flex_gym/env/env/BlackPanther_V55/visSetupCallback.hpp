//
// Created by jemin on 5/16/19.
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

#ifndef _RAISIM_GYM_VISSETUPCALLBACK_HPP
#define _RAISIM_GYM_VISSETUPCALLBACK_HPP

#include <raisim/OgreVis.hpp>

void setupCallback() {
    auto vis = raisim::OgreVis::get();

    /// light
    vis->getLight()->setDiffuseColour(1, 1, 1);
    vis->getLight()->setCastShadows(true);
    Ogre::Vector3 lightdir(-3, -3, -0.5);
    // Ogre::Vector3 lightdir(0, 0, -1);
    lightdir.normalise();
    vis->getLightNode()->setDirection({lightdir});
    vis->setCameraSpeed(300);

    /// load  textures
    vis->addResourceDirectory(vis->getResourceDir() + "/material/checkerboard");
    vis->loadMaterialFile("checkerboard.material");

    // set skybox


    vis->addResourceDirectory(vis->getResourceDir() + "/material/skybox/whitedays");

    vis->loadMaterialFile("whitedays.material");

    Ogre::Quaternion quat;
    quat.FromAngleAxis(Ogre::Radian(1.57), {1., 0, 0});

    vis->getSceneManager()->setSkyBox(true, "whitedays", 500, true, quat,
                                      Ogre::ResourceGroupManager::AUTODETECT_RESOURCE_GROUP_NAME);


    /// shdow setting
    vis->getSceneManager()->setShadowTechnique(Ogre::SHADOWTYPE_TEXTURE_ADDITIVE);
    vis->getSceneManager()->setShadowTextureSettings(2048, 3);

    /// scale related settings!! Please adapt it depending on your map size
    // beyond this distance, shadow disappears
    vis->getSceneManager()->setShadowFarDistance(3);
    // size of contact points and contact forces
    vis->setContactVisObjectSize(0.03, 0.2);
    // speed of camera motion in freelook mode
    vis->getCameraMan()->setTopSpeed(5);
}


#endif //_RAISIM_GYM_VISSETUPCALLBACK_HPP
