#!/bin/bash

#
# Copyright 2022 Benjamin Worpitz, Simeon Ehrig, Jan Stephan
# SPDX-License-Identifier: MPL-2.0
#

set +xv
source ./script/setup_utilities.sh

echo_green "<SCRIPT: install_gcc>"

: "${ALPAKA_CI_GCC_VER?'ALPAKA_CI_GCC_VER must be specified'}"
: "${ALPAKA_CI_SANITIZERS?'ALPAKA_CI_SANITIZERS must be specified'}"
: "${CXX?'CXX must be specified'}"

if agc-manager -e gcc@${ALPAKA_CI_GCC_VER}
then
    echo_green "<USE: preinstalled GCC ${ALPAKA_CI_GCC_VER}>"
else
    echo_yellow "<INSTALL: GCC ${ALPAKA_CI_GCC_VER}>"

    travis_retry sudo add-apt-repository -y ppa:ubuntu-toolchain-r/ppa # Contains gcc 10.4 (Ubuntu 20.04)
    travis_retry sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test # Contains gcc 11 (Ubuntu 20.04)
    travis_retry sudo apt-get -y --quiet update
    travis_retry sudo apt-get -y --quiet --allow-unauthenticated --no-install-recommends install g++-"${ALPAKA_CI_GCC_VER}"
fi

sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-"${ALPAKA_CI_GCC_VER}" 50
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-"${ALPAKA_CI_GCC_VER}" 50
if [[ "${ALPAKA_CI_SANITIZERS}" == *"TSan"* ]]
then
    travis_retry sudo apt-get -y --quiet --allow-unauthenticated --no-install-recommends install libtsan0
fi

which "${CXX}"
${CXX} -v
