# ----------------------------------------------------------------------------
# Master CMake file for the Kinect tools
#
#  Run with "cmake ." at the root directory to build the makefiles for 
#   the software.
#
#  February 2014, J.R. Ruiz-Sarmiento <jotaraul@uma.es>
#			University of Málaga
#			University of Osnabrück
#
#  NOTE: CMake can be obtained at http://www.cmake.org/
# ----------------------------------------------------------------------------

PROJECT(UPGM++)

if(COMMAND cmake_policy)
	cmake_policy(SET CMP0005 OLD)
endif(COMMAND cmake_policy)

# Required commands in newer CMake versions:
CMAKE_MINIMUM_REQUIRED(VERSION 2.4)
if(COMMAND cmake_policy)
      cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)
 
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

MESSAGE( STATUS "EIGEN_INCLUDE_DIRS: " ${EIGEN3_INCLUDE_DIR} )


# ---------------------------------------------
# TARGET:
# ---------------------------------------------

FILE(GLOB aux_srcs1 ${CMAKE_SOURCE_DIR}/src/*.cpp)
FILE(GLOB aux_srcs2 ${CMAKE_SOURCE_DIR}/src/*.h)
SET(SRCS ${aux_srcs1} ${aux_srcs2})


# Enable GCC profiling (GCC only)
# ===================================================
IF(CMAKE_COMPILER_IS_GNUCXX)
	SET(ENABLE_PROFILING OFF CACHE BOOL "Enable profiling in the GCC compiler (Add flags: -g -pg)")
ENDIF(CMAKE_COMPILER_IS_GNUCXX)

IF(ENABLE_PROFILING)
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -pg")
ENDIF(ENABLE_PROFILING)

IF(UNIX)
	LINK_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}")
ENDIF(UNIX)

# Define the executable target:
ADD_EXECUTABLE(${PROJECT_NAME} ${SRCS})
#ADD_EXECUTABLE(${PROJECT_NAME} WIN32 ${SRCS})

# Add MRPT library to linker:
TARGET_LINK_LIBRARIES( ${PROJECT_NAME} )

##  Copy the configuration file:
#CONFIGURE_FILE(
#	"${CMAKE_SOURCE_DIR}/${PROJECT_NAME}.ini"
#	"${CMAKE_BINARY_DIR}/${PROJECT_NAME}.ini"
#	@ONLY)


