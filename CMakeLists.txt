# ----------------------------------------------------------------------------
# Master CMake file for the UPGMpp project.
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

PROJECT(UPGMplusplus)
SET(PROJECT_NAME "UPGMplusplus")

if(COMMAND cmake_policy)
	cmake_policy(SET CMP0005 OLD)
endif(COMMAND cmake_policy)

# Required commands in newer CMake versions:
CMAKE_MINIMUM_REQUIRED(VERSION 2.4)
if(COMMAND cmake_policy)
      cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)
 
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# EIGEN
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# BOOST
set(Boost_USE_STATIC_LIBS OFF) 
set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF) 

find_package(Boost 1.46.1 COMPONENTS filesystem serialization REQUIRED)

include_directories(${Boost_INCLUDE_DIRS}) 

# libBFGS
#set(libLBFGS_INCLUDE_DIR "./" CACHE FILEPATH "Directory of the libBFGS includes directory")
#include_directories(${libLBFGS_INCLUDE_DIR})
#find_path(libLBGS_LIBRARY_DIR "./" CACHE FILEPATH "Directory of the libBFGS library directory")


# ---------------------------------------------
# TARGET:
# ---------------------------------------------

# Create UPGM++ libraries and add include directories to the examples
SET( UPGM++_LIBRARIES "base;training;inference" )

SET(INC_DIR "")
SET(LIBRARIES "")

FOREACH( LIBRARY ${UPGM++_LIBRARIES} )
	ADD_SUBDIRECTORY(libs/${LIBRARY})
	INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/libs/${LIBRARY})
	SET(INC_DIR ${INC_DIR}${CMAKE_SOURCE_DIR}/libs/${LIBRARY}\;)
	SET(LIBRARIES ${LIBRARIES}${PROJECT_BINARY_DIR}/libs/libUPGMplusplus-${LIBRARY}.so\;)
	set_target_properties(${LIBRARY} PROPERTIES PREFIX "libUPGMplusplus-")
ENDFOREACH( LIBRARY ${UPGM++_LIBRARIES} )

SET(INC_DIR ${INC_DIR}${EIGEN3_INCLUDE_DIR}\;)
SET(INC_DIR ${INC_DIR}${libLBFGS_INCLUDE_DIR})


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

# Create examples
SET( UPGM++_EXAMPLES "different_node_types;MAP_and_marginal_inference;pre_computed_weights;saving_and_loading_graphs;synthetic_data;training_example" )

FOREACH( EXAMPLE ${UPGM++_EXAMPLES} )
	# Define the executable target:
	ADD_EXECUTABLE(${EXAMPLE} ${CMAKE_SOURCE_DIR}/examples/${EXAMPLE}/example.cpp)
	# Link the executable
        TARGET_LINK_LIBRARIES(${EXAMPLE} base training inference ${Boost_LIBRARIES})
ENDFOREACH( EXAMPLE ${UPGM++_EXAMPLES} )

configure_file(${PROJECT_NAME}Config.cmake.in
  "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake" @ONLY)
  
INSTALL(FILES cmake/FindUPGMplusplus.cmake DESTINATION ${CMAKE_ROOT}/Modules/)
