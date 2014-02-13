# ----------------------------------------------------------------------------
# Master CMake file for the decoding library of the UPGMpp project.
#
#  February 2014, J.R. Ruiz-Sarmiento <jotaraul@uma.es>
#			University of Málaga
#			University of Osnabrück
#
#  NOTE: CMake can be obtained at http://www.cmake.org/
# ----------------------------------------------------------------------------

SET ( DECODING_LIB_NAME decoding )
# TARGET:
FILE(GLOB aux_srcs1 ${CMAKE_SOURCE_DIR}/libs/${DECODING_LIB_NAME}/*.cpp)
FILE(GLOB aux_srcs2 ${CMAKE_SOURCE_DIR}/libs/${DECODING_LIB_NAME}/*.hpp)

SET(SRCS ${aux_srcs1} ${aux_srcs2})

ADD_LIBRARY(${DECODING_LIB_NAME} ${SRCS})
#TARGET_LINK_LIBRARIES(${DECODING_LIB_NAME})
