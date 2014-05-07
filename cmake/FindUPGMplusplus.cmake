# - Try to find UPGMplusplus lib
#
#
# Once done this will define
#
#  EIGEN_FOUND - system has eigen lib with correct version
#  EIGEN_INCLUDE_DIR - the eigen include directory
#  EIGEN_VERSION - eigen version
#
# 2014, José Raúl Ruiz Sarmiento <jotaraul@uma.es>
#

SET( UPGM++_LIBRARIES "base;training;inference" )

FOREACH( LIBRARY ${UPGM++_LIBRARIES} )

	find_file(
	     ${LIBRARY}_LIBRARY
	     NAMES libUPGMplusplus-${LIBRARY}.so             
	     PATHS 
	     ${CMAKE_INSTALL_PREFIX}/lib
	     NO_DEFAULT_PATH
	     )

	if(${LIBRARY}_LIBRARY)
	    MESSAGE("Finding ${LIBRARY} lib ...ok!")
	    MESSAGE(${${LIBRARY}_LIBRARY})
	    SET(UPGMplusplus_LIBRARIES ${UPGMplusplus_LIBRARIES};${${LIBRARY}_LIBRARY})
	else()
	    MESSAGE("Finding ${LIBRARY} lib ...fail!")
	endif(${LIBRARY}_LIBRARY)

	
	find_path(
		${LIBRARY}_INCLUDE_DIR
		NAMES	${LIBRARY}
		PATHS
		${CMAKE_INSTALL_PREFIX}/include/UPGMplusplus
		NO_DEFAULT_PATH
		)
		
		
	if(${LIBRARY}_INCLUDE_DIR)
	    SET (${LIBRARY}_INCLUDE_DIR ${${LIBRARY}_INCLUDE_DIR}/${LIBRARY})
	    MESSAGE("Finding ${LIBRARY} include dir ...ok!")
	    MESSAGE(${${LIBRARY}_INCLUDE_DIR})
	    SET(UPGMplusplus_INCLUDE_DIRS ${UPGMplusplus_INCLUDE_DIRS};${${LIBRARY}_INCLUDE_DIR})
	else()
	    MESSAGE("Finding ${LIBRARY} include dir ...fail!")
	endif(${LIBRARY}_INCLUDE_DIR)
	
ENDFOREACH( LIBRARY ${UPGM++_LIBRARIES} )

MESSAGE("UPGMplusplus_LIBRARIES: " ${UPGMplusplus_LIBRARIES})
MESSAGE("UPGMplusplus_INCLUDE_DIRS: " ${UPGMplusplus_INCLUDE_DIRS})


