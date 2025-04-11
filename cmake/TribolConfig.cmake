# Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
# other Tribol Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: (MIT)

#------------------------------------------------------------------------------
# Generate tribol-config.cmake for importing Tribol into other CMake packages
#------------------------------------------------------------------------------

## Get Tribol version information
message(STATUS "Configuring Tribol version ${TRIBOL_VERSION_FULL}")

#------------------------------------------------------------------------------
# General Build Info
#------------------------------------------------------------------------------
install(FILES ${PROJECT_BINARY_DIR}/include/tribol/config.hpp DESTINATION include/tribol)
install(FILES ${PROJECT_BINARY_DIR}/include/shared/config.hpp DESTINATION include/shared)

# Set up some paths, preserve existing cache values (if present)
set(TRIBOL_INSTALL_INCLUDE_DIR "include" CACHE STRING "")
set(TRIBOL_INSTALL_CONFIG_DIR "lib" CACHE STRING "")
set(TRIBOL_INSTALL_LIB_DIR "lib" CACHE STRING "")
set(TRIBOL_INSTALL_BIN_DIR "bin" CACHE STRING "")
set(TRIBOL_INSTALL_CMAKE_MODULE_DIR "${TRIBOL_INSTALL_CONFIG_DIR}/cmake" CACHE STRING "")

convert_to_native_escaped_file_path(${CMAKE_INSTALL_PREFIX} TRIBOL_INSTALL_PREFIX)
set(TRIBOL_INSTALL_PREFIX ${TRIBOL_INSTALL_PREFIX} CACHE STRING "" FORCE)


include(CMakePackageConfigHelpers)

# Add version helper
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/tribol-config-version.cmake
    VERSION ${TRIBOL_VERSION_FULL}
    COMPATIBILITY AnyNewerVersion
)

# Set up cmake package config file
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/tribol-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/tribol-config.cmake
  INSTALL_DESTINATION 
    ${TRIBOL_INSTALL_CONFIG_DIR}
  PATH_VARS
    TRIBOL_INSTALL_INCLUDE_DIR
    TRIBOL_INSTALL_LIB_DIR
    TRIBOL_INSTALL_BIN_DIR
    TRIBOL_INSTALL_CMAKE_MODULE_DIR
  )

# Install BLT files
blt_install_tpl_setups(DESTINATION ${TRIBOL_INSTALL_CMAKE_MODULE_DIR})

# Install config files
install(
  FILES 
    ${CMAKE_CURRENT_BINARY_DIR}/tribol-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/tribol-config-version.cmake
  DESTINATION
    ${TRIBOL_INSTALL_CMAKE_MODULE_DIR}
)
