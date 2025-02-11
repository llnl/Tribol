# Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
# other Tribol Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: (MIT)


#------------------------------------------------------------------------------
# tribol_add_code_checks( PREFIX [prefix] )
#
# Adds code checks for all cpp/hpp files recursively under the current directory
# that regex match INCLUDES and excludes any files that regex match EXCLUDES
# 
# This creates the following parent build targets:
#  check - Runs a non file changing style check and CppCheck
#  style - In-place code formatting
#
# Creates various child build targets that follow this pattern:
#  tribol_<check|style>
#  tribol_<cppcheck|clangformat>_<check|style>
#
# This also creates targets for running clang-tidy on the src/ and test/
# directories, with a more permissive set of checks for the tests,
# called tribol_guidelines_check and tribol_guidelines_check_tests, respectively
#------------------------------------------------------------------------------
macro(tribol_add_code_checks)

  set(options)
  set(singleValueArgs PREFIX)
  set(multiValueArgs)

  # Parse the arguments to the macro
  cmake_parse_arguments(arg
       "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN})

  # Create file globbing expressions that only include directories that contain source
  set(_base_dirs "src")
  # Note: any extensions added here should also be added to BLT's lists in CMakeLists.txt
  set(_ext_expressions "*.cpp" "*.hpp" "*.inl" "*.cuh" "*.cu" "*.cpp.in" "*.hpp.in")

  set(_glob_expressions)
  foreach(_exp ${_ext_expressions})
      foreach(_base_dir ${_base_dirs})
          list(APPEND _glob_expressions "${PROJECT_SOURCE_DIR}/${_base_dir}/${_exp}")
      endforeach()
  endforeach()

  # Glob for list of files to run code checks on
  set(_sources)
  file(GLOB_RECURSE _sources ${_glob_expressions})

  blt_add_code_checks(PREFIX          ${arg_PREFIX}
                      SOURCES         ${_sources}
                      CLANGFORMAT_CFG_FILE ${PROJECT_SOURCE_DIR}/.clang-format
                      CPPCHECK_FLAGS  --enable=all --inconclusive)


  set(_src_sources)
  file(GLOB_RECURSE _src_sources "src/*.cpp" "src/*.hpp" "src/*.inl")
  list(FILTER _src_sources EXCLUDE REGEX ".*/tests/.*pp")

  blt_add_clang_tidy_target(NAME              ${arg_PREFIX}_guidelines_check
                            CHECKS            "clang-analyzer-*,clang-analyzer-cplusplus*,cppcoreguidelines-*"
                            SRC_FILES         ${_src_sources})

  # Create list of recursive test directory glob expressions
  # NOTE: GLOB operator ** did not appear to be supported by cmake and did not recursively find test subdirectories
  # NOTE: Do not include all directories at root (for example: blt)

  file(GLOB_RECURSE _test_sources "${PROJECT_SOURCE_DIR}/src/*.cpp" "${PROJECT_SOURCE_DIR}/tests/*.cpp")
  list(FILTER _test_sources INCLUDE REGEX ".*/tests/.*pp")

  blt_add_clang_tidy_target(NAME              ${arg_PREFIX}_guidelines_check_tests
                            CHECKS            "clang-analyzer-*,clang-analyzer-cplusplus*,cppcoreguidelines-*,-cppcoreguidelines-avoid-magic-numbers"
                            SRC_FILES         ${_test_sources})
                                
  if (ENABLE_COVERAGE)
      blt_add_code_coverage_target(NAME   ${arg_PREFIX}_coverage
                                   RUNNER ${CMAKE_MAKE_PROGRAM} test
                                   SOURCE_DIRECTORIES ${PROJECT_SOURCE_DIR}/src )
  endif()

endmacro(tribol_add_code_checks)

##------------------------------------------------------------------------------
## tribol_assert_path_exists( path )
##
## Checks if the specified path to a file or directory exits. If the path
## does not exist, CMake will throw a fatal error message and the configure
## step will fail.
##
##------------------------------------------------------------------------------
macro(tribol_assert_path_exists path )

  if ( NOT EXISTS ${path} )
    message( FATAL_ERROR "[${path}] does not exist!" )
  endif()

endmacro(tribol_assert_path_exists)


##------------------------------------------------------------------------------
## tribol_install( SOURCE_HEADERS    [ header1 [header2...] ]
##                 GENERATED_HEADERS [ h1 [h2...] ] )
##
## This macro installs the Tribol library, headers and CMake export targets.
##------------------------------------------------------------------------------
macro(tribol_install)

  set(options)
  set(singleValueArgs)
  set(multiValueArgs SOURCE_HEADERS GENERATED_HEADERS )

  ## parse arguments
  cmake_parse_arguments(arg
    "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN})

  if ( NOT arg_SOURCE_HEADERS )
    message( FATAL_ERROR "tribol_install() called without specifying SOURCE_HEADERS" )
  endif()

  install( TARGETS tribol
           EXPORT   tribol-targets
           LIBRARY  DESTINATION lib
           ARCHIVE  DESTINATION lib
           RUNTIME  DESTINATION bin
           INCLUDES DESTINATION include )

  ## mirror the directory structure on the install
  foreach( tribol_header_file ${arg_SOURCE_HEADERS} )
    get_filename_component( tribol_base_dir ${tribol_header_file} DIRECTORY)
    install( FILES ${tribol_header_file}
             DESTINATION "include/tribol/${tribol_base_dir}" )
  endforeach()

  ## copy over generated headers
  foreach( tribol_gen_header ${arg_GENERATED_HEADERS} )
     install( FILES ${tribol_gen_header}
              DESTINATION "include/tribol" )
  endforeach()

  install(EXPORT tribol-targets DESTINATION lib/cmake)

endmacro(tribol_install)

##------------------------------------------------------------------------------
## convert_to_native_escaped_file_path( path output )
##
## This macro converts a cmake path to a platform specific string literal
## usable in C++.  (For example, on windows C:/Path will be come C:\\Path)
##------------------------------------------------------------------------------

macro(convert_to_native_escaped_file_path path output)
    file(TO_NATIVE_PATH ${path} ${output})
    string(REPLACE "\\" "\\\\"  ${output} "${${output}}")
endmacro()

##------------------------------------------------------------------------------
## tribol_configure_file
##
## This macro is a thin wrapper over the builtin configure_file command.
## It has the same arguments/options as configure_file but introduces an
## intermediate file that is only copied to the target file if the target differs
## from the intermediate.
##------------------------------------------------------------------------------
macro(tribol_configure_file _source _target)
    set(_tmp_target ${_target}.tmp)
    configure_file(${_source} ${_tmp_target} ${ARGN})
    execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_tmp_target} ${_target})
    execute_process(COMMAND ${CMAKE_COMMAND} -E remove ${_tmp_target})
endmacro(tribol_configure_file)
