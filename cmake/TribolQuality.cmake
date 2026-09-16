# Add the standard and strict Tribol quality-check targets.
function(tribol_add_quality_targets)
  find_package(
    Python3
    COMPONENTS Interpreter
    QUIET)
  if(NOT Python3_Interpreter_FOUND)
    message(STATUS "Python3 not found; Tribol quality targets are unavailable")
    return()
  endif()

  add_custom_target(
    tribol_quality
    COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/scripts/quality/check.py
            custom spec format headers
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    COMMENT "Running dependency-free Tribol rewrite quality checks")

  add_custom_target(
    tribol_quality_strict
    COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/scripts/quality/check.py
            all --strict-tools
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    COMMENT "Running the complete Tribol rewrite quality suite")
endfunction()
