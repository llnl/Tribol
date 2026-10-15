blt_add_executable(
  NAME tribol_mortar_patch_test
  SOURCES mortar_patch_test.cpp
  OUTPUT_DIR ${EXAMPLE_OUTPUT_DIRECTORY}
  DEPENDS_ON tribol::core
  FOLDER tribol/examples)

if(TRIBOL_ENABLE_TESTS)
  blt_add_test(
    NAME tribol_mortar_patch_test
    COMMAND tribol_mortar_patch_test)
endif()
