################################### FUNCTIONS ##################################
# List of functions
#   kokkos_option

# Validate options are given with correct case and define an internal
# upper-case version for use within 

# 
#
# @FUNCTION: kokkos_deprecated_list
#
# Function that checks if a deprecated list option like Kokkos_ARCH was given.
# This prints an error and prevents configure from completing.
# It attempts to print a helpful message about updating the options for the new CMake.
# Kokkos_${SUFFIX} is the name of the option (like Kokkos_ARCH) being checked.
# Kokkos_${PREFIX}_X is the name of new option to be defined from a list X,Y,Z,...
FUNCTION(kokkos_deprecated_list SUFFIX PREFIX)
  SET(CAMEL_NAME Kokkos_${SUFFIX})
  STRING(TOUPPER ${CAMEL_NAME} UC_NAME)

  #I don't love doing it this way but better to be safe
  FOREACH(opt ${KOKKOS_GIVEN_VARIABLES})
    STRING(TOUPPER ${opt} OPT_UC)
    IF ("${OPT_UC}" STREQUAL "${UC_NAME}")
      STRING(REPLACE "," ";" optlist "${${opt}}")
      SET(ERROR_MSG "Given deprecated option list ${opt}. This must now be given as separate -D options, which assuming you spelled options correctly would be:")
      FOREACH(entry ${optlist})
        STRING(TOUPPER ${entry} ENTRY_UC)
        STRING(APPEND ERROR_MSG "\n  -DKokkos_${PREFIX}_${ENTRY_UC}=ON")
      ENDFOREACH()
      STRING(APPEND ERROR_MSG "\nRemove CMakeCache.txt and re-run. For a list of valid options, refer to BUILD.md or even look at CMakeCache.txt (before deleting it).")
      MESSAGE(SEND_ERROR ${ERROR_MSG})
    ENDIF()
  ENDFOREACH()
ENDFUNCTION()

FUNCTION(kokkos_option CAMEL_SUFFIX DEFAULT TYPE DOCSTRING)
  SET(CAMEL_NAME Kokkos_${CAMEL_SUFFIX})
  STRING(TOUPPER ${CAMEL_NAME} UC_NAME)

  # Make sure this appears in the cache with the appropriate DOCSTRING
  SET(${CAMEL_NAME} ${DEFAULT} CACHE ${TYPE} ${DOCSTRING})

  #I don't love doing it this way because it's N^2 in number options, but cest la vie
  FOREACH(opt ${KOKKOS_GIVEN_VARIABLES})
    STRING(TOUPPER ${opt} OPT_UC)
    IF ("${OPT_UC}" STREQUAL "${UC_NAME}")
      IF (NOT "${opt}" STREQUAL "${CAMEL_NAME}")
        MESSAGE(FATAL_ERROR "Matching option found for ${CAMEL_NAME} with the wrong case ${opt}. Please delete your CMakeCache.txt and change option to -D${CAMEL_NAME}=${${opt}}. This is now enforced to avoid hard-to-debug CMake cache inconsistencies.")
      ENDIF()
    ENDIF()
  ENDFOREACH()

  #okay, great, we passed the validation test - use the default
  IF (DEFINED ${CAMEL_NAME})
    SET(${UC_NAME} ${${CAMEL_NAME}} PARENT_SCOPE)
  ELSE()
    SET(${UC_NAME} ${DEFAULT} PARENT_SCOPE)
  ENDIF()

ENDFUNCTION()

FUNCTION(kokkos_append_config_line LINE)
  SET(STR_TO_APPEND "\n${LINE}")
  FILE(APPEND ${Kokkos_BINARY_DIR}/KokkosTempConfig.cmake.in ${STR_TO_APPEND})
ENDFUNCTION()

FUNCTION(kokkos_import_tpl NAME)
  IF (KOKKOS_ENABLE_${NAME})
    FIND_PACKAGE(${NAME} REQUIRED MODULE)

    #make sure this also gets "exported" in the config file
    KOKKOS_APPEND_CONFIG_LINE("ADD_LIBRARY(Kokkos:${lc_name} UNKNOWN IMPORTED)")
    KOKKOS_APPEND_CONFIG_LINE("SET_TARGET_PROPERTIES(Kokkos::${lc_name} PROPERTIES")

    IF(${NAME}_LIBRARIES)
      KOKKOS_APPEND_CONFIG_LINE("IMPORTED_LOCATION ${${NAME}_LIBRARIES}")
    ENDIF()
    IF(${NAME}_INCLUDE_DIR)
      KOKKOS_APPEND_CONFIG_LINE("INTERFACE_INCLUDE_DIRECTORIES ${${NAME}_INCLUDE_DIR}")
    ENDIF()
    KOKKOS_APPEND_CONFIG_LINE(")")
  ENDIF()
ENDFUNCTION(kokkos_import_tpl)

MACRO(kokkos_find_imported NAME)
  CMAKE_PARSE_ARGUMENTS(TPL
   ""
   "HEADER;LIBRARY"
   "HEADER_PATHS;LIBRARY_PATHS"
   ${ARGN})

  IF (TPL_HEADER)
    IF(TPL_HEADER_PATHS)
      FIND_PATH(${NAME}_INCLUDE_DIR ${TPL_HEADER} PATHS ${TPL_HEADER_PATHS})
    ELSE()
      FIND_PATH(${NAME}_INCLUDE_DIR ${TPL_HEADER} PATHS ${${NAME}_ROOT}/include ${KOKKOS_${NAME}_DIR}/include)
    ENDIF()
  ENDIF()

  IF(TPL_LIBRARY)
    IF(TPL_LIBRARY_PATHS)
      FIND_LIBRARY(${NAME}_LIBRARIES ${TPL_LIBRARY} PATHS ${TPL_LIBRARY_PATHS})
    ELSE()
      FIND_LIBRARY(${NAME}_LIBRARIES ${TPL_LIBRARY})# PATHS ${${NAME}_ROOT}/lib ${KOKKOS_${NAME}_DIR}/lib)
    ENDIF()
  ENDIF()

  INCLUDE(FindPackageHandleStandardArgs)
  IF (TPL_HEADER AND TPL_LIBRARY)
    FIND_PACKAGE_HANDLE_STANDARD_ARGS(${NAME} DEFAULT_MSG ${NAME}_INCLUDE_DIR ${NAME}_LIBRARIES)
  ELSEIF(TPL_HEADER)
    FIND_PACKAGE_HANDLE_STANDARD_ARGS(${NAME} DEFAULT_MSG ${NAME}_INCLUDE_DIR)
  ELSEIF(TPL_LIBRARY)
    FIND_PACKAGE_HANDLE_STANDARD_ARGS(${NAME} DEFAULT_MSG ${NAME}_LIBRARIES)
  ELSE()
    #not sure if I should error here? this would be a weird TPL
  ENDIF()

  MARK_AS_ADVANCED(${NAME}_INCLUDE_DIR ${NAME}_LIBRARIES)
  STRING(TOLOWER ${NAME} lc_name)
  ADD_LIBRARY(Kokkos::${lc_name} UNKNOWN IMPORTED)
  #make sure this also gets "exported" in the config file
  IF(TPL_LIBRARY)
    SET_TARGET_PROPERTIES(Kokkos::${lc_name} PROPERTIES
      IMPORTED_LOCATION ${${NAME}_LIBRARIES})
  ENDIF()
  IF(TPL_HEADER)
    SET_TARGET_PROPERTIES(Kokkos::${lc_name} PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES ${${NAME}_INCLUDE_DIR})
  ENDIF()
ENDMACRO(kokkos_find_imported)



