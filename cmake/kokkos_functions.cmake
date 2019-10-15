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

FUNCTION(kokkos_export_imported_tpl NAME)
  #make sure this also gets "exported" in the config file
  KOKKOS_APPEND_CONFIG_LINE("ADD_LIBRARY(${NAME} UNKNOWN IMPORTED)")
  KOKKOS_APPEND_CONFIG_LINE("SET_TARGET_PROPERTIES(${NAME} PROPERTIES")
  
  GET_TARGET_PROPERTY(TPL_COMPILE_OPTIONS ${NAME} INTERFACE_COMPILE_OPTIONS)
  SET(TPL_LINK_OPTIONS)
  IF(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.13.0")
    GET_TARGET_PROPERTY(TPL_LINK_OPTIONS ${NAME} INTERFACE_LINK_OPTIONS)
  ENDIF()
  GET_TARGET_PROPERTY(TPL_LINK_LIBRARIES  ${NAME} INTERFACE_LINK_LIBRARIES)
  GET_TARGET_PROPERTY(TPL_LIBRARIES ${NAME} IMPORTED_LOCATION)
  GET_TARGET_PROPERTY(TPL_INCLUDES ${NAME} INTERFACE_INCLUDE_DIRECTORIES)

  IF(TPL_LIBRARIES)
    KOKKOS_APPEND_CONFIG_LINE("IMPORTED_LOCATION ${TPL_LIBRARIES}")
  ENDIF()
  IF(TPL_INCLUDES)
    KOKKOS_APPEND_CONFIG_LINE("INTERFACE_INCLUDE_DIRECTORIES ${TPL_INCLUDES}")
  ENDIF()
  IF(TPL_COMPILE_OPTIONS)
    KOKKOS_APPEND_CONFIG_LINE("INTERFACE_COMPILE_OPTIONS ${TPL_COMPILE_OPTIONS}")
  ENDIF()
  IF(TPL_LINK_OPTIONS)
    KOKKOS_APPEND_CONFIG_LINE("INTERFACE_LINK_OPTIONS ${TPL_LINK_OPTIONS}")
  ENDIF()
  IF(TPL_LINK_LIBRARIES)
    KOKKOS_APPEND_CONFIG_LINE("INTERFACE_LINK_LIBRARIES ${TPL_LINK_LIBRARIES}")
  ENDIF()
  KOKKOS_APPEND_CONFIG_LINE(")")
ENDFUNCTION()

FUNCTION(kokkos_import_tpl MODULE_NAME IMPORTED_NAME)
  CMAKE_PARSE_ARGUMENTS(TPL
   "NO_EXPORT"
   ""
   ""
   ${ARGN})
  IF (KOKKOS_ENABLE_${MODULE_NAME})
    FIND_PACKAGE(${MODULE_NAME} REQUIRED MODULE)
    IF(NOT TARGET ${IMPORTED_NAME})
      MESSAGE(FATAL_ERROR "Find module succeeded for ${MODULE_NAME}, but did not produce valid target ${IMPORTED_NAME}")
    ENDIF()
    IF(NOT TPL_NO_EXPORT)
      KOKKOS_EXPORT_IMPORTED_TPL(${IMPORTED_NAME})
    ENDIF()
  ENDIF()
ENDFUNCTION(kokkos_import_tpl)

MACRO(kokkos_create_imported_tpl NAME)
  CMAKE_PARSE_ARGUMENTS(TPL
   ""
   ""
   "LIBRARIES;INCLUDES;COMPILE_OPTIONS;LINK_OPTIONS"
   ${ARGN})

  STRING(TOLOWER ${NAME} lc_name)
  ADD_LIBRARY(Kokkos::${lc_name} UNKNOWN IMPORTED)
  #make sure this also gets "exported" in the config file
  IF(TPL_LIBRARIES)
    SET_TARGET_PROPERTIES(Kokkos::${lc_name} PROPERTIES
      IMPORTED_LOCATION ${TPL_LIBRARIES})
  ENDIF()
  IF(TPL_INCLUDES)
    SET_TARGET_PROPERTIES(Kokkos::${lc_name} PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES ${TPL_INCLUDES})
  ENDIF()
  IF(TPL_COMPILE_OPTIONS)
    SET_TARGET_PROPERTIES(Kokkos::${lc_name} PROPERTIES
      INTERFACE_COMPILE_OPTIONS ${TPL_COMPILE_OPTIONS})
  ENDIF()
  IF(TPL_LINK_OPTIONS)
    SET_TARGET_PROPERTIES(Kokkos::${lc_name} PROPERTIES
      INTERFACE_LINK_LIBRARIES ${TPL_LINK_OPTIONS})
  ENDIF()
ENDMACRO()

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

  KOKKOS_CREATE_IMPORTED_TPL(${NAME}
    INCLUDES "${${NAME}_INCLUDE_DIR}"
    LIBRARIES "${${NAME}_LIBRARIES}")
ENDMACRO(kokkos_find_imported)



