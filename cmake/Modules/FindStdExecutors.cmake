#.rst:
# FindMemkind
# ----------
#
# Try to find an implementation of the Executors proposal P0443
#
# The following variables are defined:
#
#   STDEXECUTORS_FOUND - System has executors
#   STDEXECUTORS_INCLUDE_DIR - Executors include directory

find_path(STDEXECUTORS_INCLUDE_DIR experimental/execution)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(StdExecutors DEFAULT_MSG
        STDEXECUTORS_INCLUDE_DIR)

mark_as_advanced(STDEXECUTORS_INCLUDE_DIR)
