#only used in testing for validating compile-flag only libs work
KOKKOS_CREATE_IMPORTED_TPL(testFlags COMPILE_OPTIONS -ffakeflag)
KOKKOS_EXPORT_IMPORTED_TPL(Kokkos::testflags)

