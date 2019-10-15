#used only in validating that importing link-only libraries works
KOKKOS_CREATE_IMPORTED_TPL(testlibrary LIBRARIES testLib)
KOKKOS_EXPORT_IMPORTED_TPL(Kokkos::testlibrary)

