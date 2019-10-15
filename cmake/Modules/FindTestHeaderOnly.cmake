#used only for testing that importing header-only libs works
KOKKOS_CREATE_IMPORTED_TPL(testHeader INCLUDES fakeFolder)
KOKKOS_EXPORT_IMPORTED_TPL(Kokkos::testheader)

