/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2014) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef KOKKOS_STDEXECUTORS_KOKKOS_STDEXECUTORS_MEMORYSPACE_IMPL_HPP
#define KOKKOS_STDEXECUTORS_KOKKOS_STDEXECUTORS_MEMORYSPACE_IMPL_HPP

#include <StdExecutors/Kokkos_StdExecutors_MemorySpace.hpp>
#include <StdExecutors/Kokkos_StdExecutors_Properties.hpp>
#include <impl/Kokkos_SharedAlloc.hpp>

namespace Kokkos {
namespace Impl {

template <typename ExecutionSpace, bool Enabled=true> struct TrivialDeepCopyImpl;

template <typename ExecutionSpace>
struct TrivialDeepCopyImpl<ExecutionSpace, true> {
  TrivialDeepCopyImpl(void* dst, void const* src, size_t n) {
    memcpy(dst, src, n);
  }
  TrivialDeepCopyImpl(ExecutionSpace const& exec, void* dst, void const* src, size_t n) {
    exec.fence();
    memcpy(dst, src, n);
  }
};

template <typename ExecutionSpace>
struct TrivialDeepCopyImpl<ExecutionSpace, false>;

template <typename Executor, typename ExecutionSpace>
struct DeepCopy<
  Experimental::StdExecutorsMemorySpace<Executor>,
  Experimental::StdExecutorsMemorySpace<Executor>,
  ExecutionSpace
> : TrivialDeepCopyImpl<ExecutionSpace>
{
  using base_t = TrivialDeepCopyImpl<ExecutionSpace>;
  using base_t::base_t;
};

template <typename Executor, typename ExecutionSpace>
struct DeepCopy<
  Experimental::StdExecutorsMemorySpace<Executor>,
  HostSpace,
  ExecutionSpace
> : TrivialDeepCopyImpl<
      ExecutionSpace,
      Kokkos::Experimental::Impl::allocates_in_host_address_space_t::template static_query_v<Executor>
    >
{
  using base_t = TrivialDeepCopyImpl<ExecutionSpace>;
  using base_t::base_t;
};

template <typename Executor, typename ExecutionSpace>
struct DeepCopy<
  HostSpace,
  Experimental::StdExecutorsMemorySpace<Executor>,
  ExecutionSpace
> : TrivialDeepCopyImpl<
      ExecutionSpace,
      Kokkos::Experimental::Impl::allocates_in_host_address_space_t::template static_query_v<Executor>
    >
{
  using base_t = TrivialDeepCopyImpl<ExecutionSpace>;
  using base_t::base_t;
};

} // end namespace Impl
} // end namespace Kokkos

namespace Kokkos {
namespace Impl {

template<typename Executor>
class SharedAllocationRecord<Kokkos::Experimental::StdExecutorsMemorySpace<Executor>, void>
  : public SharedAllocationRecord<void, void>
{
private:
  using memory_space = Kokkos::Experimental::StdExecutorsMemorySpace<Executor>;
  friend Kokkos::Experimental::StdExecutorsMemorySpace<Executor>;

  using RecordBase = SharedAllocationRecord<void, void>;

  SharedAllocationRecord(SharedAllocationRecord const&) = delete;
  SharedAllocationRecord& operator=( const SharedAllocationRecord & ) = delete;

  static void deallocate( RecordBase * );

#ifdef KOKKOS_DEBUG
  /**\brief  Root record for tracked allocations from this ExecutorsMemorySpace instance */
  static RecordBase s_root_record;
#endif

  memory_space const m_space;

protected:
  ~SharedAllocationRecord() {
#if defined(KOKKOS_ENABLE_PROFILING)
    if(Kokkos::Profiling::profileLibraryLoaded()) {
      Kokkos::Profiling::deallocateData(
        Kokkos::Profiling::SpaceHandle(memory_space::name()),
        RecordBase::m_alloc_ptr->m_label,
        data(),size());
    }
#endif

    m_space.deallocate( SharedAllocationRecord< void , void >::m_alloc_ptr
      , SharedAllocationRecord< void , void >::m_alloc_size
    );
  }

  SharedAllocationRecord() = default;

  SharedAllocationRecord(
    const memory_space& arg_space,
    const std::string& arg_label,
    const size_t arg_alloc_size,
    const RecordBase::function_type arg_dealloc = &deallocate
  ) : SharedAllocationRecord< void , void >(
#ifdef KOKKOS_DEBUG
        &s_root_record,
#endif
        reinterpret_cast<SharedAllocationHeader*>( arg_space.allocate( sizeof(SharedAllocationHeader) + arg_alloc_size ) ),
        sizeof(SharedAllocationHeader) + arg_alloc_size,
        arg_dealloc
      ),
      m_space( arg_space )
    {
#if defined(KOKKOS_ENABLE_PROFILING)
      if(Kokkos::Profiling::profileLibraryLoaded()) {
        Kokkos::Profiling::allocateData(
          Kokkos::Profiling::SpaceHandle(arg_space.name()), arg_label, data(), arg_alloc_size
        );
      }
#endif
      // Fill in the Header information
      RecordBase::m_alloc_ptr->m_record = static_cast< SharedAllocationRecord< void , void > * >( this );

      strncpy(RecordBase::m_alloc_ptr->m_label, arg_label.c_str(), SharedAllocationHeader::maximum_label_length);
    }

public:

  inline
  std::string get_label() const
  {
    return std::string(RecordBase::head()->m_label);
  }

  KOKKOS_INLINE_FUNCTION static
  SharedAllocationRecord* allocate(
    memory_space const& arg_space,
    std::string const& arg_label,
    size_t const arg_alloc_size
  )
  {
    return new SharedAllocationRecord(arg_space, arg_label, arg_alloc_size);
  }


  /**\brief  Allocate tracked memory in the space */
  static
  void* allocate_tracked(
    memory_space const& arg_space,
    std::string const& arg_alloc_label,
    size_t const arg_alloc_size
  ) {
    if (!arg_alloc_size) return nullptr;

    SharedAllocationRecord* const r = allocate(arg_space, arg_alloc_label, arg_alloc_size);
    RecordBase::increment(r);

    return r->data();
  }

  /**\brief  Reallocate tracked memory in the space */
  static
  void* reallocate_tracked(void* const arg_alloc_ptr, const size_t arg_alloc_size) {
    SharedAllocationRecord* const r_old = get_record(arg_alloc_ptr);
    SharedAllocationRecord* const r_new = allocate(r_old->m_space, r_old->get_label(), arg_alloc_size);

    Kokkos::Impl::DeepCopy<memory_space, memory_space>(
      r_new->data(), r_old->data(), std::min(r_old->size(), r_new->size())
    );

    RecordBase::increment(r_new);
    RecordBase::decrement(r_old);

    return r_new->data();
  }

  /**\brief  Deallocate tracked memory in the space */
  static
  void deallocate_tracked(void* const arg_alloc_ptr) {
    if (arg_alloc_ptr != nullptr) {
      SharedAllocationRecord* const r = get_record(arg_alloc_ptr);
      RecordBase::decrement(r);
    }
  }

  static
  SharedAllocationRecord* get_record( void* alloc_ptr ) {
    using Header = SharedAllocationHeader;
    using RecordHost = SharedAllocationRecord<memory_space, void>;

    SharedAllocationHeader const* const head = alloc_ptr ? Header::get_header(alloc_ptr) : nullptr;
    RecordHost* const record = head ? static_cast<RecordHost*>(head->m_record) : nullptr;

    if (!alloc_ptr || record->m_alloc_ptr != head) {
      Kokkos::Impl::throw_runtime_exception(
        std::string("Kokkos::Impl::SharedAllocationRecord<Kokkos::StdExecutorsMemorySpace, void>::get_record ERROR")
      );
    }

    return record;
  }

  static void print_records( std::ostream &, const memory_space &, bool detail = false ) {
#ifdef KOKKOS_DEBUG
    SharedAllocationRecord<void, void>::print_host_accessible_records(s, memory_space::name, &s_root_record, detail);
#else
    throw_runtime_exception("SharedAllocationRecord<StdExecutorsMemorySpace>::print_records only works with KOKKOS_DEBUG enabled");
#endif
  }
};

#ifdef KOKKOS_DEBUG
template <typename Executor>
SharedAllocationRecord<void, void> SharedAllocationRecord<Kokkos::Experimental::StdExecutorsMemorySpace<Executor>, void>::s_root_record;
#endif

} // end namespace Impl
} // end namespace Kokkos


#endif //KOKKOS_STDEXECUTORS_KOKKOS_STDEXECUTORS_MEMORYSPACE_IMPL_HPP
