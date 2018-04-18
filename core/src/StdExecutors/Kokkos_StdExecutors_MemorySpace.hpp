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

#ifndef KOKKOS_STDEXECUTORS_KOKKOS_STDEXECUTORS_MEMORYSPACE_HPP
#define KOKKOS_STDEXECUTORS_KOKKOS_STDEXECUTORS_MEMORYSPACE_HPP

#include <Kokkos_Macros.hpp>

#ifdef KOKKOS_ENABLE_STDEXECUTORS

#include <Kokkos_Concepts.hpp>

#include <experimental/execution>

#include <memory>

namespace Kokkos {
namespace Experimental {

// Forward declaration
template <typename Executor>
class StdExecutors;

template <typename Executor>
class StdExecutorsMemorySpace {

private:

  using proto_alloc_t = std::decay_t<decltype(
    std::experimental::execution::query(
      std::declval<Executor&>(),
      std::experimental::execution::allocator
    )
  )>;
  using proto_alloc_traits = std::allocator_traits<proto_alloc_t>;

public:

  using allocator_type = typename proto_alloc_t::template rebind<char>::other;

private:

  using alloc_traits = std::allocator_traits<allocator_type>;

public:


  using memory_space = StdExecutorsMemorySpace;
  using execution_space = StdExecutors<Executor>;
  using device_type = Kokkos::Device<execution_space, memory_space>;


  using size_type = typename alloc_traits::size_type;

  StdExecutorsMemorySpace() = default;
  StdExecutorsMemorySpace(StdExecutorsMemorySpace const&) = default;
  StdExecutorsMemorySpace(StdExecutorsMemorySpace&&) = default;
  StdExecutorsMemorySpace& operator=(StdExecutorsMemorySpace const&) = default;
  StdExecutorsMemorySpace& operator=(StdExecutorsMemorySpace&&) = default;
  ~StdExecutorsMemorySpace() noexcept = default;

  explicit
  StdExecutorsMemorySpace(allocator_type const& arg_alloc)
    : m_alloc(arg_alloc)
  { }

  void* allocate(size_t const arg_alloc_size) const {
    return alloc_traits::allocate(m_alloc, arg_alloc_size);
  }

  void deallocate(void* const arg_alloc_ptr, size_t const arg_alloc_size) const {
    alloc_traits::deallocate(m_alloc, static_cast<char*>(arg_alloc_ptr), arg_alloc_size);
  }

  static constexpr const char* name() { return "StdExecutorsMemorySpace"; }

private:
  mutable allocator_type m_alloc;

};

} // end namespace Experimental
} // end namespace Kokkos

#endif // KOKKOS_ENABLE_STDEXECUTORS

#endif //KOKKOS_STDEXECUTORS_KOKKOS_STDEXECUTORS_MEMORYSPACE_HPP
