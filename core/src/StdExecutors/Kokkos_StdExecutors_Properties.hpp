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

#ifndef KOKKOS_STDEXECUTORS_KOKKOS_STDEXECUTORS_PROPERTIES_HPP
#define KOKKOS_STDEXECUTORS_KOKKOS_STDEXECUTORS_PROPERTIES_HPP

#include <Kokkos_Macros.hpp>

#ifdef KOKKOS_ENABLE_STDEXECUTORS

#include <experimental/execution>
#include <experimental/thread_pool>

namespace Kokkos {
namespace Experimental {
namespace Impl {

template <typename Executor, typename Enable=void>
struct static_query_allocates_in_host_address_space : std::false_type { };


struct allocates_in_host_address_space_t {
  static constexpr bool is_requirable = false;
  static constexpr bool is_preferable = false;

  using polymorphic_query_result_type = bool;

  template <class Executor>
  static constexpr auto static_query_v =
    static_query_allocates_in_host_address_space<Executor>::value;

};

constexpr allocates_in_host_address_space_t allocates_in_host_address_space = { };

template <>
struct static_query_allocates_in_host_address_space<
  std::experimental::static_thread_pool::executor_type, void
> : std::true_type
{ };

} // end namespace Impl
} // end namespace Experimental
} // end namespace Kokkos

#endif // KOKKOS_ENABLE_STDEXECUTORS

#endif //KOKKOS_STDEXECUTORS_KOKKOS_STDEXECUTORS_PROPERTIES_HPP
