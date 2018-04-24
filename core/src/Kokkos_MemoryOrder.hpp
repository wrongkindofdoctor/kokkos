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

#ifndef KOKKOS_MEMORY_ORDER_HPP
#define KOKKOS_MEMORY_ORDER_HPP

#include <Kokkos_Macros.hpp>
#include <type_traits>

namespace Kokkos {

//------------------------------------------------------------------------------
// Memory Orders
//
// The memory orders are different types to force detection at compile time since
// some compilers map any runtime value to memory_order_seq_cst
//------------------------------------------------------------------------------
namespace Impl {

struct MemoryOrderRelaxed { static constexpr int value = __ATOMIC_RELAXED; };
struct MemoryOrderAcquire { static constexpr int value = __ATOMIC_ACQUIRE; };
struct MemoryOrderRelease { static constexpr int value = __ATOMIC_RELEASE; };
struct MemoryOrderAcqRel  { static constexpr int value = __ATOMIC_ACQ_REL; };
struct MemoryOrderSeqCst  { static constexpr int value = __ATOMIC_SEQ_CST; };

// valid memory orders
template <typename T> struct valid_memory_order           : public std::false_type {};
template <> struct valid_memory_order<MemoryOrderRelaxed> : public std::true_type {};
template <> struct valid_memory_order<MemoryOrderAcquire> : public std::true_type {};
template <> struct valid_memory_order<MemoryOrderRelease> : public std::true_type {};
template <> struct valid_memory_order<MemoryOrderAcqRel>  : public std::true_type {};
template <> struct valid_memory_order<MemoryOrderSeqCst>  : public std::true_type {};

// valid memory orders for atomic_load
template <typename Order> struct valid_atomic_load_order         : public std::false_type {};
template <> struct valid_atomic_load_order< MemoryOrderRelaxed > : public std::true_type  {};
template <> struct valid_atomic_load_order< MemoryOrderAcquire > : public std::true_type  {};
template <> struct valid_atomic_load_order< MemoryOrderSeqCst >  : public std::true_type  {};

// valid memory orders for atomic_store
template <typename Order> struct valid_atomic_store_order         : public std::false_type {};
template <> struct valid_atomic_store_order< MemoryOrderRelaxed > : public std::true_type  {};
template <> struct valid_atomic_store_order< MemoryOrderRelease > : public std::true_type  {};
template <> struct valid_atomic_store_order< MemoryOrderSeqCst >  : public std::true_type  {};

// valid memory orders for atomic_compare_exchange
template <typename SuccessOrder, typename FailureOrder> struct valid_atomic_compare_exchange_order : public std::false_type {};
template <> struct valid_atomic_compare_exchange_order< MemoryOrderRelaxed, MemoryOrderRelaxed >   : public std::true_type  {};
template <> struct valid_atomic_compare_exchange_order< MemoryOrderAcquire, MemoryOrderRelaxed >   : public std::true_type  {};
template <> struct valid_atomic_compare_exchange_order< MemoryOrderRelease, MemoryOrderRelaxed >   : public std::true_type  {};
template <> struct valid_atomic_compare_exchange_order< MemoryOrderAcqRel , MemoryOrderRelaxed >   : public std::true_type  {};
template <> struct valid_atomic_compare_exchange_order< MemoryOrderSeqCst , MemoryOrderRelaxed >   : public std::true_type  {};
template <> struct valid_atomic_compare_exchange_order< MemoryOrderAcquire, MemoryOrderAcquire >   : public std::true_type  {};
template <> struct valid_atomic_compare_exchange_order< MemoryOrderRelease, MemoryOrderAcquire >   : public std::true_type  {};
template <> struct valid_atomic_compare_exchange_order< MemoryOrderAcqRel , MemoryOrderAcquire >   : public std::true_type  {};
template <> struct valid_atomic_compare_exchange_order< MemoryOrderSeqCst , MemoryOrderAcquire >   : public std::true_type  {};
template <> struct valid_atomic_compare_exchange_order< MemoryOrderSeqCst , MemoryOrderSeqCst  >   : public std::true_type  {};

} // namespace Impl

// Implies no inter-thread ordering constraints
constexpr Impl::MemoryOrderRelaxed memory_order_relaxed {};

// Creates an inter-thread happens-before constraint from the release (or
// stronger) semantic store to this acquire load. Can prevent hoisting of code
// to before the operation.
constexpr Impl::MemoryOrderAcquire memory_order_acquire {};

// Creates an inter-thread happens-before constraint to acquire (or stronger)
// semantic loads that read from this release store. Can prevent sinking of code
// to after the operation.
constexpr Impl::MemoryOrderRelease memory_order_release {};

// Combines the effect of memory_order_acquire and memory_order_release
constexpr Impl::MemoryOrderAcqRel memory_order_acq_rel {};

// Enforces total ordering with all other memory_order_seq_cst operations
constexpr Impl::MemoryOrderSeqCst memory_order_seq_cst {};

} // namespace Kokkos

#endif // KOKKOS_MEMORY_ORDER_HPP

