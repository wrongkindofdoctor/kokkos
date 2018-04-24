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

#ifndef KOKKOS_IMPL_ATOMIC_DEPRECATED_HPP
#define KOKKOS_IMPL_ATOMIC_DEPRECATED_HPP

#include <Kokkos_Macros.hpp>
#include <type_traits>

#if defined( KOKKOS_ENABLE_DEPRECATED_CODE )

namespace Kokkos {

template <typename T>
KOKKOS_INLINE_FUNCTION
T atomic_compare_exchange( volatile T * ptr, typename std::remove_cv<T>::type compare, typename std::remove_cv<T>::type val)
{
  T expected = compare;
  T desired = val;
  const bool result =atomic_compare_exchange( const_cast<T*>(ptr)
                                             , &expected
                                             , desired
                                             , memory_order_acq_rel
                                             , memory_order_relaxed
                                             );

  return result ? val : expected;
}

template <typename T>
KOKKOS_INLINE_FUNCTION
bool atomic_compare_exchange_strong(volatile T* ptr, typename std::remove_cv<T>::type expected, typename std::remove_cv<T>::type val)
{
  return atomic_compare_exchange( const_cast<T*>(ptr)
                                , &expected
                                , val
                                , memory_order_acq_rel
                                , memory_order_relaxed
                                );
}

template < typename T >
KOKKOS_INLINE_FUNCTION
T atomic_exchange( volatile T * ptr , typename std::remove_cv<T>::type val )
{
  return atomic_exchange( const_cast<T*>(ptr), val );
}

template < typename T >
KOKKOS_INLINE_FUNCTION
void atomic_assign( volatile T * ptr , typename std::remove_cv<T>::type val )
{
  atomic_store( const_cast<T*>(ptr), val );
}


template<typename T>
KOKKOS_INLINE_FUNCTION
void atomic_increment(volatile T* a)
{
  atomic_increment( const_cast<T*>(a) );
}

template<typename T>
KOKKOS_INLINE_FUNCTION
void atomic_decrement(volatile T* a)
{
  atomic_decrement( const_cast<T*>(a) );
}


template <typename T>
KOKKOS_INLINE_FUNCTION
void atomic_add(volatile T * ptr, typename std::remove_cv<T>::type val)
{
  atomic_add( const_cast<T*>(ptr), val );
}

template< typename T >
KOKKOS_INLINE_FUNCTION
T atomic_fetch_add( volatile T * ptr , typename std::remove_cv<T>::type val )
{
  return atomic_fetch_add( const_cast<T*>(ptr), val );
}

template< typename T >
KOKKOS_INLINE_FUNCTION
T atomic_fetch_sub( volatile T * ptr , typename std::remove_cv<T>::type val )
{
  return atomic_fetch_sub( const_cast<T*>(ptr), val);
}

template <typename T>
KOKKOS_INLINE_FUNCTION
void atomic_sub(volatile T * ptr, typename std::remove_cv<T>::type val)
{
  atomic_sub( const_cast<T*>(ptr), val);
}

template < typename T >
KOKKOS_INLINE_FUNCTION
T atomic_fetch_max(volatile T * ptr, typename std::remove_cv<T>::type val)
{
  return atomic_fetch_max( const_cast<T*>(ptr), val);
}

template < typename T >
KOKKOS_INLINE_FUNCTION
T atomic_fetch_min(volatile T * ptr, typename std::remove_cv<T>::type val)
{
  return atomic_fetch_min( const_cast<T*>(ptr), val);
}

template < typename T >
KOKKOS_INLINE_FUNCTION
T atomic_fetch_mul(volatile T * ptr, typename std::remove_cv<T>::type val)
{
  return atomic_fetch_mul( const_cast<T*>(ptr), val);
}

template < typename T >
KOKKOS_INLINE_FUNCTION
T atomic_fetch_div(volatile T * ptr, typename std::remove_cv<T>::type val)
{
  return atomic_fetch_div( const_cast<T*>(ptr), val);
}

template < typename T >
KOKKOS_INLINE_FUNCTION
T atomic_fetch_mod(volatile T * ptr, typename std::remove_cv<T>::type val)
{
  return atomic_fetch_mod( const_cast<T*>(ptr), val);
}

template< typename T >
KOKKOS_INLINE_FUNCTION
T atomic_fetch_and( volatile T * ptr , typename std::remove_cv<T>::type val )
{
  return atomic_fetch_and( const_cast<T*>(ptr), val);
}

template <typename T>
KOKKOS_INLINE_FUNCTION
void atomic_and(volatile T * ptr, typename std::remove_cv<T>::type val)
{
  atomic_and( const_cast<T*>(ptr), val);
}

template< typename T >
KOKKOS_INLINE_FUNCTION
T atomic_fetch_or( volatile T * ptr , typename std::remove_cv<T>::type val )
{
  return atomic_fetch_or( const_cast<T*>(ptr), val);
}

template <typename T>
KOKKOS_INLINE_FUNCTION
void atomic_or(volatile T * ptr, typename std::remove_cv<T>::type val)
{
  atomic_or( const_cast<T*>(ptr), val);
}

template < typename T >
KOKKOS_INLINE_FUNCTION
T atomic_fetch_xor(volatile T * ptr, typename std::remove_cv<T>::type val)
{
  return atomic_fetch_xor( const_cast<T*>(ptr), val);
}

template < typename T >
KOKKOS_INLINE_FUNCTION
T atomic_fetch_lshift(volatile T * ptr, unsigned int val)
{
  return atomic_fetch_lshift( const_cast<T*>(ptr), val);
}

template < typename T >
KOKKOS_INLINE_FUNCTION
T atomic_fetch_rshift(volatile T * ptr, unsigned int val)
{
  return atomic_fetch_rshift( const_cast<T*>(ptr), val);
}

//------------------------------------------------------------------------------

template < typename T >
KOKKOS_INLINE_FUNCTION
T atomic_max_fetch(volatile T * ptr, typename std::remove_cv<T>::type val)
{
  return atomic_max_fetch( const_cast<T*>(ptr), val);
}

template < typename T >
KOKKOS_INLINE_FUNCTION
T atomic_min_fetch(volatile T * ptr, typename std::remove_cv<T>::type val)
{
  return atomic_min_fetch( const_cast<T*>(ptr), val);
}

template < typename T >
KOKKOS_INLINE_FUNCTION
T atomic_mul_fetch(volatile T * ptr, typename std::remove_cv<T>::type val)
{
  return atomic_mul_fetch( const_cast<T*>(ptr), val);
}

template < typename T >
KOKKOS_INLINE_FUNCTION
T atomic_div_fetch(volatile T * ptr, typename std::remove_cv<T>::type val)
{
  return atomic_div_fetch( const_cast<T*>(ptr), val);
}

template < typename T >
KOKKOS_INLINE_FUNCTION
T atomic_mod_fetch(volatile T * ptr, typename std::remove_cv<T>::type val)
{
  return atomic_mod_fetch( const_cast<T*>(ptr), val);
}

template < typename T >
KOKKOS_INLINE_FUNCTION
T atomic_and_fetch(volatile T * ptr, typename std::remove_cv<T>::type val)
{
  return atomic_and_fetch( const_cast<T*>(ptr), val);
}

template < typename T >
KOKKOS_INLINE_FUNCTION
T atomic_or_fetch(volatile T * ptr, typename std::remove_cv<T>::type val)
{
  return atomic_or_fetch( const_cast<T*>(ptr), val);
}

template < typename T >
KOKKOS_INLINE_FUNCTION
T atomic_xor_fetch(volatile T * ptr, typename std::remove_cv<T>::type val)
{
  return atomic_xor_fetch( const_cast<T*>(ptr), val);
}

template < typename T >
KOKKOS_INLINE_FUNCTION
T atomic_lshift_fetch(volatile T * ptr, unsigned int val)
{
  return atomic_lshift_fetch( const_cast<T*>(ptr), val);
}

template < typename T >
KOKKOS_INLINE_FUNCTION
T atomic_rshift_fetch(volatile T * ptr, unsigned int val)
{
  return atomic_rshift_fetch( const_cast<T*>(ptr), val);
}


KOKKOS_FORCEINLINE_FUNCTION
void memory_fence() noexcept
{
  atomic_thread_fence( memory_order_acq_rel );
}

KOKKOS_FORCEINLINE_FUNCTION
void store_fence() noexcept
{
  atomic_thread_fence( memory_order_release );
}

KOKKOS_FORCEINLINE_FUNCTION
void load_fence() noexcept
{
  atomic_thread_fence( memory_order_acquire );
}

template < typename T >
KOKKOS_FORCEINLINE_FUNCTION
T volatile_load(T* ptr) noexcept
{
  return atomic_load( ptr, memory_order_relaxed );
}

template < typename T >
KOKKOS_FORCEINLINE_FUNCTION
T volatile_load(volatile T* ptr) noexcept
{
  return atomic_load( const_cast<T*>(ptr), memory_order_relaxed );
}

} // namespace Kokkos

#endif

#endif // KOKKOS_IMPL_ATOMIC_DEPRECATED_HPP
