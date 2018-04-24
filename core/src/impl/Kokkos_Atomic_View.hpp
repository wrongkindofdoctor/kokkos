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
#ifndef KOKKOS_ATOMIC_VIEW_HPP
#define KOKKOS_ATOMIC_VIEW_HPP

#include <Kokkos_Macros.hpp>
#include <Kokkos_Atomic.hpp>

namespace Kokkos { namespace Impl {

//The following tag is used to prevent an implicit call of the constructor when trying
//to assign a literal 0 int ( = 0 );
struct AtomicViewConstTag {};

template<class ViewTraits>
class AtomicDataElement
{
public:
  typedef typename ViewTraits::value_type value_type;
  typedef typename ViewTraits::const_value_type const_value_type;
  typedef typename ViewTraits::non_const_value_type non_const_value_type;
  value_type* const ptr;

private:
  static constexpr non_const_value_type one = static_cast<non_const_value_type>(1);

public:


  KOKKOS_FORCEINLINE_FUNCTION
  AtomicDataElement(value_type* ptr_, AtomicViewConstTag )
   : ptr{ptr_}
  {}

  KOKKOS_FORCEINLINE_FUNCTION
  const_value_type operator = (const_value_type val) const {
    atomic_store( ptr, val, memory_order_relaxed );
    return val;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  void inc() const {
    atomic_increment(ptr, memory_order_relaxed);
  }

  KOKKOS_FORCEINLINE_FUNCTION
  void dec() const {
    atomic_decrement(ptr, memory_order_relaxed);
  }

  KOKKOS_FORCEINLINE_FUNCTION
  const_value_type operator ++ () const {
    return atomic_add_fetch(ptr, one, memory_order_relaxed );
  }

  KOKKOS_FORCEINLINE_FUNCTION
  const_value_type operator -- () const {
    return atomic_sub_fetch(ptr, one, memory_order_relaxed );
  }

  KOKKOS_FORCEINLINE_FUNCTION
  const_value_type operator ++ (int) const {
    return atomic_fetch_add(ptr, one, memory_order_relaxed);
  }

  KOKKOS_FORCEINLINE_FUNCTION
  const_value_type operator -- (int) const {
    return atomic_fetch_sub(ptr, one, memory_order_relaxed);
  }

  KOKKOS_FORCEINLINE_FUNCTION
  const_value_type operator += (const_value_type val) const {
    return atomic_add_fetch(ptr, val, memory_order_relaxed );
  }

  KOKKOS_FORCEINLINE_FUNCTION
  const_value_type operator -= (const_value_type val) const {
    return atomic_sub_fetch(ptr, val, memory_order_relaxed );
  }

  KOKKOS_FORCEINLINE_FUNCTION
  const_value_type operator *= (const_value_type val) const {
    return atomic_mul_fetch(ptr,val,memory_order_relaxed);
  }

  KOKKOS_FORCEINLINE_FUNCTION
  const_value_type operator /= (const_value_type val) const {
    return atomic_div_fetch(ptr,val,memory_order_relaxed);
  }

  KOKKOS_FORCEINLINE_FUNCTION
  const_value_type operator %= (const_value_type val) const {
    return atomic_mod_fetch(ptr,val,memory_order_relaxed);
  }

  KOKKOS_FORCEINLINE_FUNCTION
  const_value_type operator &= (const_value_type val) const {
    return atomic_and_fetch(ptr,val,memory_order_relaxed);
  }

  KOKKOS_FORCEINLINE_FUNCTION
  const_value_type operator ^= (const_value_type val) const {
    return atomic_xor_fetch(ptr,val,memory_order_relaxed);
  }

  KOKKOS_FORCEINLINE_FUNCTION
  const_value_type operator |= (const_value_type val) const {
    return atomic_or_fetch(ptr,val,memory_order_relaxed);
  }

  KOKKOS_FORCEINLINE_FUNCTION
  const_value_type operator <<= (unsigned int val) const {
    return atomic_lshift_fetch(ptr,val,memory_order_relaxed);
  }

  KOKKOS_FORCEINLINE_FUNCTION
  const_value_type operator >>= (unsigned int val) const {
    return atomic_rshift_fetch(ptr,val,memory_order_relaxed);
  }

  KOKKOS_FORCEINLINE_FUNCTION
  const_value_type operator + (const_value_type val) const {
    return atomic_load(ptr,memory_order_relaxed)+val;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  const_value_type operator - (const_value_type val) const {
    return atomic_load(ptr,memory_order_relaxed)-val;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  const_value_type operator * (const_value_type val) const {
    return atomic_load(ptr,memory_order_relaxed)*val;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  const_value_type operator / (const_value_type val) const {
    return atomic_load(ptr,memory_order_relaxed)/val;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  const_value_type operator % (const_value_type val) const {
    return atomic_load(ptr,memory_order_relaxed)^val;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  const_value_type operator ! () const {
    return !atomic_load(ptr,memory_order_relaxed);
  }

  KOKKOS_FORCEINLINE_FUNCTION
  const_value_type operator && (const_value_type val) const {
    return atomic_load(ptr,memory_order_relaxed)&&val;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  const_value_type operator || (const_value_type val) const {
    return atomic_load(ptr,memory_order_relaxed)|val;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  const_value_type operator & (const_value_type val) const {
    return atomic_load(ptr,memory_order_relaxed)&val;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  const_value_type operator | (const_value_type val) const {
    return atomic_load(ptr,memory_order_relaxed)|val;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  const_value_type operator ^ (const_value_type val) const {
    return atomic_load(ptr,memory_order_relaxed)^val;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  const_value_type operator ~ () const {
    return ~atomic_load(ptr,memory_order_relaxed);
  }

  KOKKOS_FORCEINLINE_FUNCTION
  const_value_type operator << (const unsigned int val) const {
    return atomic_load(ptr,memory_order_relaxed)<<val;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  const_value_type operator >> (const unsigned int val) const {
    return atomic_load(ptr,memory_order_relaxed)>>val;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  bool operator == (const_value_type val) const {
    return atomic_load(ptr,memory_order_relaxed) == val;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  bool operator != (const_value_type val) const {
    return atomic_load(ptr,memory_order_relaxed) != val;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  bool operator >= (const_value_type val) const {
    return atomic_load(ptr,memory_order_relaxed) >= val;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  bool operator <= (const_value_type val) const {
    return atomic_load(ptr,memory_order_relaxed) <= val;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  bool operator < (const_value_type val) const {
    return atomic_load(ptr,memory_order_relaxed) < val;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  bool operator > (const_value_type val) const {
    return atomic_load(ptr,memory_order_relaxed) > val;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  operator const_value_type () const {
    return atomic_load(ptr,memory_order_relaxed);
  }
};

template<class ViewTraits>
class AtomicViewDataHandle {
public:
  typename ViewTraits::value_type* ptr;

  KOKKOS_INLINE_FUNCTION
  AtomicViewDataHandle()
    : ptr(nullptr)
  {}

  KOKKOS_INLINE_FUNCTION
  AtomicViewDataHandle(typename ViewTraits::value_type* ptr_)
    :ptr(ptr_)
  {}

  template<class iType>
  KOKKOS_INLINE_FUNCTION
  AtomicDataElement<ViewTraits> operator[] (const iType& i) const {
    return AtomicDataElement<ViewTraits>(ptr+i,AtomicViewConstTag());
  }


  KOKKOS_INLINE_FUNCTION
  operator typename ViewTraits::value_type * () const { return ptr ; }

};

template<unsigned Size>
struct Kokkos_Atomic_is_only_allowed_with_32bit_and_64bit_scalars;

template<>
struct Kokkos_Atomic_is_only_allowed_with_32bit_and_64bit_scalars<4> {
  typedef int type;
};

template<>
struct Kokkos_Atomic_is_only_allowed_with_32bit_and_64bit_scalars<8> {
  typedef int64_t type;
};

}} // namespace Kokkos::Impl

#endif

