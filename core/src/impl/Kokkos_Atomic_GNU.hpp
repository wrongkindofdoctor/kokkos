#ifndef KOKKOS_IMPL_ATOMIC_GNU_HPP
#define KOKKOS_IMPL_ATOMIC_GNU_HPP

#include <Kokkos_Macros.hpp>
#include <Kokkos_MemoryOrder.hpp>
#include <impl/Kokkos_Atomic_Ops.hpp>

#include <type_traits>
#include <cstdint>
#include <cstddef>

namespace Kokkos {

//===============================================================================
// The 'atomic’ methods can be used with any integral scalar or pointer type
// that is 1, 2, 4, or 8 bytes in length. 16-byte integral types are also allowed
// if ‘__int128’ is supported by the architecture.

// The four non-arithmetic functions (load, store, exchange, and compare_exchange)
// all have a generic version as well. This generic version works on any
// 'trivially_copyable' data type.  It uses a lock-free built-in function if the
// specific data type size makes that possible; otherwise, an external call is
// left to be resolved at run time.
//
// All atomic operations require that the data is 'naturally' or 'trivially'
// aligned, i.e. a data of type T is aligned to the sizeof(T).
//
// Note: volatile types are not 'trivially_copyable'.  To maintain backwards
// compatibility we provide deprecated volatile overloads for all atomic methods.
//
// See https://gcc.gnu.org/onlinedocs/gcc/_005f_005fatomic-Builtins.html
//===============================================================================


namespace Impl {

#if defined KOKKOS_ENABLE_INT128
constexpr size_t max_atomic_integer_size = sizeof( __int128 );
#else
constexpr size_t max_atomic_integer_size = sizeof( int64_t );
#endif

template <typename T>
struct explicit_atomic_op
  : public std::conditional<  !std::is_volatile<T>::value
                           &&  (  std::is_pointer<T>::value
                              || (std::is_integral<T>::value
                                 && (sizeof(T) <= max_atomic_integer_size) ) )
                           , std::true_type
                           , std::false_type
                           >::type
{};

template <typename T>
struct generic_atomic_op
  : public std::conditional<  !std::is_volatile<T>::value
                           &&  !(  std::is_pointer<T>::value
                               || (std::is_integral<T>::value
                                  && (sizeof(T) <= max_atomic_integer_size) ) )
                           , std::true_type
                           , std::false_type
                           >::type
{};

template <typename T>
struct arithmetic_atomic_op
  : public std::conditional<  !std::is_volatile<T>::value
                           && (   std::is_pointer<T>::value
                              || (std::is_integral<T>::value
                                 && (sizeof(T) <= max_atomic_integer_size)
                                 && !std::is_same<T,bool>::value ) )
                           , std::true_type
                           , std::false_type
                           >::type
{};

template <typename T>
struct non_arithmetic_atomic_op
  : public std::conditional<  !std::is_volatile<T>::value
                           && !(  std::is_pointer<T>::value
                              || (std::is_integral<T>::value
                                 && (sizeof(T) <= max_atomic_integer_size)
                                 && !std::is_same<T,bool>::value ) )
                           , std::true_type
                           , std::false_type
                           >::type
{};

} // namespace Impl


//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
// T atomic_load( T * ptr, memorder = memory_order_acquire )
//
//   Atomically load and return contents of *ptr
//
//   Valid memory orders:
//     memory_order_relaxed, memory_order_acquire and memory_order_seq_cst
//------------------------------------------------------------------------------

template< typename T, typename Order>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if< Impl::explicit_atomic_op<T>::value, T>::type
atomic_load( T* ptr
           , Order = memory_order_acquire
           ) noexcept
{
  static_assert( Impl::valid_atomic_load_order<Order>::value
               , "Error: Invalid memory order for atomic_load" );
  static_assert( std::is_trivially_copyable<T>::value
               , "Error: atomic_load requires a trivially_copyable type" );
  return __atomic_load_n( ptr, Order::value );
}

template< typename T, typename Order>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if< Impl::generic_atomic_op<T>::value, T>::type
atomic_load( T* ptr
           , Order = memory_order_acquire
           ) noexcept
{
  static_assert( Impl::valid_atomic_load_order<Order>::value
               , "Error: Invalid memory order for atomic_load" );
  static_assert( std::is_trivially_copyable<T>::value
               , "Error: atomic_load requires a trivially_copyable type" );
  T ret;
  __atomic_load( ptr, &ret, Order::value );
  return ret;
}


//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
// void atomic_store( T * ptr, T val, memorder = memory_order_release )
//
//   Atomically store val into *ptr
//
//   Valid memory orders:
//     memory_order_relaxed, memory_order_release and memory_order_seq_cst
//------------------------------------------------------------------------------

template< typename T, typename Order>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if< Impl::explicit_atomic_op<T>::value, void>::type
atomic_store( T* ptr
            , typename std::remove_cv<T>::type val
            , Order = memory_order_release
            ) noexcept
{
  static_assert( Impl::valid_atomic_store_order<Order>::value
               , "Error: Invalid memory order for atomic_store" );
  static_assert( std::is_trivially_copyable<T>::value
               , "Error: atomic_store requires a trivially_copyable type" );
  __atomic_store_n( ptr, val, Order::value );
}

template< typename T, typename Order>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if< Impl::generic_atomic_op<T>::value, void>::type
atomic_store( T* ptr
            , typename std::remove_cv<T>::type val
            , Order = memory_order_release
            ) noexcept
{
  static_assert( Impl::valid_atomic_store_order<Order>::value
               , "Error: Invalid memory order for atomic_store" );
  static_assert( std::is_trivially_copyable<T>::value
               , "Error: atomic_store requires a trivially_copyable type" );
  __atomic_store( ptr, &val, Order::value );
}


//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
// T atomic_exchange( T * ptr, T val, memorder = memory_order_acq_rel )
//
//   Atomically writes val into *ptr
//   Return the previous contents of *ptr
//
//   Valid memory orders:
//     memory_order_relaxed, memory_order_acquire, memory_order_release,
//     memory_order_acq_rel and memory_order_seq_cst
//------------------------------------------------------------------------------

template< typename T, typename Order>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if< Impl::explicit_atomic_op<T>::value, T>::type
atomic_exchange( T* ptr
               , typename std::remove_cv<T>::type val
               , Order = memory_order_acq_rel
               ) noexcept
{
  static_assert( Impl::valid_memory_order<Order>::value
               , "Error: Invalid memory order for atomic_exchange" );
  static_assert( std::is_trivially_copyable<T>::value
               , "Error: atomic_exchange requires a trivially_copyable type" );
  return __atomic_exchange_n( ptr, val, Order::value );
}

template< typename T, typename Order>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if< Impl::generic_atomic_op<T>::value, T>::type
atomic_exchange( T* ptr
               , typename std::remove_cv<T>::type val
               , Order = memory_order_acq_rel ) noexcept
{
  static_assert( Impl::valid_memory_order<Order>::value
               , "Error: Invalid memory order for atomic_exchange" );
  static_assert( std::is_trivially_copyable<T>::value
               , "Error: atomic_exchange requires a trivially_copyable type" );
  T ret;
  __atomic_store( ptr, &val, &ret, Order::value );
  return ret;
}


//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
// bool atomic_compare_exchange( T * ptr, T * expected, T desired
//                             , memorder success = memory_order_acq_rel
//                             , memorder failure = memory_order_relaxed
//                             )
// bool atomic_compare_exchange_weak( T * ptr, T * expected, T desired
//                                  , memorder success = memory_order_acq_rel
//                                  , memorder failure = memory_order_relaxed
//                                  )
//
//   This compares the contents of *ptr with the contents of *expected.
//   If equal, the operation is a read-modify-write operation that writes desired
//   into *ptr. If they are not equal, the operation is a read and the current
//   contents of *ptr are written into *expected.
//
//   There are no restrictions on the success memory order.  The failure memory
//   order cannot be memory_order_release or memory_order_acq_rel.  It also cannot
//   be stronger that the success memory order
//
//   atomic_compare_exchange NEVER fails spuriously
//   atomic_compare_exchange_weak MAY fail spuriously
//------------------------------------------------------------------------------


template< typename T, typename SuccessOrder, typename FailureOrder>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if< Impl::explicit_atomic_op<T>::value, bool>::type
atomic_compare_exchange( T* ptr
                       , T* expected
                       , typename std::remove_cv<T>::type desired
                       , SuccessOrder = memory_order_acq_rel
                       , FailureOrder = memory_order_relaxed
                       ) noexcept
{
  static_assert( Impl::valid_atomic_compare_exchange_order<SuccessOrder,FailureOrder>::value
               , "Error: Invalid memory order for atomic_compare_exchange" );
  static_assert( std::is_trivially_copyable<T>::value
               , "Error: atomic_compare_exchange requires a trivially_copyable type" );
  return __atomic_compare_exchange_n( ptr, expected, desired
                                    , false, SuccessOrder::value, FailureOrder::value );
}

template< typename T, typename SuccessOrder, typename FailureOrder>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if< Impl::generic_atomic_op<T>::value, bool>::type
atomic_compare_exchange( T* ptr
                       , T* expected
                       , typename std::remove_cv<T>::type desired
                       , SuccessOrder = memory_order_acq_rel
                       , FailureOrder = memory_order_relaxed
                       ) noexcept
{
  static_assert( Impl::valid_atomic_compare_exchange_order<SuccessOrder,FailureOrder>::value
               , "Error: Invalid memory order for atomic_compare_exchange" );
  static_assert( std::is_trivially_copyable<T>::value
               , "Error: atomic_compare_exchange requires a trivially_copyable type" );
  return __atomic_compare_exchange( ptr, expected, &desired
                                  , false, SuccessOrder::value, FailureOrder::value );
}

template< typename T, typename SuccessOrder, typename FailureOrder>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if< Impl::explicit_atomic_op<T>::value, bool>::type
atomic_compare_exchange_weak( T* ptr
                            , typename std::remove_cv<T>::type * expected
                            , typename std::remove_cv<T>::type desired
                            , const SuccessOrder = memory_order_acq_rel
                            , const FailureOrder = memory_order_relaxed
                            ) noexcept
{
  static_assert( Impl::valid_atomic_compare_exchange_order<SuccessOrder,FailureOrder>::value
               , "Error: Invalid memory order for atomic_compare_exchange_weak" );
  static_assert( std::is_trivially_copyable<T>::value
               , "Error: atomic_compare_exchange_weak requires a trivially_copyable type" );
  return __atomic_compare_exchange_n( ptr, expected, desired
                                    , true, SuccessOrder::value, FailureOrder::value );
}

template< typename T, typename SuccessOrder, typename FailureOrder>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if< Impl::generic_atomic_op<T>::value, bool>::type
atomic_compare_exchange_weak( T* ptr
                            , typename std::remove_cv<T>::type * expected
                            , typename std::remove_cv<T>::type desired
                            , const SuccessOrder = memory_order_acq_rel
                            , const FailureOrder = memory_order_relaxed
                            ) noexcept
{
  static_assert( Impl::valid_atomic_compare_exchange_order<SuccessOrder,FailureOrder>::value
               , "Error: Invalid memory order for atomic_compare_exchange_weak" );
  static_assert( std::is_trivially_copyable<T>::value
               , "Error: atomic_compare_exchange_weak requires a trivially_copyable type" );
  return __atomic_compare_exchange( ptr, expected, &desired
                                  , true, SuccessOrder::value, FailureOrder::value );
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
// void atomic_thread_fence( memorder = memory_order_acq_rel ) noexcept
//
// This built-in function acts as a synchronization fence between threads based
// on the specified memory order.
//
// All memory orders are valid.
//------------------------------------------------------------------------------

template <typename Order>
KOKKOS_FORCEINLINE_FUNCTION
void atomic_thread_fence( Order = memory_order_acq_rel ) noexcept
{
  static_assert( Impl::valid_memory_order<Order>::value
               , "Error: Invalid memory order for atomic_thread_fence" );
  __atomic_thread_fence( Order::value );
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

namespace Impl {

template <typename Op, typename T, typename Order>
KOKKOS_FORCEINLINE_FUNCTION
T atomic_fetch_op( Op
                 , T * ptr
                 , typename std::remove_cv<T>::type val
                 , Order order = memory_order_acq_rel
                 ) noexcept
{
  static_assert( Impl::valid_memory_order<Order>::value
               , "Error: Invalid memory order for atomic_fetch_op" );
  static_assert( std::is_trivially_copyable<T>::value
               , "Error: atomic_fetch_op requires a trivially_copyable type" );

  T expected = atomic_load(ptr, memory_order_relaxed);
  T desired  = Op::apply(expected, val);

  while( !atomic_compare_exchange_weak( ptr, &expected, desired, order, memory_order_relaxed ) ) {
    desired = Op::apply(expected, val);
  }

  return expected;
}

template <typename Op, typename T, typename Order>
KOKKOS_FORCEINLINE_FUNCTION
T atomic_op_fetch( Op
                 , T * ptr
                 , typename std::remove_cv<T>::type val
                 , Order order = memory_order_acq_rel
                 ) noexcept
{
  static_assert( Impl::valid_memory_order<Order>::value
               , "Error: Invalid memory order for atomic_op_fetch" );
  static_assert( std::is_trivially_copyable<T>::value
               , "Error: atomic_op_fetch requires a trivially_copyable type" );

  T expected = atomic_load(ptr, memory_order_relaxed);
  T desired  = Op::apply(expected, val);

  while( !atomic_compare_exchange_weak( ptr, &expected, desired, order, memory_order_relaxed ) ) {
    desired = Op::apply(expected, val);
  }

  return desired;
}

} // namespace Impl



//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
// T atomic_fetch_and( T * ptr, T val, memorder = memory_order_acq_rel )
// T atomic_fetch_xor( T * ptr, T val, memorder = memory_order_acq_rel )
// T atomic_fetch_or( T * ptr, T val, memorder = memory_order_acq_rel )
// T atomic_fetch_nand( T * ptr, T val, memorder = memory_order_acq_rel )
//
//   These functions perform the operation suggested by the name, and return the
//   value that had previously been in *ptr.
//
// T atomic_and_fetch( T * ptr, T val, memorder = memory_order_acq_rel )
// T atomic_xor_fetch( T * ptr, T val, memorder = memory_order_acq_rel )
// T atomic_or_fetch( T * ptr, T val, memorder = memory_order_acq_rel )
// T atomic_nand_fetch( T * ptr, T val, memorder = memory_order_acq_rel )
//
//   These functions perform the operation suggested by the name, and return the
//   result of the operation.
//
// Operations on pointer arguments are performed as if the operands were of the
// uintptr_t type. That is, they are not scaled by the size of the type to which
// the pointer points.

template <typename T, typename Order>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if< Impl::arithmetic_atomic_op<T>::value, T>::type
atomic_fetch_and( T * ptr
                , typename std::remove_cv<T>::type val
                , Order = memory_order_acq_rel
                ) noexcept
{
  static_assert( Impl::valid_memory_order<Order>::value
               , "Error: Invalid memory order for atomic_fetch_and" );
  static_assert( std::is_trivially_copyable<T>::value
               , "Error: atomic_fetch_and requires a trivially_copyable type" );
  return __atomic_fetch_and( ptr, val, Order::value );
}

template <typename T, typename Order>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if< Impl::arithmetic_atomic_op<T>::value, T>::type
atomic_fetch_xor( T * ptr
                , typename std::remove_cv<T>::type val
                , Order = memory_order_acq_rel
                ) noexcept
{
  static_assert( Impl::valid_memory_order<Order>::value
               , "Error: Invalid memory order for atomic_fetch_xor" );
  static_assert( std::is_trivially_copyable<T>::value
               , "Error: atomic_fetch_xor requires a trivially_copyable type" );
  return __atomic_fetch_xor( ptr, val, Order::value );
}

template <typename T, typename Order>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if< Impl::arithmetic_atomic_op<T>::value, T>::type
atomic_fetch_or( T * ptr
              , typename std::remove_cv<T>::type val
              , Order = memory_order_acq_rel
              ) noexcept
{
  static_assert( Impl::valid_memory_order<Order>::value
               , "Error: Invalid memory order for atomic_fetch_or" );
  static_assert( std::is_trivially_copyable<T>::value
               , "Error: atomic_fetch_or requires a trivially_copyable type" );
  return __atomic_fetch_or( ptr, val, Order::value );
}

template <typename T, typename Order>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if< Impl::arithmetic_atomic_op<T>::value, T>::type
atomic_fetch_nand( T * ptr
                 , typename std::remove_cv<T>::type val
                 , Order = memory_order_acq_rel
                 ) noexcept
{
  static_assert( Impl::valid_memory_order<Order>::value
               , "Error: Invalid memory order for atomic_fetch_nand" );
  static_assert( std::is_trivially_copyable<T>::value
               , "Error: atomic_fetch_nand requires a trivially_copyable type" );
  return __atomic_fetch_nand( ptr, val, Order::value );
}

template <typename T, typename Order>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if< Impl::arithmetic_atomic_op<T>::value, T>::type
atomic_and_fetch( T * ptr
                , typename std::remove_cv<T>::type val
                , Order = memory_order_acq_rel
                ) noexcept
{
  static_assert( Impl::valid_memory_order<Order>::value
               , "Error: Invalid memory order for atomic_and_fetch" );
  static_assert( std::is_trivially_copyable<T>::value
               , "Error: atomic_and_fetch requires a trivially_copyable type" );
  return __atomic_and_fetch( ptr, val, Order::value );
}

template <typename T, typename Order>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if< Impl::arithmetic_atomic_op<T>::value, T>::type
atomic_xor_fetch( T * ptr
                , typename std::remove_cv<T>::type val
                , Order = memory_order_acq_rel
                ) noexcept
{
  static_assert( Impl::valid_memory_order<Order>::value
               , "Error: Invalid memory order for atomic_xor_fetch" );
  static_assert( std::is_trivially_copyable<T>::value
               , "Error: atomic_xor_fetch requires a trivially_copyable type" );
  return __atomic_xor_fetch( ptr, val, Order::value );
}

template <typename T, typename Order>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if< Impl::arithmetic_atomic_op<T>::value, T>::type
atomic_or_fetch( T * ptr
               , typename std::remove_cv<T>::type val
               , Order = memory_order_acq_rel
               ) noexcept
{
  static_assert( Impl::valid_memory_order<Order>::value
               , "Error: Invalid memory order for atomic_or_fetch" );
  static_assert( std::is_trivially_copyable<T>::value
               , "Error: atomic_or_fetch requires a trivially_copyable type" );
  return __atomic_or_fetch( ptr, val, Order::value );
}

template <typename T, typename Order>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if< Impl::arithmetic_atomic_op<T>::value, T>::type
atomic_nand_fetch( T * ptr
                 , typename std::remove_cv<T>::type val
                 , Order = memory_order_acq_rel
                 ) noexcept
{
  static_assert( Impl::valid_memory_order<Order>::value
               , "Error: Invalid memory order for atomic_nand_fetch" );
  static_assert( std::is_trivially_copyable<T>::value
               , "Error: atomic_nand_fetch requires a trivially_copyable type" );
  return __atomic_nand_fetch( ptr, val, Order::value );
}

//------------------------------------------------------------------------------
//   GENERIC
//------------------------------------------------------------------------------

template <typename T, typename Order>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if< Impl::non_arithmetic_atomic_op<T>::value, T>::type
atomic_fetch_and( T * ptr
                , typename std::remove_cv<T>::type val
                , Order order = memory_order_acq_rel
                ) noexcept
{
  static_assert( Impl::valid_memory_order<Order>::value
               , "Error: Invalid memory order for atomic_fetch_and" );
  static_assert( std::is_trivially_copyable<T>::value
               , "Error: atomic_fetch_and requires a trivially_copyable type" );
  return Impl::atomic_fetch_op( Impl::AndOper{}, ptr, val, order );
}

template <typename T, typename Order>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if< Impl::non_arithmetic_atomic_op<T>::value, T>::type
atomic_fetch_xor( T * ptr
                , typename std::remove_cv<T>::type val
                , Order order = memory_order_acq_rel
                ) noexcept
{
  static_assert( Impl::valid_memory_order<Order>::value
               , "Error: Invalid memory order for atomic_fetch_xor" );
  static_assert( std::is_trivially_copyable<T>::value
               , "Error: atomic_fetch_xor requires a trivially_copyable type" );
  return Impl::atomic_fetch_op( Impl::XorOper{}, ptr, val, order );
}

template <typename T, typename Order>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if< Impl::non_arithmetic_atomic_op<T>::value, T>::type
atomic_fetch_or( T * ptr
               , typename std::remove_cv<T>::type val
               , Order order = memory_order_acq_rel
               ) noexcept
{
  static_assert( Impl::valid_memory_order<Order>::value
               , "Error: Invalid memory order for atomic_fetch_or" );
  static_assert( std::is_trivially_copyable<T>::value
               , "Error: atomic_fetch_or requires a trivially_copyable type" );
  return Impl::atomic_fetch_op( Impl::OrOper{}, ptr, val, order );
}

template <typename T, typename Order>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if< Impl::non_arithmetic_atomic_op<T>::value, T>::type
atomic_fetch_nand( T * ptr
                 , typename std::remove_cv<T>::type val
                 , Order order = memory_order_acq_rel
                 ) noexcept
{
  static_assert( Impl::valid_memory_order<Order>::value
               , "Error: Invalid memory order for atomic_fetch_nand" );
  static_assert( std::is_trivially_copyable<T>::value
               , "Error: atomic_fetch_nand requires a trivially_copyable type" );
  return Impl::atomic_fetch_op( Impl::NandOper{}, ptr, val, order );
}

template <typename T, typename Order>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if< Impl::non_arithmetic_atomic_op<T>::value, T>::type
atomic_and_fetch( T * ptr
                , typename std::remove_cv<T>::type val
                , Order order = memory_order_acq_rel
                ) noexcept
{
  static_assert( Impl::valid_memory_order<Order>::value
               , "Error: Invalid memory order for atomic_and_fetch" );
  static_assert( std::is_trivially_copyable<T>::value
               , "Error: atomic_and_fetch requires a trivially_copyable type" );
  return Impl::atomic_op_fetch( Impl::AndOper{}, ptr, val, order );
}

template <typename T, typename Order>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if< Impl::non_arithmetic_atomic_op<T>::value, T>::type
atomic_xor_fetch( T * ptr
                , typename std::remove_cv<T>::type val
                , Order order = memory_order_acq_rel
                ) noexcept
{
  static_assert( Impl::valid_memory_order<Order>::value
               , "Error: Invalid memory order for atomic_xor_fetch" );
  static_assert( std::is_trivially_copyable<T>::value
               , "Error: atomic_xor_fetch requires a trivially_copyable type" );
  return Impl::atomic_op_fetch( Impl::XorOper{}, ptr, val, order );
}

template <typename T, typename Order>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if< Impl::non_arithmetic_atomic_op<T>::value, T>::type
atomic_or_fetch( T * ptr
               , typename std::remove_cv<T>::type val
               , Order order = memory_order_acq_rel
               ) noexcept
{
  static_assert( Impl::valid_memory_order<Order>::value
               , "Error: Invalid memory order for atomic_or_fetch" );
  static_assert( std::is_trivially_copyable<T>::value
               , "Error: atomic_or_fetch requires a trivially_copyable type" );
  return Impl::atomic_op_fetch( Impl::OrOper{}, ptr, val, order );
}

template <typename T, typename Order>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if< Impl::non_arithmetic_atomic_op<T>::value, T>::type
atomic_nand_fetch( T * ptr
                 , typename std::remove_cv<T>::type val
                 , Order order = memory_order_acq_rel
                 ) noexcept
{
  static_assert( Impl::valid_memory_order<Order>::value
               , "Error: Invalid memory order for atomic_nand_fetch" );
  static_assert( std::is_trivially_copyable<T>::value
               , "Error: atomic_nand_fetch requires a trivially_copyable type" );
  return Impl::atomic_op_fetch( Impl::NandOper{}, ptr, val, order );
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
// T atomic_fetch_add( T * ptr, T val, memorder = memory_order_acq_rel )
// T atomic_fetch_sub( T * ptr, T val, memorder = memory_order_acq_rel )
//
//   These functions perform the operation suggested by the name, and return the
//   value that had previously been in *ptr.
//
// T atomic_add_fetch( T * ptr, T val, memorder = memory_order_acq_rel )
// T atomic_sub_fetch( T * ptr, T val, memorder = memory_order_acq_rel )
//
//   These functions perform the operation suggested by the name, and return the
//   result of the operation.
//
// Operations on pointer arguments are performed as if the operands were of the
// uintptr_t type. That is, they are not scaled by the size of the type to which
// the pointer points.

template <typename T, typename Order>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if< Impl::arithmetic_atomic_op<T>::value, T>::type
atomic_fetch_add( T * ptr
                , typename std::remove_cv<T>::type val
                , Order = memory_order_acq_rel
                ) noexcept
{
  static_assert( Impl::valid_memory_order<Order>::value
               , "Error: Invalid memory order for atomic_fetch_add" );
  static_assert( std::is_trivially_copyable<T>::value
               , "Error: atomic_fetch_add requires a trivially_copyable type" );
  return __atomic_fetch_add( ptr, val, Order::value );
}

template <typename T, typename Order>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if< Impl::arithmetic_atomic_op<T>::value, T>::type
atomic_fetch_sub( T * ptr
                , typename std::remove_cv<T>::type val
                , Order = memory_order_acq_rel
                ) noexcept
{
  static_assert( Impl::valid_memory_order<Order>::value
               , "Error: Invalid memory order for atomic_fetch_sub" );
  static_assert( std::is_trivially_copyable<T>::value
               , "Error: atomic_fetch_sub requires a trivially_copyable type" );
  return __atomic_fetch_sub( ptr, val, Order::value );
}

template <typename T, typename Order>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if< Impl::arithmetic_atomic_op<T>::value, T>::type
atomic_add_fetch( T * ptr
                , typename std::remove_cv<T>::type val
                , Order = memory_order_acq_rel
                ) noexcept
{
  static_assert( Impl::valid_memory_order<Order>::value
               , "Error: Invalid memory order for atomic_add_fetch" );
  static_assert( std::is_trivially_copyable<T>::value
               , "Error: atomic_add_fetch requires a trivially_copyable type" );
  return __atomic_add_fetch( ptr, val, Order::value );
}

template <typename T, typename Order>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if< Impl::arithmetic_atomic_op<T>::value, T>::type
atomic_sub_fetch( T * ptr
                , typename std::remove_cv<T>::type val
                , Order = memory_order_acq_rel
                ) noexcept
{
  static_assert( Impl::valid_memory_order<Order>::value
               , "Error: Invalid memory order for atomic_sub_fetch" );
  static_assert( std::is_trivially_copyable<T>::value
               , "Error: atomic_sub_fetch requires a trivially_copyable type" );
  return __atomic_sub_fetch( ptr, val, Order::value );
}

//------------------------------------------------------------------------------
//   GENERIC
//------------------------------------------------------------------------------

template <typename T, typename Order>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if< Impl::non_arithmetic_atomic_op<T>::value, T>::type
atomic_fetch_add( T * ptr
                , typename std::remove_cv<T>::type val
                , Order order = memory_order_acq_rel
                ) noexcept
{
  static_assert( Impl::valid_memory_order<Order>::value
               , "Error: Invalid memory order for atomic_fetch_add" );
  static_assert( std::is_trivially_copyable<T>::value
               , "Error: atomic_fetch_add requires a trivially_copyable type" );
  return Impl::atomic_fetch_op( Impl::AddOper{}, ptr, val, order );
}

template <typename T, typename Order>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if< Impl::non_arithmetic_atomic_op<T>::value, T>::type
atomic_fetch_sub( T * ptr
                , typename std::remove_cv<T>::type val
                , Order order = memory_order_acq_rel
                ) noexcept
{
  static_assert( Impl::valid_memory_order<Order>::value
               , "Error: Invalid memory order for atomic_fetch_sub" );
  static_assert( std::is_trivially_copyable<T>::value
               , "Error: atomic_fetch_sub requires a trivially_copyable type" );
  return Impl::atomic_fetch_op( Impl::SubOper{}, ptr, val, order );
}

template <typename T, typename Order>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if< Impl::non_arithmetic_atomic_op<T>::value, T>::type
atomic_add_fetch( T * ptr
                , typename std::remove_cv<T>::type val
                , Order order = memory_order_acq_rel
                ) noexcept
{
  static_assert( Impl::valid_memory_order<Order>::value
               , "Error: Invalid memory order for atomic_add_fetch" );
  static_assert( std::is_trivially_copyable<T>::value
               , "Error: atomic_add_fetch requires a trivially_copyable type" );
  return Impl::atomic_fetch_op( Impl::AddOper{}, ptr, val, order );
}

template <typename T, typename Order>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if< Impl::non_arithmetic_atomic_op<T>::value, T>::type
atomic_sub_fetch( T * ptr
                , typename std::remove_cv<T>::type val
                , Order order = memory_order_acq_rel
                ) noexcept
{
  static_assert( Impl::valid_memory_order<Order>::value
               , "Error: Invalid memory order for atomic_sub_fetch" );
  static_assert( std::is_trivially_copyable<T>::value
               , "Error: atomic_sub_fetch requires a trivially_copyable type" );
  return Impl::atomic_fetch_op( Impl::SubOper{}, ptr, val, order );
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
// void atomic_add( T * ptr, T val, memorder = memory_order_acq_rel )
// void atomic_sub( T * ptr, T val, memorder = memory_order_acq_rel )
//------------------------------------------------------------------------------

template <typename T, typename Order>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if< !std::is_volatile<T>::value, void>::type
atomic_add( T * ptr
          , typename std::remove_cv<T>::type val
          , Order order = memory_order_acq_rel
          ) noexcept
{
  static_assert( Impl::valid_memory_order<Order>::value
               , "Error: Invalid memory order for atomic_add" );
  static_assert( std::is_trivially_copyable<T>::value
               , "Error: atomic_add requires a trivially_copyable type" );
  atomic_add_fetch( ptr, val, order );
}

template <typename T, typename Order>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if< !std::is_volatile<T>::value, void>::type
atomic_sub( T * ptr
          , typename std::remove_cv<T>::type val
          , Order order = memory_order_acq_rel
          ) noexcept
{
  static_assert( Impl::valid_memory_order<Order>::value
               , "Error: Invalid memory order for atomic_sub" );
  static_assert( std::is_trivially_copyable<T>::value
               , "Error: atomic_sub requires a trivially_copyable type" );
  atomic_sub_fetch( ptr, val, order );
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
// void atomic_increment( T* ptr, memorder = memory_order_acq_rel )
// void atomic_decrement( T* ptr, memorder = memory_order_acq_rel )
//
// TODO: reintroduce assembly
//       need to correctly handle memory order
//------------------------------------------------------------------------------
template <typename T, typename Order>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if< !std::is_volatile<T>::value, void>::type
atomic_increment( T * ptr
                , Order order = memory_order_acq_rel
                ) noexcept
{
  static_assert( Impl::valid_memory_order<Order>::value
               , "Error: Invalid memory order for atomic_increment" );
  static_assert( std::is_trivially_copyable<T>::value
               , "Error: atomic_increment requires a trivially_copyable type" );
  constexpr T one = static_cast<T>(1);
  atomic_fetch_add( ptr, one, order );
}

template <typename T, typename Order>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if< !std::is_volatile<T>::value, void>::type
atomic_decrement( T * ptr
                , Order order = memory_order_acq_rel
                ) noexcept
{
  static_assert( Impl::valid_memory_order<Order>::value
               , "Error: Invalid memory order for atomic_decrement" );
  static_assert( std::is_trivially_copyable<T>::value
               , "Error: atomic_decrement requires a trivially_copyable type" );
  constexpr T one = static_cast<T>(1);
  atomic_fetch_sub( ptr, one, order );
}



//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
// T atomic_fetch_mul( T * ptr, T val, memorder = memory_order_acq_rel )
// T atomic_fetch_div( T * ptr, T val, memorder = memory_order_acq_rel )
// T atomic_fetch_mod( T * ptr, T val, memorder = memory_order_acq_rel )
//
//   These functions perform the operation suggested by the name, and return the
//   value that had previously been in *ptr.
//
// T atomic_mul_fetch( T * ptr, T val, memorder = memory_order_acq_rel )
// T atomic_div_fetch( T * ptr, T val, memorder = memory_order_acq_rel )
// T atomic_mod_fetch( T * ptr, T val, memorder = memory_order_acq_rel )
//
//   These functions perform the operation suggested by the name, and return the
//   result of the operation.
//
// Operations on pointer arguments are performed as if the operands were of the
// uintptr_t type. That is, they are not scaled by the size of the type to which
// the pointer points.


template <typename T, typename Order>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if< !std::is_volatile<T>::value, T>::type
atomic_fetch_mul( T * ptr
                , typename std::remove_cv<T>::type val
                , Order order = memory_order_acq_rel
                ) noexcept
{
  static_assert( Impl::valid_memory_order<Order>::value
               , "Error: Invalid memory order for atomic_fetch_mul" );
  static_assert( std::is_trivially_copyable<T>::value
               , "Error: atomic_fetch_mul requires a trivially_copyable type" );
  return Impl::atomic_fetch_op( Impl::MulOper{}, ptr, val, order );
}

template <typename T, typename Order>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if< !std::is_volatile<T>::value, T>::type
atomic_fetch_div( T * ptr
                , typename std::remove_cv<T>::type val
                , Order order = memory_order_acq_rel
                ) noexcept
{
  static_assert( Impl::valid_memory_order<Order>::value
               , "Error: Invalid memory order for atomic_fetch_div" );
  static_assert( std::is_trivially_copyable<T>::value
               , "Error: atomic_fetch_div requires a trivially_copyable type" );
  return Impl::atomic_fetch_op( Impl::DivOper{}, ptr, val, order );
}

template <typename T, typename Order>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if< !std::is_volatile<T>::value, T>::type
atomic_fetch_mod( T * ptr
                , typename std::remove_cv<T>::type val
                , Order order = memory_order_acq_rel
                ) noexcept
{
  static_assert( Impl::valid_memory_order<Order>::value
               , "Error: Invalid memory order for atomic_fetch_mod" );
  static_assert( std::is_trivially_copyable<T>::value
               , "Error: atomic_fetch_mod requires a trivially_copyable type" );
  return Impl::atomic_fetch_op( Impl::ModOper{}, ptr, val, order );
}

template <typename T, typename Order>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if< !std::is_volatile<T>::value, T>::type
atomic_mul_fetch( T * ptr
                , typename std::remove_cv<T>::type val
                , Order order = memory_order_acq_rel
                ) noexcept
{
  static_assert( Impl::valid_memory_order<Order>::value
               , "Error: Invalid memory order for atomic_mul_fetch" );
  static_assert( std::is_trivially_copyable<T>::value
               , "Error: atomic_mul_fetch requires a trivially_copyable type" );
  return Impl::atomic_op_fetch( Impl::MulOper{}, ptr, val, order );
}

template <typename T, typename Order>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if< !std::is_volatile<T>::value, T>::type
atomic_div_fetch( T * ptr
                , typename std::remove_cv<T>::type val
                , Order order = memory_order_acq_rel
                ) noexcept
{
  static_assert( Impl::valid_memory_order<Order>::value
               , "Error: Invalid memory order for atomic_div_fetch" );
  static_assert( std::is_trivially_copyable<T>::value
               , "Error: atomic_div_fetch requires a trivially_copyable type" );
  return Impl::atomic_op_fetch( Impl::DivOper{}, ptr, val, order );
}

template <typename T, typename Order>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if< !std::is_volatile<T>::value, T>::type
atomic_mod_fetch( T * ptr
                , typename std::remove_cv<T>::type val
                , Order order = memory_order_acq_rel
                ) noexcept
{
  static_assert( Impl::valid_memory_order<Order>::value
               , "Error: Invalid memory order for atomic_mod_fetch" );
  static_assert( std::is_trivially_copyable<T>::value
               , "Error: atomic_mod_fetch requires a trivially_copyable type" );
  return Impl::atomic_op_fetch( Impl::ModOper{}, ptr, val, order );
}


//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
// T atomic_fetch_min( T * ptr, T val, memorder = memory_order_acq_rel )
// T atomic_fetch_max( T * ptr, T val, memorder = memory_order_acq_rel )
//
//   These functions perform the operation suggested by the name, and return the
//   value that had previously been in *ptr.
//
// T atomic_max_fetch( T * ptr, T val, memorder = memory_order_acq_rel )
// T atomic_min_fetch( T * ptr, T val, memorder = memory_order_acq_rel )
//
//   These functions perform the operation suggested by the name, and return the
//   result of the operation.
//
// Operations on pointer arguments are performed as if the operands were of the
// uintptr_t type. That is, they are not scaled by the size of the type to which
// the pointer points.

template <typename T, typename Order>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if< !std::is_volatile<T>::value, T>::type
atomic_fetch_min( T * ptr
                , typename std::remove_cv<T>::type val
                , Order order = memory_order_acq_rel
                ) noexcept
{
  static_assert( Impl::valid_memory_order<Order>::value
               , "Error: Invalid memory order for atomic_fetch_min" );
  static_assert( std::is_trivially_copyable<T>::value
               , "Error: atomic_fetch_min requires a trivially_copyable type" );
  return Impl::atomic_fetch_op( Impl::MinOper{}, ptr, val, order );
}

template <typename T, typename Order>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if< !std::is_volatile<T>::value, T>::type
atomic_fetch_max( T * ptr
                , typename std::remove_cv<T>::type val
                , Order order = memory_order_acq_rel
                ) noexcept
{
  static_assert( Impl::valid_memory_order<Order>::value
               , "Error: Invalid memory order for atomic_fetch_max" );
  static_assert( std::is_trivially_copyable<T>::value
               , "Error: atomic_fetch_max requires a trivially_copyable type" );
  return Impl::atomic_fetch_op( Impl::MaxOper{}, ptr, val, order );
}

template <typename T, typename Order>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if< !std::is_volatile<T>::value, T>::type
atomic_min_fetch( T * ptr
                , typename std::remove_cv<T>::type val
                , Order order = memory_order_acq_rel
                ) noexcept
{
  static_assert( Impl::valid_memory_order<Order>::value
               , "Error: Invalid memory order for atomic_min_fetch" );
  static_assert( std::is_trivially_copyable<T>::value
               , "Error: atomic_min_fetch requires a trivially_copyable type" );
  return Impl::atomic_op_fetch( Impl::MinOper{}, ptr, val, order );
}

template <typename T, typename Order>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if< !std::is_volatile<T>::value, T>::type
atomic_max_fetch( T * ptr
                , typename std::remove_cv<T>::type val
                , Order order = memory_order_acq_rel
                ) noexcept
{
  static_assert( Impl::valid_memory_order<Order>::value
               , "Error: Invalid memory order for atomic_max_fetch" );
  static_assert( std::is_trivially_copyable<T>::value
               , "Error: atomic_max_fetch requires a trivially_copyable type" );
  return Impl::atomic_op_fetch( Impl::MaxOper{}, ptr, val, order );
}


//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
// T atomic_fetch_lshift( T * ptr, unsigned val, memorder = memory_order_acq_rel )
// T atomic_fetch_rshift( T * ptr, unsigned val, memorder = memory_order_acq_rel )
//
//   These functions perform the operation suggested by the name, and return the
//   value that had previously been in *ptr.
//
// T atomic_lshift_fetch( T * ptr, unsigned val, memorder = memory_order_acq_rel )
// T atomic_rshift_fetch( T * ptr, unsigned val, memorder = memory_order_acq_rel )
//
//   These functions perform the operation suggested by the name, and return the
//   result of the operation.
//
// Operations on pointer arguments are performed as if the operands were of the
// uintptr_t type. That is, they are not scaled by the size of the type to which
// the pointer points.

template <typename T, typename Order>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if< !std::is_volatile<T>::value, T>::type
atomic_fetch_lshift( T * ptr
                   , unsigned int val
                   , Order order = memory_order_acq_rel
                   ) noexcept
{
  static_assert( Impl::valid_memory_order<Order>::value
               , "Error: Invalid memory order for atomic_fetch_lshift" );
  static_assert( std::is_trivially_copyable<T>::value
               , "Error: atomic_fetch_lshift requires a trivially_copyable type" );
  return Impl::atomic_fetch_op( Impl::LShiftOper{}, ptr, val, order );
}

template <typename T, typename Order>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if< !std::is_volatile<T>::value, T>::type
atomic_fetch_rshift( T * ptr
                   , unsigned int val
                   , Order order = memory_order_acq_rel
                   ) noexcept
{
  static_assert( Impl::valid_memory_order<Order>::value
               , "Error: Invalid memory order for atomic_fetch_rshift" );
  static_assert( std::is_trivially_copyable<T>::value
               , "Error: atomic_fetch_rshift requires a trivially_copyable type" );
  return Impl::atomic_fetch_op( Impl::RShiftOper{}, ptr, val, order );
}

template <typename T, typename Order>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if< !std::is_volatile<T>::value, T>::type
atomic_lshift_fetch( T * ptr
                   , unsigned int val
                   , Order order = memory_order_acq_rel
                   ) noexcept
{
  static_assert( Impl::valid_memory_order<Order>::value
               , "Error: Invalid memory order for atomic_lshift_fetch" );
  static_assert( std::is_trivially_copyable<T>::value
               , "Error: atomic_lshift_fetch requires a trivially_copyable type" );
  return Impl::atomic_op_fetch( Impl::LShiftOper{}, ptr, val, order );
}

template <typename T, typename Order>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if< !std::is_volatile<T>::value, T>::type
atomic_rshift_fetch( T * ptr
                   , unsigned int val
                   , Order order = memory_order_acq_rel
                   ) noexcept
{
  static_assert( Impl::valid_memory_order<Order>::value
               , "Error: Invalid memory order for atomic_rshift_fetch" );
  static_assert( std::is_trivially_copyable<T>::value
               , "Error: atomic_rshift_fetch requires a trivially_copyable type" );
  return Impl::atomic_op_fetch( Impl::RShiftOper{}, ptr, val, order );
}




} // namespace Kokkos

#endif // KOKKOS_IMPL_ATOMIC_GNU_HPP
