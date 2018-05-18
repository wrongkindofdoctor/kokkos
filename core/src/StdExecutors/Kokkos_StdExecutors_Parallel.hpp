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

#ifndef KOKKOS_KOKKOS_STDEXECUTORS_PARALLEL_HPP
#define KOKKOS_KOKKOS_STDEXECUTORS_PARALLEL_HPP

#include <Kokkos_Macros.hpp>

#if defined( KOKKOS_ENABLE_STDEXECUTORS )

#include <Kokkos_ExecPolicy.hpp>
#include <Kokkos_StdExecutors.hpp>
#include <KokkosExp_MDRangePolicy.hpp>
#include <experimental/execution>


namespace Kokkos {
namespace Impl {

template <typename FunctorType, typename Executor, typename... Traits>
class ParallelFor<
  FunctorType,
  Kokkos::RangePolicy<Traits...>,
  Kokkos::Experimental::StdExecutors<Executor>
>
{
private:
  using execution_space_t = Kokkos::Experimental::StdExecutors<Executor>;
  using policy_t = Kokkos::RangePolicy<Traits...>;
  using work_range_t = typename policy_t::WorkRange;
  using member_t = typename policy_t::member_type;
  using work_tag_t = typename policy_t::work_tag;

  execution_space_t* m_instance;
  FunctorType const m_functor;
  policy_t const m_policy;

  template <typename Index>
  inline static void _invoke(FunctorType const& functor, Index idx, std::true_type) {
    functor(idx);
  }

  template <typename Index>
  inline static void _invoke(FunctorType const& functor, Index idx, std::false_type) {
    functor(work_tag_t{}, idx);
  }


public:

  inline
  ParallelFor(
    FunctorType const& arg_functor,
    policy_t const& arg_policy
  ) : m_instance(execution_space_t::s_default_instance.get()),
      m_functor(arg_functor),
      m_policy(arg_policy)
  { }

  inline void execute() const
  {
    // TODO Don't ignore schedule type?
    // TODO obey chunking
    auto& impl = m_instance->m_impl;
    using future_type = typename Kokkos::Experimental::Impl::StdExecutorsImpl<Executor>::future_type;
    auto n = m_policy.end() - m_policy.begin();

    impl->fence_futures.emplace_back(std::make_unique<future_type>(
      impl->m_executor.bulk_twoway_execute([functor=m_functor,offset=m_policy.begin()](auto n, auto) {
        ParallelFor::_invoke(functor, offset+n, std::is_same<work_tag_t, void>{});
      }, n, [](){}, [](){ return 0; })
    ));
  }
};

template <typename FunctorType, typename Executor, typename... Traits>
class ParallelFor<
  FunctorType,
  Kokkos::MDRangePolicy<Traits...>,
  Kokkos::Experimental::StdExecutors<Executor>
>
{
private:
  using execution_space_t = Kokkos::Experimental::StdExecutors<Executor>;
  using policy_t = Kokkos::MDRangePolicy<Traits...>;
  using work_range_t = typename policy_t::WorkRange;
  using member_t = typename policy_t::member_type;
  using work_tag_t = typename policy_t::work_tag;

  execution_space_t* m_instance;
  FunctorType const m_functor;
  policy_t const m_policy;

  using iterate_type = typename Kokkos::Impl::HostIterateTile< policy_t, FunctorType, typename policy_t::work_tag, void >;

  template <typename Functor, typename Index>
  inline static void _invoke(Functor&& functor, Index idx, std::true_type) {
    std::forward<Functor>(functor)(idx);
  }

  template <typename Functor, typename Index>
  inline static void _invoke(Functor&& functor, Index idx, std::false_type) {
    std::forward<Functor>(functor)(work_tag_t{}, idx);
  }


public:

  inline
  ParallelFor(
    FunctorType const& arg_functor,
    policy_t const& arg_policy
  ) : m_instance(execution_space_t::s_default_instance.get()),
      m_functor(arg_functor),
      m_policy(arg_policy)
  { }

  inline void execute() const
  {
    // TODO Don't ignore schedule type?
    // TODO obey chunking
    auto& impl = m_instance->m_impl;
    using future_type = typename Kokkos::Experimental::Impl::StdExecutorsImpl<Executor>::future_type;
    auto n = m_policy.end() - m_policy.begin();

    impl->fence_futures.emplace_back(std::make_unique<future_type>(
      impl->m_executor.bulk_twoway_execute([functor=m_functor,policy=m_policy](auto n, auto) {
        ParallelFor::_invoke(iterate_type(functor, policy), n + policy.begin(), std::is_same<work_tag_t, void>{});
      }, n, [](){}, [](){ return 0; })
    ));
  }
};

// TODO finish this once execution space scratch allocation stuff is set up
template <typename FunctorType, typename Executor, typename ReducerType, typename... Traits>
class ParallelReduce<
  FunctorType,
  Kokkos::RangePolicy<Traits...>,
  ReducerType,
  Kokkos::Experimental::StdExecutors<Executor>
>
{
  private:
    using execution_space_t = Kokkos::Experimental::StdExecutors<Executor>;
    using policy_t = Kokkos::RangePolicy<Traits...>;
    using work_range_t = typename policy_t::WorkRange;
    using member_t = typename policy_t::member_type;
    using work_tag_t = typename policy_t::work_tag;
 
  typedef Kokkos::Impl::if_c< std::is_same<InvalidType,ReducerType>::value, FunctorType, ReducerType> ReducerConditional; 
//  using ReducerConditional = std::conditional_t< std::is_same_v<InvalidType,ReducerType>, FunctorType, ReducerType>;
  using ReducerTypeFwd = typename ReducerConditional::type;
  using WorkTagFwd = typename std::conditional_t< std::is_same_v<InvalidType,ReducerType>, work_tag_t, void>;
  using Analysis = FunctorAnalysis<FunctorPatternInterface::REDUCE , policy_t, FunctorType>;
  using ValueInit = Kokkos::Impl::FunctorValueInit<ReducerTypeFwd, WorkTagFwd>;
  using ValueJoin = Kokkos::Impl::FunctorValueJoin<ReducerTypeFwd, WorkTagFwd>;
  using value_type = typename Analysis::value_type;
  using pointer_type = typename Analysis::pointer_type;
  using reference_type = typename Analysis::reference_type;

  execution_space_t* m_instance;
  FunctorType const m_functor;
  policy_t const m_policy;
  ReducerType const m_reducer;
  pointer_type const m_result_ptr;


  template <typename Index>
  inline static void _invoke(FunctorType const& functor, Index idx, value_type& val, std::true_type) {
    functor(idx,val);
  }

  template <typename Index>
  inline static void _invoke(FunctorType const& functor, Index idx, value_type& val, std::false_type) {
    functor(work_tag_t{}, idx,val);
  }


public:

  template <typename ViewType>
  inline
  ParallelReduce(
    FunctorType const& arg_functor,
    policy_t const& arg_policy,
    ViewType const&  arg_view,
    typename std::enable_if<
      Kokkos::is_view< ViewType >::value &&
      !Kokkos::is_reducer_type<ReducerType>::value,
      void*
    >::type = NULL
  ) : m_instance(execution_space_t::s_default_instance.get()),
      m_functor(arg_functor),
      m_policy(arg_policy),
      m_reducer( InvalidType()),
      m_result_ptr(arg_view.data())
  { }

  inline
  ParallelReduce(
    FunctorType const& arg_functor,
    policy_t const& arg_policy,
    ReducerType const& reducer
  ) : m_instance(execution_space_t::s_default_instance.get()),
      m_functor(arg_functor),
      m_policy(arg_policy),
      m_reducer(reducer),
      m_result_ptr(reducer.view().data())
  { }

  inline void execute() const
  {
    // TODO Don't ignore schedule type?
    // TODO obey chunking?
    // TODO support something else than += reduction
    auto& impl = m_instance->m_impl;
    using future_type = std::experimental::execution::executor_future_t<Executor, std::atomic<value_type>>;//typename Kokkos::Experimental::Impl::StdExecutorsImpl<Executor>::future_type;
    auto n = m_policy.end() - m_policy.begin();

      std::atomic<value_type> result  = 0;

      auto my_fut_result = 
      impl->m_executor.bulk_twoway_execute([functor=m_functor,offset=m_policy.begin()](auto i, std::atomic<value_type>* result, auto) {
        value_type val = 0;
        ParallelReduce::_invoke(functor, offset+i, val, std::is_same<work_tag_t, void>{});
        *result += val;
      }, n, [&]()-> std::atomic<value_type>* {return &result;}, []() {return 0;} )
    ;

    auto result2 = my_fut_result.get();
    m_result_ptr[0] = result;
  }
};

} // end namespace Impl
} // end namespace Kokkos

#endif //defined( KOKKOS_ENABLE_STDEXECUTORS )

#endif //KOKKOS_KOKKOS_STDEXECUTORS_PARALLEL_HPP
