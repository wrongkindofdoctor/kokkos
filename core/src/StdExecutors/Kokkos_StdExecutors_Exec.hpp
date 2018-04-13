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

#ifndef STDEXECUTORS_KOKKOS_STDEXECUTORS_EXEC_HPP
#define STDEXECUTORS_KOKKOS_STDEXECUTORS_EXEC_HPP

#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_StdExecutors.hpp>
#include <Kokkos_hwloc.hpp>

#include <vector>
#include <utility>
#include <iostream>
#include <memory>
#include <experimental/execution>
#include <experimental/thread_pool>

namespace Kokkos {
namespace Experimental {
namespace Impl {

template <typename Executor>
struct StdExecutorsImpl {

  static_assert(
    std::experimental::execution::can_query_v<
      Executor,
      std::experimental::execution::context_t
    >,
    "Only executors with a query-able context are currently supported"
  );

  template <typename ProtoExecutor>
  static auto _init_executor_reqs(ProtoExecutor exec) {
    auto req_exec = std::experimental::execution::require(
      exec,
      std::experimental::execution::bulk,
      std::experimental::execution::twoway,
      std::experimental::execution::not_continuation
    );
    return std::experimental::execution::prefer(
      req_exec,
      std::experimental::execution::bulk_parallel_execution
    );
  }

  using executor_type = Executor;
  using executor_actual_type = decltype(_init_executor_reqs(std::declval<Executor>()));
  using future_type = std::experimental::execution::executor_future_t<executor_type, void>;

  using executor_context_type = std::decay_t<decltype(std::experimental::execution::query(
    std::declval<Executor>(),
    std::experimental::execution::context
  ))>;

  static std::shared_ptr<executor_context_type> s_default_context;

  std::shared_ptr<executor_context_type> m_context = nullptr;
  executor_type m_executor;

  std::vector<std::unique_ptr<future_type>> fence_futures;


  StdExecutorsImpl()
    : m_context(s_default_context),
      m_executor(_init_executor_reqs(m_context->executor()))
  { }

  explicit
  StdExecutorsImpl(executor_context_type&& ctxt)
    : m_context(std::make_shared<executor_context_type>(std::move(ctxt))),
      m_executor(_init_executor_reqs(m_context->executor()))
  { }

  explicit
  StdExecutorsImpl(Executor exec)
    : m_context(nullptr),
      m_executor(std::move(exec))
  { }

  bool is_asynchronous() const {
    return not std::experimental::execution::query(
      m_executor,
      std::experimental::execution::always_blocking
    );
  }

  void fence() {
    for(auto&& f : fence_futures) { f->get(); }
    fence_futures.clear();
  }

  bool in_parallel() const {
    return not fence_futures.empty();
  }

  void print_configuration(std::ostream& o, bool verbose) const {
    o << "(executor print configuration not yet implemented)\n";
  }

  int thread_pool_size() const {
    // Can't be done with only P0443; return 1 for now
    return 1;
  }

  int thread_pool_rank() const {
    // Can't be done with only P0443; just return 0 for now
    return 0;
  }

};

} // end namespace Impl
} // end namespace Experimental
} // end namespace Kokkos


namespace Kokkos {
namespace Experimental {

template <typename Executor>
StdExecutors<Executor>::StdExecutors() noexcept
  : m_impl(s_default_instance->m_impl)
{ }

template <typename Executor>
StdExecutors<Executor>::StdExecutors(
  std::shared_ptr<Impl::StdExecutorsImpl<Executor>> const& impl
) noexcept
  : m_impl(impl)
{ }


template <typename Executor>
void
StdExecutors<Executor>::fence(
  Kokkos::Experimental::StdExecutors<Executor> const& es
) {
  es.m_impl->fence();
}

template <typename Executor>
void
StdExecutors<Executor>::fence() {
  s_default_instance->m_impl->fence();
}

template <typename Executor>
bool
StdExecutors<Executor>::is_asynchronous(
  Kokkos::Experimental::StdExecutors<Executor> const& es
) noexcept {
  es.m_impl->is_asynchronous();
}

template <typename Executor>
bool
StdExecutors<Executor>::is_asynchronous() noexcept {
  s_default_instance->m_impl->is_asynchronous();
}

template <typename Executor>
bool
StdExecutors<Executor>::in_parallel(
  Kokkos::Experimental::StdExecutors<Executor> const& es
) noexcept {
  return es.m_impl->in_parallel();
}

template <typename Executor>
bool
StdExecutors<Executor>::in_parallel() noexcept {
  return s_default_instance->m_impl->in_parallel();
}

template <typename Executor>
int
StdExecutors<Executor>::thread_pool_size() noexcept {
  return s_default_instance->m_impl->thread_pool_size();
}

template <typename Executor>
int
StdExecutors<Executor>::thread_pool_rank() noexcept {
  return s_default_instance->m_impl->thread_pool_rank();
}

template <typename Executor>
void
StdExecutors<Executor>::print_configuration(
  std::ostream& o,
  bool const verbose
) {
  s_default_instance->m_impl->print_configuration(o, verbose);
}

template <typename Executor>
bool
StdExecutors<Executor>::is_initialized() noexcept {
  return bool(StdExecutors<Executor>::s_default_instance);
}

template <typename Executor>
void
StdExecutors<Executor>::initialize(int thread_count) {
  assert(not StdExecutors<Executor>::s_default_instance);

  if(thread_count <= 0) {
    if(Kokkos::hwloc::available()) {
      thread_count = Kokkos::hwloc::get_available_numa_count()
        * Kokkos::hwloc::get_available_cores_per_numa()
        * Kokkos::hwloc::get_available_threads_per_core();
    }
    else {
      thread_count = 8;
    }
  }

  // Allocate space:
  size_t pool_reduce_bytes = 32 * static_cast<size_t>(thread_count);
  size_t team_reduce_bytes = 32 * static_cast<size_t>(thread_count);
  size_t team_shared_bytes = 1024 * static_cast<size_t>(thread_count);
  size_t thread_local_bytes = 1024;
  // TODO allocate space for each thread's scratch

  using executor_actual_type = typename Impl::StdExecutorsImpl<Executor>::executor_actual_type;
  if(not bool(Impl::StdExecutorsImpl<Executor>::s_default_context)) {
    if(not bool(Impl::StdExecutorsImpl<executor_actual_type>::s_default_context)) {
      Impl::StdExecutorsImpl<executor_actual_type>::s_default_context = std::make_shared<
        typename Impl::StdExecutorsImpl<executor_actual_type>::executor_context_type
      >(
        thread_count
      );
    }
    Impl::StdExecutorsImpl<Executor>::s_default_context = Impl::StdExecutorsImpl<executor_actual_type>::s_default_context;
  }
  // Have to use `new` because it's a private constructor that make_shared can't access
  StdExecutors<Executor>::s_default_instance.reset(
    new StdExecutors<Executor>(
      std::make_shared<Impl::StdExecutorsImpl<Executor>>()
    )
  );
}

template <typename Executor>
void
StdExecutors<Executor>::finalize() {
  StdExecutors<Executor>::s_default_instance = nullptr;
}

template <typename Executor>
std::unique_ptr<StdExecutors<Executor>> StdExecutors<Executor>::s_default_instance = nullptr;

template <typename Executor>
std::shared_ptr<typename Impl::StdExecutorsImpl<Executor>::executor_context_type>
  Impl::StdExecutorsImpl<Executor>::s_default_context = nullptr;

} // end namespace Experimental
} // end namespace Kokkos

#endif // STDEXECUTORS_KOKKOS_STDEXECUTORS_EXEC_HPP
