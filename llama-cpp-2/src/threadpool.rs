//! Safe wrapper around a ggml CPU threadpool (`ggml_threadpool_new`).
//!
//! Without an attached threadpool the CPU backend builds a throwaway pool on
//! every graph compute when ggml is built without OpenMP, which costs more the
//! more threads it spins up. `llama.cpp`'s `common` attaches one persistent pool
//! per context; [`crate::context::LlamaContext::attach_threadpool`] does the same.

use crate::ptr::Ptr;

/// Scheduling priority of the pool's worker threads (`ggml_sched_priority`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ThreadpoolPriority {
    /// `GGML_SCHED_PRIO_LOW`
    Low,
    /// `GGML_SCHED_PRIO_NORMAL`
    #[default]
    Normal,
    /// `GGML_SCHED_PRIO_MEDIUM`
    Medium,
    /// `GGML_SCHED_PRIO_HIGH`
    High,
    /// `GGML_SCHED_PRIO_REALTIME`
    Realtime,
}

impl From<ThreadpoolPriority> for llama_cpp_sys_2::ggml_sched_priority {
    fn from(value: ThreadpoolPriority) -> Self {
        match value {
            ThreadpoolPriority::Low => llama_cpp_sys_2::GGML_SCHED_PRIO_LOW,
            ThreadpoolPriority::Normal => llama_cpp_sys_2::GGML_SCHED_PRIO_NORMAL,
            ThreadpoolPriority::Medium => llama_cpp_sys_2::GGML_SCHED_PRIO_MEDIUM,
            ThreadpoolPriority::High => llama_cpp_sys_2::GGML_SCHED_PRIO_HIGH,
            ThreadpoolPriority::Realtime => llama_cpp_sys_2::GGML_SCHED_PRIO_REALTIME,
        }
    }
}

/// Errors creating a [`LlamaThreadpool`].
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum ThreadpoolCreateError {
    /// The thread count was below one.
    #[error("a threadpool needs at least one thread, got {0}")]
    InvalidThreadCount(i32),
    /// ggml returned null. Checked, though ggml v0.8.0 aborts the process
    /// instead when it cannot allocate the pool or start a thread.
    #[error("null reference from ggml")]
    NullReturn,
}

/// A persistent ggml CPU threadpool. Freed on drop.
#[derive(Debug)]
pub struct LlamaThreadpool {
    pub(crate) threadpool: Ptr<llama_cpp_sys_2::ggml_threadpool>,
}

// SAFETY: the pool is not bound to the thread that created it; ggml drives it
// from whichever thread computes the graph.
unsafe impl Send for LlamaThreadpool {}

impl LlamaThreadpool {
    /// Start a pool of `n_threads` workers with ggml's default parameters
    /// (no CPU mask, polling level 50) and the given priority.
    ///
    /// # Errors
    ///
    /// [`ThreadpoolCreateError::InvalidThreadCount`] when `n_threads < 1`,
    /// [`ThreadpoolCreateError::NullReturn`] if ggml returns null (v0.8.0
    /// aborts instead when it cannot allocate the pool or start a thread).
    pub fn new(
        n_threads: i32,
        priority: Option<ThreadpoolPriority>,
    ) -> Result<Self, ThreadpoolCreateError> {
        if n_threads < 1 {
            return Err(ThreadpoolCreateError::InvalidThreadCount(n_threads));
        }
        let mut params = unsafe { llama_cpp_sys_2::ggml_threadpool_params_default(n_threads) };
        params.prio = priority.unwrap_or_default().into();
        let threadpool = unsafe { llama_cpp_sys_2::ggml_threadpool_new(&raw mut params) };
        let threadpool = Ptr::new(threadpool).ok_or(ThreadpoolCreateError::NullReturn)?;
        Ok(Self { threadpool })
    }
}

impl Drop for LlamaThreadpool {
    fn drop(&mut self) {
        unsafe { llama_cpp_sys_2::ggml_threadpool_free(self.threadpool.as_mut_ptr()) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_no_threads() {
        assert_eq!(
            LlamaThreadpool::new(0, None).unwrap_err(),
            ThreadpoolCreateError::InvalidThreadCount(0)
        );
    }

    #[test]
    fn starts_and_frees_a_pool() {
        let pool = LlamaThreadpool::new(2, Some(ThreadpoolPriority::Normal)).unwrap();
        drop(pool);
    }
}
