use crate::prover::callbacks::Callbacks;
use crate::prover::context::ProverContext;
use era_cudart::event::{CudaEvent, CudaEventCreateFlags};
use era_cudart::memory::memory_copy_async;
use era_cudart::result::CudaResult;
use era_cudart::slice::{CudaSlice, CudaSliceMut};
use era_cudart::stream::CudaStreamWaitEventFlags;
use std::ops::Deref;
use std::sync::{Arc, Mutex};

pub struct Transfer<'a, C: ProverContext> {
    pub(crate) allocated: CudaEvent,
    pub(crate) transferred: CudaEvent,
    pub(crate) callbacks: Callbacks<'a>,
    _phantom: std::marker::PhantomData<C>,
}

impl<'a, C: ProverContext> Transfer<'a, C> {
    pub(crate) fn new() -> CudaResult<Self> {
        Ok(Self {
            allocated: CudaEvent::create_with_flags(CudaEventCreateFlags::DISABLE_TIMING)?,
            transferred: CudaEvent::create_with_flags(CudaEventCreateFlags::DISABLE_TIMING)?,
            callbacks: Callbacks::new(),
            _phantom: std::marker::PhantomData,
        })
    }

    pub(crate) fn record_allocated(&self, context: &C) -> CudaResult<()> {
        self.allocated.record(context.get_exec_stream())
    }

    pub(crate) fn ensure_allocated(&self, context: &C) -> CudaResult<()> {
        context
            .get_h2d_stream()
            .wait_event(&self.allocated, CudaStreamWaitEventFlags::DEFAULT)
    }

    pub fn schedule<T>(
        &mut self,
        src: Arc<impl CudaSlice<T> + Send + Sync + ?Sized + 'a>,
        dst: &mut (impl CudaSliceMut<T> + ?Sized),
        context: &C,
    ) -> CudaResult<()> {
        assert_eq!(src.len(), dst.len());
        self.ensure_allocated(context)?;
        let stream = context.get_h2d_stream();
        memory_copy_async(dst, src.deref(), stream)?;
        let src = Mutex::new(Some(src));
        let f = move || {
            src.lock().unwrap().take();
        };
        self.callbacks.schedule(f, stream)
    }

    pub(crate) fn record_transferred(&self, context: &C) -> CudaResult<()> {
        self.transferred.record(context.get_h2d_stream())
    }

    pub fn ensure_transferred(&self, context: &C) -> CudaResult<()> {
        context
            .get_exec_stream()
            .wait_event(&self.transferred, CudaStreamWaitEventFlags::DEFAULT)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prover::context::{MemPoolProverContext, ProverContextConfig};

    #[test]
    fn test_transfer() -> CudaResult<()> {
        let context = MemPoolProverContext::new(&ProverContextConfig::default())?;
        let src = Arc::new(vec![0; 1024]);
        let mut transfer = Transfer::new()?;
        let mut dst = context.alloc(1024)?;
        transfer.record_allocated(&context)?;
        transfer.schedule(src, &mut dst, &context)?;
        transfer.record_transferred(&context)?;
        Ok(())
    }
}
