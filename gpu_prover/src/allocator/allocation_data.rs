use std::marker::PhantomData;
use std::ptr::NonNull;

#[derive(Debug)]
pub(crate) struct StaticAllocationData<T> {
    pub ptr: NonNull<T>,
    pub len: usize,
    _owns_t: PhantomData<T>,
}

impl<T> StaticAllocationData<T> {
    pub fn new(ptr: NonNull<T>, len: usize) -> Self {
        Self {
            ptr,
            len,
            _owns_t: PhantomData,
        }
    }
}

unsafe impl<T> Send for StaticAllocationData<T> where Vec<T>: Send {}

unsafe impl<T> Sync for StaticAllocationData<T> where Vec<T>: Sync {}
