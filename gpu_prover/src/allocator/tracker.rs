use std::alloc::AllocError;
use std::collections::{BTreeMap, BTreeSet, Bound};
use std::ptr::NonNull;

pub struct StaticAllocationsTracker {
    ptr: NonNull<u8>,
    len: usize,
    free_len_by_ptr: BTreeMap<NonNull<u8>, usize>,
    free_ptrs_by_len: BTreeMap<usize, BTreeSet<NonNull<u8>>>,
}

impl StaticAllocationsTracker {
    pub fn new(ptr: NonNull<u8>, len: usize) -> Self {
        let mut free_len_by_ptr = BTreeMap::new();
        free_len_by_ptr.insert(ptr, len);
        let mut free_ptrs_by_len = BTreeMap::new();
        let mut ptrs = BTreeSet::new();
        ptrs.insert(ptr);
        free_ptrs_by_len.insert(len, ptrs);
        Self {
            ptr,
            len,
            free_len_by_ptr,
            free_ptrs_by_len,
        }
    }

    pub fn alloc(&mut self, len: usize) -> Result<NonNull<u8>, AllocError> {
        // dbg!("alloc", len);
        let mut cursor = self.free_ptrs_by_len.lower_bound_mut(Bound::Included(&len));
        if let Some((&free_len, free_ptrs)) = cursor.peek_next() {
            let ptr = free_ptrs.pop_first().unwrap();
            if free_ptrs.is_empty() {
                cursor.remove_next();
            }
            self.free_len_by_ptr.remove(&ptr);
            if free_len > len {
                let new_free_len = free_len - len;
                let new_free_ptr = unsafe { ptr.add(len) };
                self.free_len_by_ptr.insert(new_free_ptr, new_free_len);
                self.free_ptrs_by_len
                    .entry(new_free_len)
                    .or_default()
                    .insert(new_free_ptr);
            }
            // dbg!(ptr);
            // dbg!(&self.free_len_by_ptr);
            // dbg!(&self.free_ptrs_by_len);
            Ok(ptr)
        } else {
            Err(AllocError)
        }
    }

    pub fn free(&mut self, mut ptr: NonNull<u8>, mut len: usize) {
        // dbg!("free", ptr, len);
        unsafe {
            let offset = ptr.offset_from(self.ptr);
            if offset < 0 || (offset as usize + len) > self.len {
                panic!("out of bounds free");
            }
            let mut cursor = self.free_len_by_ptr.lower_bound_mut(Bound::Included(&ptr));
            if let Some((&next_ptr, &mut next_len)) = cursor.peek_next() {
                let offset = next_ptr.offset_from(ptr) as usize;
                if offset < len {
                    panic!("double free");
                }
                if offset == len {
                    cursor.remove_next();
                    let ptrs = self.free_ptrs_by_len.get_mut(&next_len).unwrap();
                    ptrs.remove(&next_ptr);
                    if ptrs.is_empty() {
                        self.free_ptrs_by_len.remove(&next_len);
                    }
                    len += next_len;
                }
            }
            if let Some((&prev_ptr, &mut prev_len)) = cursor.peek_prev() {
                let offset = ptr.offset_from(prev_ptr) as usize;
                if offset < prev_len {
                    panic!("double free");
                }
                if offset == prev_len {
                    cursor.remove_prev();
                    let ptrs = self.free_ptrs_by_len.get_mut(&prev_len).unwrap();
                    ptrs.remove(&prev_ptr);
                    if ptrs.is_empty() {
                        self.free_ptrs_by_len.remove(&prev_len);
                    }
                    ptr = prev_ptr;
                    len += prev_len;
                }
            }
        }
        self.free_len_by_ptr.insert(ptr, len);
        self.free_ptrs_by_len.entry(len).or_default().insert(ptr);
        // dbg!(&self.free_len_by_ptr);
        // dbg!(&self.free_ptrs_by_len);
    }
}

unsafe impl Send for StaticAllocationsTracker {}
