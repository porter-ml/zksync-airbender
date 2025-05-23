use non_determinism_source::NonDeterminismSource;
use std::cell::RefCell;

thread_local! {
    static SOURCE_ITERATOR: RefCell<Option<Box<dyn Iterator<Item = u32> + Send + Sync + 'static>>> = RefCell::new(None);
}

#[derive(Clone, Copy, Debug)]
pub struct ThreadLocalBasedSource;

impl NonDeterminismSource for ThreadLocalBasedSource {
    #[inline(always)]
    fn read_word() -> u32 {
        read_word()
    }
    #[inline(always)]
    fn read_reduced_field_element(modulus: u32) -> u32 {
        read_field_element(modulus)
    }
}

pub fn set_iterator(iterator: impl Iterator<Item = u32> + Send + Sync + 'static) {
    SOURCE_ITERATOR.with_borrow_mut(|el| {
        if let Some(it) = el.as_mut() {
            assert!(it.next().is_none());
        }

        *el = Some(Box::new(iterator))
    });
}

pub fn try_read_word() -> Option<u32> {
    SOURCE_ITERATOR.with_borrow_mut(|el| {
        if let Some(it) = el.as_mut() {
            return it.next();
        } else {
            panic!("iterator must be set")
        }
    })
}

fn read_word() -> u32 {
    try_read_word().expect("next word from thread local source")
}

#[inline(always)]
fn read_field_element(modulus: u32) -> u32 {
    read_word() % modulus
}
