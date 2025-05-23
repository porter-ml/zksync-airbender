use super::*;

#[inline(always)]
pub fn read_value<T: Sized + Copy>(place: ColumnAddress, witness_row: &[T], memory_row: &[T]) -> T {
    unsafe {
        match place {
            ColumnAddress::WitnessSubtree(offset) => {
                debug_assert!(
                    offset < witness_row.len(),
                    "witness row contains {} elements, but index is {}",
                    witness_row.len(),
                    offset
                );
                witness_row.as_ptr().add(offset).read()
            }
            ColumnAddress::MemorySubtree(offset) => {
                debug_assert!(
                    offset < memory_row.len(),
                    "memory row contains {} elements, but index is {}",
                    memory_row.len(),
                    offset
                );
                memory_row.as_ptr().add(offset).read()
            }
            _ => unreachable!("can only read from witness or memory tree here"),
        }
    }
}

#[inline(always)]
pub fn read_value_ext<T: Sized + Copy>(
    place: ColumnAddress,
    witness_row: &[T],
    memory_row: &[T],
    scratch_space: &[T],
) -> T {
    unsafe {
        match place {
            ColumnAddress::WitnessSubtree(offset) => {
                debug_assert!(
                    offset < witness_row.len(),
                    "witness row contains {} elements, but index is {}",
                    witness_row.len(),
                    offset
                );
                witness_row.as_ptr().add(offset).read()
            }
            ColumnAddress::MemorySubtree(offset) => {
                debug_assert!(
                    offset < memory_row.len(),
                    "memory row contains {} elements, but index is {}",
                    memory_row.len(),
                    offset
                );
                memory_row.as_ptr().add(offset).read()
            }
            ColumnAddress::OptimizedOut(offset) => {
                debug_assert!(
                    offset < scratch_space.len(),
                    "optimized variables scratch space contains {} elements, but index is {}",
                    scratch_space.len(),
                    offset
                );
                scratch_space.as_ptr().add(offset).read()
            }
            _ => unreachable!("can only read from witness or memory tree here, or scratch space"),
        }
    }
}

#[inline(always)]
pub fn read_value_with_setup_access<T: Sized + Copy>(
    place: ColumnAddress,
    witness_row: &[T],
    memory_row: &[T],
    setup_row: &[T],
) -> T {
    unsafe {
        match place {
            ColumnAddress::WitnessSubtree(offset) => {
                debug_assert!(
                    offset < witness_row.len(),
                    "witness row contains {} elements, but index is {}",
                    witness_row.len(),
                    offset
                );
                witness_row.as_ptr().add(offset).read()
            }
            ColumnAddress::MemorySubtree(offset) => {
                debug_assert!(
                    offset < memory_row.len(),
                    "memory row contains {} elements, but index is {}",
                    memory_row.len(),
                    offset
                );
                memory_row.as_ptr().add(offset).read()
            }
            ColumnAddress::SetupSubtree(offset) => {
                debug_assert!(
                    offset < setup_row.len(),
                    "setup row contains {} elements, but index is {}",
                    setup_row.len(),
                    offset
                );
                setup_row.as_ptr().add(offset).read()
            }
            _ => unreachable!("can only read from witness, memory or setup tree here"),
        }
    }
}

#[inline(always)]
pub fn write_value<T: Sized + Copy>(
    place: ColumnAddress,
    value: T,
    witness_row: &mut [T],
    memory_row: &mut [T],
) {
    unsafe {
        match place {
            ColumnAddress::WitnessSubtree(offset) => {
                debug_assert!(
                    offset < witness_row.len(),
                    "witness row contains {} elements, but index is {}",
                    witness_row.len(),
                    offset
                );
                *witness_row.get_unchecked_mut(offset) = value;
            }
            ColumnAddress::MemorySubtree(offset) => {
                debug_assert!(
                    offset < memory_row.len(),
                    "memory row contains {} elements, but index is {}",
                    memory_row.len(),
                    offset
                );
                *memory_row.get_unchecked_mut(offset) = value;
            }
            _ => unreachable!("can only write into witness or memory tree here"),
        }
    }
}

#[inline(always)]
pub fn write_value_ext<T: Sized + Copy>(
    place: ColumnAddress,
    value: T,
    witness_row: &mut [T],
    memory_row: &mut [T],
    scratch_space: &mut [T],
) {
    unsafe {
        match place {
            ColumnAddress::WitnessSubtree(offset) => {
                debug_assert!(
                    offset < witness_row.len(),
                    "witness row contains {} elements, but index is {}",
                    witness_row.len(),
                    offset
                );
                *witness_row.get_unchecked_mut(offset) = value;
            }
            ColumnAddress::MemorySubtree(offset) => {
                debug_assert!(
                    offset < memory_row.len(),
                    "memory row contains {} elements, but index is {}",
                    memory_row.len(),
                    offset
                );
                *memory_row.get_unchecked_mut(offset) = value;
            }
            ColumnAddress::OptimizedOut(offset) => {
                debug_assert!(
                    offset < scratch_space.len(),
                    "optimized out scratch space contains {} elements, but index is {}",
                    scratch_space.len(),
                    offset
                );
                *scratch_space.get_unchecked_mut(offset) = value;
            }
            _ => unreachable!("can only write into witness or memory tree here, or scratch space"),
        }
    }
}
