use field::Mersenne31ComplexVectorized;
use field::Mersenne31ComplexVectorizedInterleaved;
use field::WIDTH;
use field::{Mersenne31Complex, Mersenne31Field};
use std::ops::Deref;
use std::ops::DerefMut;
use trace_holder::RowMajorTraceFixedColumnsView;

// Utility to help reorder data as a part of computing RadixD FFTs. Conceputally, it works like a transpose, but with the column indexes bit-reversed.
// Use a lookup table to avoid repeating the slow bit reverse operations.
// Unrolling the outer loop by a factor D helps speed things up.
// const parameter D (for Divisor) determines the divisor to use for the "bit reverse", and how much to unroll. `input.len() / height` must be a power of D.
pub fn bitreversed_transpose<T: Copy, const D: usize>(
    height: usize,
    input: &[T],
    output: &mut [T],
) {
    let width = input.len() / height;

    // Let's make sure the arguments are ok
    assert!(D > 1 && input.len() % height == 0 && input.len() == output.len());

    let strided_width = width / D;
    let rev_digits = {
        assert!(D.is_power_of_two());
        let width_bits = width.trailing_zeros();
        let d_bits = D.trailing_zeros();

        // verify that width is a power of d
        assert!(width_bits % d_bits == 0);
        width_bits / d_bits
    };

    for x in 0..strided_width {
        let mut i = 0;
        let x_fwd = [(); D].map(|_| {
            let value = D * x + i;
            i += 1;
            value
        }); // If we had access to rustc 1.63, we could use std::array::from_fn instead
        let x_rev = x_fwd.map(|x| reverse_bits::<D>(x, rev_digits));
        // let x_rev = x_fwd;
        // println!("x_fwd: {:?}, x_rev: {:?}", x_fwd, x_rev);

        // Assert that the the bit reversed indices will not exceed the length of the output.
        // The highest index the loop reaches is: (x_rev[n] + 1)*height - 1
        // The last element of the data is at index: width*height - 1
        // Thus it is sufficient to assert that x_rev[n]<width.
        for r in x_rev {
            assert!(r < width);
        }
        for y in 0..height {
            for (fwd, rev) in x_fwd.iter().zip(x_rev.iter()) {
                let input_index = *fwd + y * width;
                let output_index = y + *rev * height;

                unsafe {
                    let temp = *input.get_unchecked(input_index);
                    *output.get_unchecked_mut(output_index) = temp;
                }
            }
        }
    }
}

pub fn bitreversed_transpose_mirrored<T: Copy, const D: usize>(
    height: usize,
    input: &[T],
    output: &mut [T],
) {
    let width = input.len() / height;

    // Let's make sure the arguments are ok
    assert!(D > 1 && input.len() % height == 0 && input.len() == output.len());

    let strided_width = width / D;
    let rev_digits = {
        assert!(D.is_power_of_two());
        let width_bits = width.trailing_zeros();
        let d_bits = D.trailing_zeros();

        // verify that width is a power of d
        assert!(width_bits % d_bits == 0);
        width_bits / d_bits
    };

    for x in 0..strided_width {
        let mut i = 0;
        let x_fwd = [(); D].map(|_| {
            let value = D * x + i;
            i += 1;
            value
        }); // If we had access to rustc 1.63, we could use std::array::from_fn instead
        let x_rev = x_fwd.map(|x| reverse_bits::<D>(x, rev_digits));
        // let x_rev = x_fwd;
        // println!("x_fwd: {:?}, x_rev: {:?}", x_fwd, x_rev);

        // Assert that the the bit reversed indices will not exceed the length of the output.
        // The highest index the loop reaches is: (x_rev[n] + 1)*height - 1
        // The last element of the data is at index: width*height - 1
        // Thus it is sufficient to assert that x_rev[n]<width.
        for r in x_rev {
            assert!(r < width);
        }
        for y in 0..height {
            for (fwd, rev) in x_fwd.iter().zip(x_rev.iter()) {
                let input_index = *fwd + y * width;
                let output_index = y + *rev * height;

                unsafe {
                    let temp = *input.get_unchecked(output_index);
                    *output.get_unchecked_mut(input_index) = temp;
                }
            }
        }
    }
}

// Repeatedly divide `value` by divisor `D`, `iters` times, and apply the remainders to a new value
// When D is a power of 2, this is exactly equal (implementation and assembly)-wise to a bit reversal
// When D is not a power of 2, think of this function as a logical equivalent to a bit reversal
pub fn reverse_bits<const D: usize>(value: usize, rev_digits: u32) -> usize {
    assert!(D > 1);

    let mut result: usize = 0;
    let mut value = value;
    for _ in 0..rev_digits {
        result = (result * D) + (value % D);
        value = value / D;
    }
    result
}

pub fn create_bitrversed_transpose_lookup<const D: usize>(
    size: usize,
    height: usize,
) -> Vec<usize> {
    let mut res = vec![0; size];

    let width = size / height;

    // Let's make sure the arguments are ok
    assert!(D > 1 && size % height == 0);

    let strided_width = width / D;
    let rev_digits = {
        assert!(D.is_power_of_two());
        let width_bits = width.trailing_zeros();
        let d_bits = D.trailing_zeros();

        // verify that width is a power of d
        assert!(width_bits % d_bits == 0);
        width_bits / d_bits
    };

    for x in 0..strided_width {
        let mut i = 0;
        let x_fwd = [(); D].map(|_| {
            let value = D * x + i;
            i += 1;
            value
        }); // If we had access to rustc 1.63, we could use std::array::from_fn instead
        let x_rev = x_fwd.map(|x| reverse_bits::<D>(x, rev_digits));
        // let x_rev = x_fwd;
        // println!("x_fwd: {:?}, x_rev: {:?}", x_fwd, x_rev);

        // Assert that the the bit reversed indices will not exceed the length of the output.
        // The highest index the loop reaches is: (x_rev[n] + 1)*height - 1
        // The last element of the data is at index: width*height - 1
        // Thus it is sufficient to assert that x_rev[n]<width.
        for r in x_rev {
            assert!(r < width);
        }
        for y in 0..height {
            for (fwd, rev) in x_fwd.iter().zip(x_rev.iter()) {
                let input_index = *fwd + y * width;
                let output_index = y + *rev * height;

                res[output_index] = input_index;
            }
        }
    }

    res
}

pub fn get_input_index(output_index: usize, lookup: &[usize]) -> usize {
    lookup[output_index]
}

pub fn nonbitreversed_transpose<T: Copy, const D: usize>(
    height: usize,
    input: &[T],
    output: &mut [T],
) {
    let width = input.len() / height;

    // Let's make sure the arguments are ok
    assert!(D > 1 && input.len() % height == 0 && input.len() == output.len());

    let strided_width = width / D;
    // let rev_digits = {
    //     assert!(D.is_power_of_two());
    //     let width_bits = width.trailing_zeros();
    //     let d_bits = D.trailing_zeros();

    //     // verify that width is a power of d
    //     assert!(width_bits % d_bits == 0);
    //     width_bits / d_bits
    // };

    for x in 0..strided_width {
        let mut i = 0;
        let x_fwd = [(); D].map(|_| {
            let value = D * x + i;
            i += 1;
            value
        }); // If we had access to rustc 1.63, we could use std::array::from_fn instead
            // let x_rev = x_fwd.map(|x| reverse_bits::<D>(x, rev_digits));
        let x_rev = x_fwd;

        // Assert that the the bit reversed indices will not exceed the length of the output.
        // The highest index the loop reaches is: (x_rev[n] + 1)*height - 1
        // The last element of the data is at index: width*height - 1
        // Thus it is sufficient to assert that x_rev[n]<width.
        for r in x_rev {
            assert!(r < width);
        }
        for y in 0..height {
            for (fwd, rev) in x_fwd.iter().zip(x_rev.iter()) {
                let input_index = *fwd + y * width;
                let output_index = y + *rev * height;

                unsafe {
                    let temp = *input.get_unchecked(input_index);
                    *output.get_unchecked_mut(output_index) = temp;
                }
            }
        }
    }
}

#[test]
fn test_bitreversed_transpose() {
    let input = (0..256).map(|x| x as u8).collect::<Vec<_>>();
    let mut output_bitreversed = vec![0; input.len()];
    // let mut output_bitreversed_double = vec![0; input.len()];
    // let mut output_nonbitreversed = vec![0; input.len()];
    // let mut output_bitreverse_inplace = vec![0; input.len()];
    let mut output = vec![0; input.len()];

    bitreversed_transpose::<_, 4>(16, &input, &mut output_bitreversed);

    bitreversed_transpose_mirrored::<_, 4>(16, &output_bitreversed, &mut output);

    // bitreversed_transpose::<_, 4>(16, &output_bitreversed, &mut output_bitreversed_double);

    // nonbitreversed_transpose::<_, 4>(16, &input, &mut output_nonbitreversed);

    // output_bitreverse_inplace = input.clone();
    // bitreverse_enumeration_inplace(&mut output_bitreverse_inplace);
    // bitreversed_transpose::<_, 4>(16, &output_bitreverse_inplace, &mut output_bitreversed_double);

    // println!("{:?}", output_bitreversed);
    // println!("{:?}", output_bitreverse_inplace);
    // println!("{:?}", output_bitreversed_double);
    // println!("{:?}", output_nonbitreversed);
    println!("{:?}", output);
    println!("equal: {}", output == input);
    // assert_eq!(output, );
}

pub trait LoadStore<T>: DerefMut {
    unsafe fn load(&self, idx: usize) -> T;
    unsafe fn store(&mut self, val: T, idx: usize);
}

impl LoadStore<Mersenne31Complex> for &mut [Mersenne31Complex] {
    #[inline(always)]
    unsafe fn load(&self, idx: usize) -> Mersenne31Complex {
        debug_assert!(idx < self.len());
        *self.get_unchecked(idx)
    }
    #[inline(always)]
    unsafe fn store(&mut self, val: Mersenne31Complex, idx: usize) {
        debug_assert!(idx < self.len());
        *self.get_unchecked_mut(idx) = val;
    }
}
impl<const N: usize> LoadStore<Mersenne31Complex> for &mut [Mersenne31Complex; N] {
    #[inline(always)]
    unsafe fn load(&self, idx: usize) -> Mersenne31Complex {
        debug_assert!(idx < self.len());
        *self.get_unchecked(idx)
    }
    #[inline(always)]
    unsafe fn store(&mut self, val: Mersenne31Complex, idx: usize) {
        debug_assert!(idx < self.len());
        *self.get_unchecked_mut(idx) = val;
    }
}

impl LoadStore<Mersenne31ComplexVectorized> for &mut [Mersenne31ComplexVectorized] {
    #[inline(always)]
    unsafe fn load(&self, idx: usize) -> Mersenne31ComplexVectorized {
        debug_assert!(idx < self.len());
        *self.get_unchecked(idx)
    }
    #[inline(always)]
    unsafe fn store(&mut self, val: Mersenne31ComplexVectorized, idx: usize) {
        debug_assert!(idx < self.len());
        *self.get_unchecked_mut(idx) = val;
    }
}
impl<const N: usize> LoadStore<Mersenne31ComplexVectorized>
    for &mut [Mersenne31ComplexVectorized; N]
{
    #[inline(always)]
    unsafe fn load(&self, idx: usize) -> Mersenne31ComplexVectorized {
        debug_assert!(idx < self.len());
        *self.get_unchecked(idx)
    }
    #[inline(always)]
    unsafe fn store(&mut self, val: Mersenne31ComplexVectorized, idx: usize) {
        debug_assert!(idx < self.len());
        *self.get_unchecked_mut(idx) = val;
    }
}

impl LoadStore<Mersenne31ComplexVectorizedInterleaved>
    for &mut [Mersenne31ComplexVectorizedInterleaved]
{
    #[inline(always)]
    unsafe fn load(&self, idx: usize) -> Mersenne31ComplexVectorizedInterleaved {
        debug_assert!(idx < self.len());
        *self.get_unchecked(idx)
    }
    #[inline(always)]
    unsafe fn store(&mut self, val: Mersenne31ComplexVectorizedInterleaved, idx: usize) {
        debug_assert!(idx < self.len());
        *self.get_unchecked_mut(idx) = val;
    }
}
impl<const N: usize> LoadStore<Mersenne31ComplexVectorizedInterleaved>
    for &mut [Mersenne31ComplexVectorizedInterleaved; N]
{
    #[inline(always)]
    unsafe fn load(&self, idx: usize) -> Mersenne31ComplexVectorizedInterleaved {
        debug_assert!(idx < self.len());
        *self.get_unchecked(idx)
    }
    #[inline(always)]
    unsafe fn store(&mut self, val: Mersenne31ComplexVectorizedInterleaved, idx: usize) {
        debug_assert!(idx < self.len());
        *self.get_unchecked_mut(idx) = val;
    }
}

impl<const N: usize> LoadStore<[Mersenne31Complex; N]> for &mut [[Mersenne31Complex; N]] {
    #[inline(always)]
    unsafe fn load(&self, idx: usize) -> [Mersenne31Complex; N] {
        debug_assert!(idx < self.len());
        *self.get_unchecked(idx)
    }
    #[inline(always)]
    unsafe fn store(&mut self, val: [Mersenne31Complex; N], idx: usize) {
        debug_assert!(idx < self.len());
        *self.get_unchecked_mut(idx) = val;
    }
}

impl<const N: usize, const M: usize> LoadStore<Mersenne31ComplexVectorizedInterleaved>
    for &mut RowMajorTraceFixedColumnsView<Mersenne31Field, N, M>
{
    #[inline(always)]
    unsafe fn load(&self, idx: usize) -> Mersenne31ComplexVectorizedInterleaved {
        debug_assert!(idx < self.len());
        let row = self.get_row(idx);
        row.as_ptr()
            .cast::<Mersenne31ComplexVectorizedInterleaved>()
            .read()
        // *self.get_unchecked(idx)
    }
    #[inline(always)]
    unsafe fn store(&mut self, val: Mersenne31ComplexVectorizedInterleaved, idx: usize) {
        debug_assert!(idx < self.len());
        let val_cast = val.chunk_0.0.as_ptr().cast::<[Mersenne31Field; M]>().read();
        let old_val = self.get_row_mut(idx);
        *old_val = val_cast;
        // *self.get_unchecked_mut(idx) = val;
    }
}

pub(crate) trait Load: Deref + Sync + Send + Copy {
    unsafe fn load(&self, idx: usize) -> Mersenne31Complex;
}

impl Load for &[Mersenne31Complex] {
    #[inline(always)]
    unsafe fn load(&self, idx: usize) -> Mersenne31Complex {
        debug_assert!(idx < self.len());
        *self.get_unchecked(idx)
    }
}

// Loop over exact chunks of the provided buffer. Very similar in semantics to ChunksExactMut, but generates smaller code and requires no modulo operations
// Returns Ok() if every element ended up in a chunk, Err() if there was a remainder
pub fn iter_chunks<T>(
    mut buffer: &mut [T],
    chunk_size: usize,
    mut chunk_fn: impl FnMut(&mut [T]),
) -> Result<(), ()> {
    // Loop over the buffer, splicing off chunk_size at a time, and calling chunk_fn on each
    while buffer.len() >= chunk_size {
        let (head, tail) = buffer.split_at_mut(chunk_size);
        buffer = tail;

        chunk_fn(head);
    }

    // We have a remainder if there's data still in the buffer -- in which case we want to indicate to the caller that there was an unwanted remainder
    if buffer.len() == 0 {
        Ok(())
    } else {
        Err(())
    }
}

pub fn iter_chunks_zip<T, U>(
    mut buffer0: &mut [T],
    mut buffer1: &[U],
    chunk_size: usize,
    mut chunk_fn: impl FnMut(&mut [T], &[U]),
) -> Result<(), ()> {
    assert!(buffer0.len() == buffer1.len());
    // Loop over the buffer, splicing off chunk_size at a time, and calling chunk_fn on each
    while buffer0.len() >= chunk_size {
        let (head0, tail0) = buffer0.split_at_mut(chunk_size);
        buffer0 = tail0;
        let (head1, tail1) = buffer1.split_at(chunk_size);
        buffer1 = tail1;

        chunk_fn(head0, head1);
    }

    // We have a remainder if there's data still in the buffer -- in which case we want to indicate to the caller that there was an unwanted remainder
    if buffer0.len() == 0 {
        Ok(())
    } else {
        Err(())
    }
}

pub fn slice_to_slice_of_slices_with_length_mut(
    slice: &mut [Mersenne31Complex],
    inner_slice_length: usize,
) -> &mut [&mut [Mersenne31Complex]] {
    if slice.len() % inner_slice_length != 0 {
        panic!("Slice length must be a multiple of inner slice length");
    }

    let num_inner_slices = slice.len() / inner_slice_length;
    let slice_of_slices: &mut [&mut [Mersenne31Complex]] = unsafe {
        std::slice::from_raw_parts_mut(
            slice.as_mut_ptr() as *mut &mut [Mersenne31Complex],
            num_inner_slices,
        )
    };

    // for inner_slice in slice_of_slices {
    //     assert_eq!(inner_slice.len(), inner_slice_length);
    // }

    slice_of_slices
}

pub fn cast_to_mut_slice_of_arrays<T, const N: usize>(slice: &mut [T]) -> &mut [[T; N]] {
    let num_arrays = slice.len() / N;
    assert_eq!(num_arrays * N, slice.len());

    let slice_of_arrays: &mut [[T; N]] =
        unsafe { std::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut [T; N], num_arrays) };

    slice_of_arrays
}

pub fn cast_to_mut_slice_of_vectors<T>(
    slice: &mut [T],
) -> &mut [Mersenne31ComplexVectorizedInterleaved] {
    let num_vectors = slice.len() / WIDTH;
    assert_eq!(num_vectors * WIDTH, slice.len());

    let slice_of_vectors: &mut [Mersenne31ComplexVectorizedInterleaved] = unsafe {
        std::slice::from_raw_parts_mut(
            slice.as_mut_ptr() as *mut Mersenne31ComplexVectorizedInterleaved,
            num_vectors,
        )
    };

    slice_of_vectors
}
