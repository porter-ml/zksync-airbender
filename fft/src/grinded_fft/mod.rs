pub mod butterfly16;
pub mod butterfly2;
pub mod butterfly4;
pub mod butterfly8;
pub mod grinded_lde;
pub mod radix_4_step;
pub mod twiddles;
pub mod utils;

/// Represents a FFT direction, IE a forward FFT or an inverse FFT
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum FftDirection {
    Forward,
    Inverse,
}
