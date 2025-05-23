#[inline(always)]
// #[cfg(target_arch = "riscv32")]
#[cfg(all(target_arch = "riscv32", target_feature = "zbb"))]
pub fn rotate_right<const AMT: u32>(value: u32) -> u32 {
    let mut output;
    unsafe {
        core::arch::asm!(
            "rori {rd}, {rs1}, {amt}",
            rs1 = in(reg) value,
            rd = out(reg) output,
            amt = const AMT,
            options(nomem, nostack, preserves_flags)
        );
    }

    output
}

#[inline(always)]
// #[cfg(not(target_arch = "riscv32"))]
#[cfg(not(all(target_arch = "riscv32", target_feature = "zbb")))]
pub fn rotate_right<const AMT: u32>(value: u32) -> u32 {
    value.rotate_right(AMT)
}
