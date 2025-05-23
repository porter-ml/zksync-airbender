#[test]
fn test() {
    const CSRRW_NONDETERMINISM_OPCODE: u32 = 0x7c0110f3; // 1984
    const CSRRW_BLAKE2ROUNDEXTENDED_OPCODE: u32 = 0x7c7110f3; // 1991
    const CSRRW_U256BIGINTOPS_OPCODE: u32 = 0x7ca110f3; // 1994
    crate::test_single_opcode(
        "csrrw x1, 1984, x2",
        Some(CSRRW_NONDETERMINISM_OPCODE),
        [0; 32],
        None,
    );
    crate::test_single_opcode(
        "csrrw x1, 1991, x2",
        Some(CSRRW_BLAKE2ROUNDEXTENDED_OPCODE),
        [0; 32],
        None,
    );
    crate::test_single_opcode(
        "csrrw x1, 1994, x2",
        Some(CSRRW_U256BIGINTOPS_OPCODE),
        [0; 32],
        None,
    );
}
