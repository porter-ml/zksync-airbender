struct Test;
impl crate::TestCase for Test {
    const TESTCASES: &str = include_str!("../data/rem-01.S");
    const MATCH: &str = "TEST_RR_OP";
    const PATCH_MATCH: Option<&str> = None;
    const OPFIELDS: &[usize] = &[0, 1, 2, 3];
    const PATCH_IMMEDIATE: Option<usize> = None;
    const PATCH_IMMEDIATE_WITH_REGISTER: bool = false;
    const INITIAL_REGISTERS_INDEX: &[(usize, usize)] = &[(2, 5), (3, 6)];
    const PATCH_INITIAL_REGISTER: Option<&str> = None;
    const FINAL_REGISTER_INDEX: Option<(usize, usize)> = Some((1, 4));
}
#[test]
fn test() {
    <Test as crate::TestCase>::test();
    crate::test_single_opcode(
        "rem, x3, x1, x2",
        None,
        {
            let mut xs = [0; 32];
            xs[1] = i32::MIN as u32;
            xs[2] = -1_i32 as u32;
            xs
        },
        Some((3, 0)),
    );
}
