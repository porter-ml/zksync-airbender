struct Test;
impl crate::TestCase for Test {
    const TESTCASES: &str = include_str!("../data/xori-01.S");
    const MATCH: &str = "TEST_IMM_OP";
    const PATCH_MATCH: Option<&str> = None;
    const OPFIELDS: &[usize] = &[0, 1, 2, 5];
    const PATCH_IMMEDIATE: Option<usize> = Some(5);
    const PATCH_IMMEDIATE_WITH_REGISTER: bool = false;
    const INITIAL_REGISTERS_INDEX: &[(usize, usize)] = &[(2, 4)];
    const PATCH_INITIAL_REGISTER: Option<&str> = None;
    const FINAL_REGISTER_INDEX: Option<(usize, usize)> = Some((1, 3));
}
#[test]
fn test() {
    <Test as crate::TestCase>::test();
}
