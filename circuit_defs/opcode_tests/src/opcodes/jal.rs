struct Test;
impl crate::TestCase for Test {
    const TESTCASES: &str = include_str!("../data/jal-01.S");
    const MATCH: &str = "TEST_JAL_OP";
    const PATCH_MATCH: Option<&str> = Some("jal");
    const OPFIELDS: &[usize] = &[0, 2, 3];
    const PATCH_IMMEDIATE: Option<usize> = Some(3);
    const PATCH_IMMEDIATE_WITH_REGISTER: bool = false;
    const INITIAL_REGISTERS_INDEX: &[(usize, usize)] = &[];
    const PATCH_INITIAL_REGISTER: Option<&str> = None;
    const FINAL_REGISTER_INDEX: Option<(usize, usize)> = None;
}
#[test]
fn test() {
    <Test as crate::TestCase>::test();
}
