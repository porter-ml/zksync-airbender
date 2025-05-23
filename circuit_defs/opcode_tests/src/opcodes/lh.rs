struct Test;
impl crate::TestCase for Test {
    const TESTCASES: &str = include_str!("../data/lh-align-01.S");
    const MATCH: &str = "TEST_LOAD";
    const PATCH_MATCH: Option<&str> = None;
    const OPFIELDS: &[usize] = &[7, 4, 3, 5];
    const PATCH_IMMEDIATE: Option<usize> = Some(5);
    const PATCH_IMMEDIATE_WITH_REGISTER: bool = true;
    const INITIAL_REGISTERS_INDEX: &[(usize, usize)] = &[(3, 9)];
    const PATCH_INITIAL_REGISTER: Option<&str> = Some("2048");
    const FINAL_REGISTER_INDEX: Option<(usize, usize)> = None;
}
#[test]
fn test() {
    <Test as crate::TestCase>::test();
}
