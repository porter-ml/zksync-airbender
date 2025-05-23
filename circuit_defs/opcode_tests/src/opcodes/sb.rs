struct Test;
impl crate::TestCase for Test {
    const TESTCASES: &str = include_str!("../data/sb-align-01.S");
    const MATCH: &str = "TEST_STORE";
    const PATCH_MATCH: Option<&str> = None;
    const OPFIELDS: &[usize] = &[8, 4, 3, 6];
    const PATCH_IMMEDIATE: Option<usize> = Some(6);
    const PATCH_IMMEDIATE_WITH_REGISTER: bool = true;
    const INITIAL_REGISTERS_INDEX: &[(usize, usize)] = &[(4, 5), (3, 10)];
    const PATCH_INITIAL_REGISTER: Option<&str> = Some("2097152");
    const FINAL_REGISTER_INDEX: Option<(usize, usize)> = None;
}
#[test]
fn test() {
    <Test as crate::TestCase>::test();
}
