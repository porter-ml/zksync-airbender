struct Test;
impl crate::TestCase for Test {
    const TESTCASES: &str = include_str!("../data/lui-01.S");
    const MATCH: &str = "TEST_CASE";
    const PATCH_MATCH: Option<&str> = None;
    const OPFIELDS: &[usize] = &[5, 6];
    const PATCH_IMMEDIATE: Option<usize> = Some(6);
    const PATCH_IMMEDIATE_WITH_REGISTER: bool = false;
    const INITIAL_REGISTERS_INDEX: &[(usize, usize)] = &[];
    const PATCH_INITIAL_REGISTER: Option<&str> = None;
    const FINAL_REGISTER_INDEX: Option<(usize, usize)> = Some((1, 2));
}
#[test]
fn test() {
    <Test as crate::TestCase>::test();
}
