struct Test;
impl crate::TestCase for Test {
    const TESTCASES: &str = include_str!("../data/auipc-01.S");
    const MATCH: &str = "TEST_AUIPC";
    const PATCH_MATCH: Option<&str> = None;
    const OPFIELDS: &[usize] = &[0, 1, 3];
    const PATCH_IMMEDIATE: Option<usize> = Some(3);
    const PATCH_IMMEDIATE_WITH_REGISTER: bool = false;
    const INITIAL_REGISTERS_INDEX: &[(usize, usize)] = &[];
    const PATCH_INITIAL_REGISTER: Option<&str> = None;
    const FINAL_REGISTER_INDEX: Option<(usize, usize)> = Some((1, 2));
}
#[test]
fn test() {
    <Test as crate::TestCase>::test();
}
