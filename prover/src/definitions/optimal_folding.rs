#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FoldingDescription {
    pub initial_degree: usize,
    pub folding_sequence: &'static [usize],
    pub total_caps_size_log2: usize,
    pub final_monomial_degree_log2: usize,
}

const DUMMY_DESCRIPTION: FoldingDescription = FoldingDescription {
    initial_degree: 0,
    folding_sequence: &[],
    total_caps_size_log2: 0,
    final_monomial_degree_log2: 0,
};

// we make this array such that one can access this array at index equal to the trace len log 2,
// and get a sequence
pub const OPTIMAL_FOLDING_PROPERTIES: &[FoldingDescription] = &[
    DUMMY_DESCRIPTION, // 0
    DUMMY_DESCRIPTION, // 1
    DUMMY_DESCRIPTION, // 2
    DUMMY_DESCRIPTION, // 3
    DUMMY_DESCRIPTION, // 4
    DUMMY_DESCRIPTION, // 5
    DUMMY_DESCRIPTION, // 6
    DUMMY_DESCRIPTION, // 7
    DUMMY_DESCRIPTION, // 8
    DUMMY_DESCRIPTION, // 9
    DUMMY_DESCRIPTION, // 10
    DUMMY_DESCRIPTION, // 11
    DUMMY_DESCRIPTION, // 12
    DUMMY_DESCRIPTION, // 13
    DUMMY_DESCRIPTION, // 14
    DUMMY_DESCRIPTION, // 15
    DUMMY_DESCRIPTION, // 16
    FoldingDescription {
        initial_degree: 17,
        folding_sequence: &[3, 3, 3, 3],
        total_caps_size_log2: 7,
        final_monomial_degree_log2: 5,
    }, // 17
    FoldingDescription {
        initial_degree: 18,
        folding_sequence: &[3, 3, 3, 3],
        total_caps_size_log2: 7,
        final_monomial_degree_log2: 6,
    }, // 18
    FoldingDescription {
        initial_degree: 19,
        folding_sequence: &[4, 3, 3, 3],
        total_caps_size_log2: 7,
        final_monomial_degree_log2: 6,
    }, // 19
    FoldingDescription {
        initial_degree: 20,
        folding_sequence: &[3, 3, 3, 3, 3], // 4, 4, 4
        total_caps_size_log2: 7,
        final_monomial_degree_log2: 5, // 8
    }, // 20
    FoldingDescription {
        initial_degree: 21,
        folding_sequence: &[4, 3, 3, 3, 3],
        total_caps_size_log2: 7,
        final_monomial_degree_log2: 5,
    }, // 21
    FoldingDescription {
        initial_degree: 22,
        folding_sequence: &[4, 4, 3, 3, 3],
        total_caps_size_log2: 7,
        final_monomial_degree_log2: 5,
    }, // 22
    FoldingDescription {
        initial_degree: 23,
        folding_sequence: &[4, 4, 3, 3, 3],
        total_caps_size_log2: 7,
        final_monomial_degree_log2: 6,
    }, // 23
    FoldingDescription {
        initial_degree: 24,
        folding_sequence: &[4, 4, 4, 3, 3],
        total_caps_size_log2: 7,
        final_monomial_degree_log2: 6,
    }, // 24
    FoldingDescription {
        initial_degree: 25,
        folding_sequence: &[4, 4, 4, 4, 3],
        total_caps_size_log2: 7,
        final_monomial_degree_log2: 6,
    }, // 25
];
