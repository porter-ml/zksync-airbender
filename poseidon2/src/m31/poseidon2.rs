pub use super::naive::*;

#[cfg(test)]
mod tests {
    use super::*;

    type F = field::Mersenne31Field;

    #[test]
    fn test_poseidon2_width_16() {
        let mut input: [F; 16] = [
            894848333, 1437655012, 1200606629, 1690012884, 71131202, 1749206695, 1717947831,
            120589055, 19776022, 42382981, 1831865506, 724844064, 171220207, 1299207443, 227047920,
            1783754913,
        ]
        .map(F::from_nonreduced_u32);

        let expected: [F; 16] = [
            1124552602, 2127602268, 1834113265, 1207687593, 1891161485, 245915620, 981277919,
            627265710, 1534924153, 1580826924, 887997842, 1526280482, 547791593, 1028672510,
            1803086471, 323071277,
        ]
        .map(F::from_nonreduced_u32);

        poseidon_permutation(&mut input);
        assert_eq!(input, expected);
    }
}
