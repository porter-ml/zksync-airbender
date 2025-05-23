mod delegation;
pub mod naive;
mod poseidon2;

pub const HASH_SIZE_U32_WORDS: usize = 8;

use field::{Field, Mersenne31Field, PrimeField};

pub use self::delegation::Poseidon2Compressor;
pub use self::poseidon2::{poseidon2_compress, poseidon_permutation};

pub(crate) const POSEIDON2_INTERNAL_MATRIX_DIAG_16_SHIFTS: [u8; 15] =
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13, 14, 15, 16];

pub const RF: usize = 8;
pub const RP: usize = 14;
pub const EXTERNAL_INITIAL_CONSTANTS: [[u32; 16]; RF / 2] = [
    [
        670752198, 2052960689, 867595173, 1121120522, 1732216065, 1777538858, 974826695, 857651441,
        1509218160, 933669702, 308743513, 1606546523, 1395707998, 1248974626, 733565087,
        1614794869,
    ],
    [
        1457687568, 311580733, 2055660101, 1735187654, 1563765150, 358422393, 615368408,
        1022914986, 1745808542, 1451694789, 1010294888, 478426997, 974777474, 836569592, 553962986,
        354722588,
    ],
    [
        1099724285, 957403621, 1171073730, 1314307614, 1575313895, 511348931, 1777322674,
        743793854, 821769216, 365270850, 2100202195, 1610545562, 1781773041, 1642480066, 968153742,
        107763776,
    ],
    [
        304102504, 1048805912, 670079580, 1825005418, 699322108, 372969254, 1347088819, 1017368981,
        695522824, 1491107118, 1656304581, 934311777, 1538050768, 1121275927, 1281424936,
        1609172128,
    ],
];
pub const EXTERNAL_TERMINAL_CONSTANTS: [[u32; 16]; RF / 2] = [
    [
        302658704, 2055094098, 16103019, 802016690, 359041126, 1491417545, 151742200, 122792040,
        802809388, 2143547951, 2020259742, 437172020, 1610027373, 1217130568, 1833171446,
        2135403312,
    ],
    [
        60728125, 173288461, 1580136315, 2058149815, 1766051075, 458819359, 1495214374, 696367131,
        367271168, 4549961, 718747682, 1943893587, 1536582683, 1574838747, 1735444335, 848039704,
    ],
    [
        1689611743, 173154748, 427470023, 1004172913, 2077368442, 782638163, 1744615017,
        1082619536, 297763826, 1160504957, 618979668, 1687696498, 37211066, 2117379525, 1790329919,
        1183379851,
    ],
    [
        545339302, 1229207547, 723170958, 1927785244, 1080767281, 1903150401, 1929310598, 95801870,
        637696247, 1214340530, 1722126248, 1823128363, 926128391, 210718841, 1667233644, 688337540,
    ],
];
pub const INTERNAL_CONSTANTS: [u32; RP] = [
    129024239, 1282387121, 2004475442, 535738304, 1985680653, 895998816, 1108547306, 776893336,
    1108245527, 574331301, 1825109420, 1194870642, 1497066195, 1664793266,
];

pub const EXTERNAL_MATRIX: [[Mersenne31Field; 16]; 16] = const {
    let mut result = [[Mersenne31Field::ZERO; 16]; 16];
    // row by row
    let mut column_set = 0;
    while column_set < 4 {
        let mut row_set = 0;
        while row_set < 4 {
            // [ 2 3 1 1 ]
            // [ 1 2 3 1 ]
            // [ 1 1 2 3 ]
            // [ 3 1 1 2 ]

            result[row_set * 4][column_set * 4] = Mersenne31Field(2);
            result[row_set * 4][column_set * 4 + 1] = Mersenne31Field(3);
            result[row_set * 4][column_set * 4 + 2] = Mersenne31Field(1);
            result[row_set * 4][column_set * 4 + 3] = Mersenne31Field(1);

            result[row_set * 4 + 1][column_set * 4] = Mersenne31Field(1);
            result[row_set * 4 + 1][column_set * 4 + 1] = Mersenne31Field(2);
            result[row_set * 4 + 1][column_set * 4 + 2] = Mersenne31Field(3);
            result[row_set * 4 + 1][column_set * 4 + 3] = Mersenne31Field(1);

            result[row_set * 4 + 2][column_set * 4] = Mersenne31Field(1);
            result[row_set * 4 + 2][column_set * 4 + 1] = Mersenne31Field(1);
            result[row_set * 4 + 2][column_set * 4 + 2] = Mersenne31Field(2);
            result[row_set * 4 + 2][column_set * 4 + 3] = Mersenne31Field(3);

            result[row_set * 4 + 3][column_set * 4] = Mersenne31Field(3);
            result[row_set * 4 + 3][column_set * 4 + 1] = Mersenne31Field(1);
            result[row_set * 4 + 3][column_set * 4 + 2] = Mersenne31Field(1);
            result[row_set * 4 + 3][column_set * 4 + 3] = Mersenne31Field(2);

            if row_set == column_set {
                result[row_set * 4][column_set * 4].0 <<= 1;
                result[row_set * 4][column_set * 4 + 1].0 <<= 1;
                result[row_set * 4][column_set * 4 + 2].0 <<= 1;
                result[row_set * 4][column_set * 4 + 3].0 <<= 1;

                result[row_set * 4 + 1][column_set * 4].0 <<= 1;
                result[row_set * 4 + 1][column_set * 4 + 1].0 <<= 1;
                result[row_set * 4 + 1][column_set * 4 + 2].0 <<= 1;
                result[row_set * 4 + 1][column_set * 4 + 3].0 <<= 1;

                result[row_set * 4 + 2][column_set * 4].0 <<= 1;
                result[row_set * 4 + 2][column_set * 4 + 1].0 <<= 1;
                result[row_set * 4 + 2][column_set * 4 + 2].0 <<= 1;
                result[row_set * 4 + 2][column_set * 4 + 3].0 <<= 1;

                result[row_set * 4 + 3][column_set * 4].0 <<= 1;
                result[row_set * 4 + 3][column_set * 4 + 1].0 <<= 1;
                result[row_set * 4 + 3][column_set * 4 + 2].0 <<= 1;
                result[row_set * 4 + 3][column_set * 4 + 3].0 <<= 1;
            }

            row_set += 1;
        }
        column_set += 1;
    }

    result
};

pub const INTERNAL_MATRIX: [[Mersenne31Field; 16]; 16] = const {
    let mut result = [[Mersenne31Field::ONE; 16]; 16];
    // only overwrite diagonal
    result[0][0] = Mersenne31Field::MINUS_ONE;
    let mut i = 1;
    while i < 16 {
        result[i][i] = Mersenne31Field((1 << POSEIDON2_INTERNAL_MATRIX_DIAG_16_SHIFTS[i - 1]) + 1);
        i += 1;
    }

    result
};

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn calc_test() {
        dbg!(EXTERNAL_MATRIX);
        dbg!(INTERNAL_MATRIX);
    }
}
