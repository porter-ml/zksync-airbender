use criterion::*;
use field::Mersenne31Field;
use poseidon2::m31::poseidon2_compress;

fn naive(crit: &mut Criterion) {
    let input: [Mersenne31Field; 16] = [
        894848333, 1437655012, 1200606629, 1690012884, 71131202, 1749206695, 1717947831, 120589055,
        19776022, 42382981, 1831865506, 724844064, 171220207, 1299207443, 227047920, 1783754913,
    ]
    .map(Mersenne31Field::from_nonreduced_u32);

    crit.bench_function("Naive impl - compression", |b| {
        b.iter(|| {
            let _ = poseidon2_compress(core::hint::black_box(&input));
        });
    });
}

criterion_group!(benches, naive,);

criterion_main!(benches);
