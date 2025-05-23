use crate::constraint::*;
use crate::cs::circuit::Circuit;
use crate::types::*;
use field::PrimeField;

pub(crate) fn enforce_add_sub_relation<F: PrimeField, CS: Circuit<F>>(
    cs: &mut CS,
    carry_out: Boolean,
    a_s: &[Register<F>],
    b_s: &[Register<F>],
    c_s: &[Register<F>],
    flags: &[Boolean],
) {
    assert_eq!(a_s.len(), b_s.len());
    assert_eq!(a_s.len(), c_s.len());
    assert_eq!(a_s.len(), flags.len());

    let mut constraint_low = Constraint::empty();
    let mut constraint_high = Constraint::empty();

    let mut dependencies = vec![];

    for (((a, b), c), flag) in a_s.iter().zip(b_s.iter()).zip(c_s.iter()).zip(flags.iter()) {
        let Boolean::Is(flag) = *flag else { todo!() };
        let Num::Var(a_low) = a.0[0] else { todo!() };
        let Num::Var(a_high) = a.0[1] else { todo!() };
        let Num::Var(b_low) = b.0[0] else { todo!() };
        let Num::Var(b_high) = b.0[1] else { todo!() };
        let Num::Var(c_low) = c.0[0] else { todo!() };
        let Num::Var(c_high) = c.0[1] else { todo!() };
        constraint_low = constraint_low + (Term::from(flag) * Term::from(a_low));
        constraint_low = constraint_low + (Term::from(flag) * Term::from(b_low));
        constraint_low = constraint_low - (Term::from(flag) * Term::from(c_low));

        constraint_high = constraint_high + (Term::from(flag) * Term::from(a_high));
        constraint_high = constraint_high + (Term::from(flag) * Term::from(b_high));
        constraint_high = constraint_high - (Term::from(flag) * Term::from(c_high));

        dependencies.push((flag, a_low, b_low, c_low)); // we only need that for carry low
    }

    let carry_intermediate = Boolean::new(cs);
    let carry_intermediate_var = carry_intermediate.get_variable().unwrap();

    let value_fn = move |placer: &mut CS::WitnessPlacer| {
        use crate::cs::witness_placer::*;

        let mut carry = <CS::WitnessPlacer as WitnessTypeSet<F>>::Mask::constant(false);

        for (flag, a, b, c) in dependencies.iter() {
            let mask = placer.get_boolean(*flag);
            let mut result = placer.get_u16(*a).widen();
            let b = placer.get_u16(*b).widen();
            let c = placer.get_u16(*c).widen();
            result.add_assign(&b);
            result.sub_assign(&c);
            let carry_candidate = result.get_bit(16);
            carry.assign_masked(&mask, &carry_candidate);
        }

        placer.assign_mask(carry_intermediate_var, &carry);
    };

    cs.set_values(value_fn);

    let constraint_low = constraint_low
        - Term::<F>::from((
            F::from_u64_unchecked(1 << 16),
            carry_intermediate.get_variable().unwrap(),
        ));
    cs.add_constraint(constraint_low);

    let constraint_high = constraint_high
        + Term::<F>::from(carry_intermediate.get_variable().unwrap())
        - Term::<F>::from((
            F::from_u64_unchecked(1 << 16),
            carry_out.get_variable().unwrap(),
        ));
    cs.add_constraint(constraint_high);
}

// #[track_caller]
// fn choose_enforce_add_sub_value_fn<F: PrimeField>(
//     num_terms: usize,
// ) -> fn(WitnessGenSource<'_, F>, WitnessGenDest<'_, F>, &[F], &TableDriver<F>, TableType) {
//     assert!(num_terms > 0);

//     seq_macro::seq!(M in 0..6 {
//         if num_terms == M {
//             // return true_fn_~N::collapse_max_quadratic_constraint_into_fixed_witness_eval_fn_~M::<F, true, N, M>;
//             return enforce_add_sub_value_fn::<F, M>;
//         }
//     });

//     panic!("unsupported number of terms: {}", num_terms);
// }

// #[track_caller]
// #[inline(always)]
// fn enforce_add_sub_value_fn<F: PrimeField, const NUM_TERMS: usize>(
//     input: WitnessGenSource<'_, F>,
//     mut output: WitnessGenDest<'_, F>,
//     _constants: &[F],
//     _table_driver: &TableDriver<F>,
//     _table_type: TableType,
// ) {
//     for i in 0..NUM_TERMS {
//         let flag = input[4 * i].as_boolean();
//         if flag {
//             let a = input[4 * i + 1].as_u64_reduced();
//             let b = input[4 * i + 2].as_u64_reduced();
//             let c = input[4 * i + 3].as_u64_reduced();
//             // a + b = c + of
//             let t = a + b - c;
//             let carry = t >= (1 << 16);

//             output[0] = F::from_boolean(carry);
//             return;
//         }
//     }

//     output[0] = F::ZERO;
// }
