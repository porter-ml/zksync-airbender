use super::*;
use crate::constraint::{Constraint, Term};
use crate::cs::circuit::*;
use crate::cs::witness_placer::*;
use crate::types::*;
use field::PrimeField;

#[track_caller]
pub fn mask_linear_term<F: PrimeField, C: Circuit<F>>(
    cs: &mut C,
    term: Term<F>,
    mask: Boolean,
) -> Variable {
    cs.add_variable_from_constraint(term * mask.get_terms())
}

#[track_caller]
pub fn collapse_max_quadratic_constraint_into<F: PrimeField, C: Circuit<F>>(
    cs: &mut C,
    constraint: Constraint<F>,
    result: Variable,
) {
    return collapse_max_quadratic_constraint_into_fixed(cs, constraint, result);
}

pub(crate) fn check_constants<F: PrimeField>(num1: &Num<F>, num2: &Num<F>) -> (bool, bool) {
    let first_is_constant = match num1 {
        Num::Constant(_) => true,
        _ => false,
    };

    let second_is_constant = match num2 {
        Num::Constant(_) => true,
        _ => false,
    };

    (first_is_constant, second_is_constant)
}

pub fn mask_by_boolean_into_accumulator_constraint<F: PrimeField>(
    boolean: &Boolean,
    variable: &Num<F>,
    accumulator: Constraint<F>,
) -> Constraint<F> {
    match (variable, boolean) {
        (&Num::Var(self_var), _) => {
            match boolean {
                &Boolean::Constant(flag) => {
                    if flag {
                        let constr = accumulator + Term::from(self_var);
                        constr
                    } else {
                        accumulator
                    }
                }
                &Boolean::Is(bit) => {
                    let constr = (Term::from(self_var) * Term::from(bit)) + accumulator;
                    constr
                }
                &Boolean::Not(not_bit) => {
                    // a - a*bit + accumulator
                    let constr =
                        Term::from(self_var) * (Term::from(1) - Term::from(not_bit)) + accumulator;
                    constr
                }
            }
        }
        (&Num::Constant(variable), &Boolean::Is(bit)) => {
            let constr = Term::from_field(variable) * Term::from(bit) + accumulator;
            constr
        }
        (&Num::Constant(constant), &Boolean::Not(bit)) => {
            let constr =
                Term::from_field(constant) * (Term::from(1) - Term::from(bit)) + accumulator;
            constr
        }
        (&Num::Constant(constant), &Boolean::Constant(bit)) => {
            if bit {
                let constr = accumulator + Term::from_field(constant);
                constr
            } else {
                accumulator
            }
        }
    }
}

pub fn mask_by_boolean_into_accumulator_constraint_with_shift<F: PrimeField>(
    boolean: &Boolean,
    variable: &Num<F>,
    accumulator: Constraint<F>,
    shift: F,
) -> Constraint<F> {
    match (variable, boolean) {
        (&Num::Var(self_var), _) => {
            match boolean {
                &Boolean::Constant(flag) => {
                    if flag {
                        let constr = accumulator + Term::from((shift, self_var));
                        constr
                    } else {
                        accumulator
                    }
                }
                &Boolean::Is(bit) => {
                    let constr = (Term::from((shift, self_var)) * Term::from(bit)) + accumulator;
                    constr
                }
                &Boolean::Not(not_bit) => {
                    // a - a*bit + accumulator
                    let constr = Term::from((shift, self_var))
                        * (Term::from(1) - Term::from(not_bit))
                        + accumulator;
                    constr
                }
            }
        }
        (&Num::Constant(constant), &Boolean::Is(bit)) => {
            let mut constant = constant;
            constant.mul_assign(&shift);
            let constr = Term::from_field(constant) * Term::from(bit) + accumulator;
            constr
        }
        (&Num::Constant(constant), &Boolean::Not(bit)) => {
            let mut constant = constant;
            constant.mul_assign(&shift);
            let constr =
                Term::from_field(constant) * (Term::from(1) - Term::from(bit)) + accumulator;
            constr
        }
        (&Num::Constant(constant), &Boolean::Constant(bit)) => {
            let mut constant = constant;
            constant.mul_assign(&shift);
            if bit {
                let constr = accumulator + Term::from_field(constant);
                constr
            } else {
                accumulator
            }
        }
    }
}

/// returns 0 if condition == `false` and `a` if condition == `true`
pub fn mask_into_constraint<F: PrimeField>(a: &Num<F>, condition: &Boolean) -> Constraint<F> {
    match (a, condition) {
        (&Num::Constant(a), &Boolean::Constant(flag)) => {
            if flag {
                Constraint::from_field(a)
            } else {
                Constraint::from(0)
            }
        }
        (&Num::Var(var), &Boolean::Constant(flag)) => {
            if flag {
                Constraint::from(var)
            } else {
                Constraint::from(0)
            }
        }
        (&Num::Var(var), &Boolean::Is(bit)) => {
            let cnstr: Constraint<F> = { Term::from(var) * Term::from(bit) };
            cnstr
        }
        (&Num::Var(var), &Boolean::Not(bit)) => {
            let cnstr: Constraint<F> = { Term::from(var) * (Term::from(1) - Term::from(bit)) };
            cnstr
        }
        (&Num::Constant(a), &Boolean::Is(bit)) => {
            let cnstr: Constraint<F> = { Term::from_field(a) * Term::from(bit) };
            cnstr
        }
        (&Num::Constant(a), &Boolean::Not(bit)) => {
            let cnstr: Constraint<F> = { Term::from_field(a) * (Term::from(1) - Term::from(bit)) };
            cnstr
        }
    }
}

pub fn mask_linear_term_into_constraint<F: PrimeField>(
    a: &Constraint<F>,
    condition: &Boolean,
) -> Constraint<F> {
    assert!(a.degree() <= 1);
    let result = if a.degree() == 0 {
        let constant_value = a.as_constant();
        let mut result = Constraint::<F>::from(condition.get_terms());
        result.scale(constant_value);

        result
    } else {
        let term = condition.get_terms();
        a.clone() * term
    };
    assert!(result.degree() <= 2);

    result
}

pub fn mask_linear_term_by_boolean_into_accumulator_constraint<F: PrimeField>(
    boolean: &Boolean,
    input: &Constraint<F>,
    accumulator: Constraint<F>,
) -> Constraint<F> {
    accumulator + (input.clone() * boolean.get_terms())
}

#[track_caller]
fn collapse_max_quadratic_constraint_into_fixed<F: PrimeField, CS: Circuit<F>>(
    cs: &mut CS,
    constraint: Constraint<F>,
    result: Variable,
) {
    let (quadratic_terms, linear_terms, constant_term) = constraint.clone().split_max_quadratic();

    // split quadratic terms and linear terms into cases where coefficient is 1 or not
    let mut quadratic_trivial_additions = vec![];
    let mut quadratic_trivial_subtractions = vec![];
    let mut quadratic_nontrivial = vec![];
    for (c, a, b) in quadratic_terms.into_iter() {
        assert!(c != F::ZERO);
        if c == F::ONE {
            quadratic_trivial_additions.push((a, b));
        } else if c == F::MINUS_ONE {
            quadratic_trivial_subtractions.push((a, b));
        } else {
            quadratic_nontrivial.push((c, a, b));
        }
    }

    let mut linear_trivial_additions = vec![];
    let mut linear_trivial_subtractions = vec![];
    let mut linear_nontrivial = vec![];
    for (c, a) in linear_terms.into_iter() {
        assert!(c != F::ZERO);
        if c == F::ONE {
            linear_trivial_additions.push(a);
        } else if c == F::MINUS_ONE {
            linear_trivial_subtractions.push(a);
        } else {
            linear_nontrivial.push((c, a));
        }
    }

    let value_fn = move |placer: &mut CS::WitnessPlacer| {
        let mut value = <CS::WitnessPlacer as WitnessTypeSet<F>>::Field::constant(constant_term);

        for (a, b) in quadratic_trivial_additions.iter() {
            let a = placer.get_field(*a);
            let b = placer.get_field(*b);
            value.add_assign_product(&a, &b);
        }

        for (a, b) in quadratic_trivial_subtractions.iter() {
            let mut a = placer.get_field(*a);
            let b = placer.get_field(*b);
            a.mul_assign(&b);
            value.sub_assign(&a);
        }

        for (constant, a, b) in quadratic_nontrivial.iter() {
            let constant = <CS::WitnessPlacer as WitnessTypeSet<F>>::Field::constant(*constant);
            let mut a = placer.get_field(*a);
            let b = placer.get_field(*b);
            a.mul_assign(&constant);
            value.add_assign_product(&a, &b);
        }

        for a in linear_trivial_additions.iter() {
            let a = placer.get_field(*a);
            value.add_assign(&a);
        }

        for a in linear_trivial_subtractions.iter() {
            let a = placer.get_field(*a);
            value.sub_assign(&a);
        }

        for (constant, a) in linear_nontrivial.iter() {
            let constant = <CS::WitnessPlacer as WitnessTypeSet<F>>::Field::constant(*constant);
            let a = placer.get_field(*a);
            value.add_assign_product(&constant, &a);
        }

        placer.assign_field(result, &value);
    };

    cs.set_values(value_fn);
}

// #[track_caller]
// fn choose_value_fn<F: PrimeField>(
//     has_constant: bool,
//     num_quadtratic_terms: usize,
//     num_linear_terms: usize,
// ) -> fn(WitnessGenSource<'_, F>, WitnessGenDest<'_, F>, &[F], &TableDriver<F>, TableType) {
//     // if has_constant == false && num_quadtratic_terms == 6 && num_linear_terms == 0 {
//     //     panic!("debug")
//     // };

//     // TODO: make macro to iterate over tuples

//     if has_constant {
//         super_seq_macro::seq!(N in [0, 1] {
//             if num_quadtratic_terms == N {
//                 super_seq_macro::seq!(M in [0, 1, 2, 16] {
//                     if num_linear_terms == M {
//                         return collapse_max_quadratic_constraint_into_fixed_witness_eval_fn::<F, true, N, M>;
//                     }
//                 });

//                 panic!("with constant term: unsupported number of linear terms: {} for {} quadratic terms", num_linear_terms, num_quadtratic_terms);
//             }
//         });

//         panic!(
//             "with constant term: unsupported number of quadratic terms: {}",
//             num_quadtratic_terms
//         );
//     } else {
//         super_seq_macro::seq!(N in [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 13, 15, 16, 17, 19, 20, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63] {
//             if num_quadtratic_terms == N {
//                 super_seq_macro::seq!(M in [0, 1, 2, 3, 4, 13, 17, 22, 30] {
//                     if num_linear_terms == M {
//                         return collapse_max_quadratic_constraint_into_fixed_witness_eval_fn::<F, false, N, M>;
//                     }
//                 });

//                 panic!("without constant term: unsupported number of linear terms: {} for {} quadratic terms", num_linear_terms, num_quadtratic_terms);
//             }
//         });

//         panic!(
//             "without constant term: unsupported number of quadratic terms: {}",
//             num_quadtratic_terms
//         );
//     }
// }

// // #[track_caller]
// // fn choose_value_fn<F: PrimeField>(
// //     has_constant: bool,
// //     num_quadtratic_terms: usize,
// //     num_linear_terms: usize,
// // ) -> fn(WitnessGenSource<'_, F>, WitnessGenDest<'_, F>, &[F], &TableDriver<F>, TableType) {
// //     if has_constant == false && num_quadtratic_terms == 6 && num_linear_terms == 0 {
// //         panic!("debug")
// //     };
// //     if has_constant {
// //         seq_macro::seq!(N in 0..22 {
// //             if num_quadtratic_terms == N {
// //                 super_seq_macro::seq!(M in 0..32-N {
// //                     if num_linear_terms == M {
// //                         return true_fn_~N::collapse_max_quadratic_constraint_into_fixed_witness_eval_fn_~M::<F, true, N, M>;
// //                         // return collapse_max_quadratic_constraint_into_fixed_witness_eval_fn::<F, true, N, M>;
// //                     }
// //                 });

// //                 panic!("with constant term: unsupported number of linear terms: {} for {} quadratic terms", num_linear_terms, num_quadtratic_terms);
// //             }
// //         });

// //         panic!(
// //             "with constant term: unsupported number of quadratic terms: {}",
// //             num_quadtratic_terms
// //         );
// //     } else {
// //         seq_macro::seq!(N in 0..22 {
// //             if num_quadtratic_terms == N {
// //                 super_seq_macro::seq!(M in 0..32-N {
// //                     if num_linear_terms == M {
// //                         return false_fn_~N::collapse_max_quadratic_constraint_into_fixed_witness_eval_fn_~M::<F, false, N, M>;
// //                         // return collapse_max_quadratic_constraint_into_fixed_witness_eval_fn::<F, false, N, M>;
// //                     }
// //                 });

// //                 panic!("without constant term: unsupported number of linear terms: {} for {} quadratic terms", num_linear_terms, num_quadtratic_terms);
// //             }
// //         });

// //         panic!(
// //             "without constant term: unsupported number of quadratic terms: {}",
// //             num_quadtratic_terms
// //         );
// //     }
// // }

// #[track_caller]
// #[inline(always)]
// fn collapse_max_quadratic_constraint_into_fixed_witness_eval_fn<
//     F: PrimeField,
//     const HAS_CONSTANT_TERM: bool,
//     const NUM_QUADRATIC: usize,
//     const NUM_LINEAR: usize,
// >(
//     input: WitnessGenSource<'_, F>,
//     mut output: WitnessGenDest<'_, F>,
//     constants: &[F],
//     _table_driver: &TableDriver<F>,
//     _table_type: TableType,
// ) {
//     unsafe {
//         let mut result = if HAS_CONSTANT_TERM {
//             *constants.get_unchecked(0)
//         } else {
//             F::ZERO
//         };
//         for j in 0..NUM_QUADRATIC {
//             let a = input[2 * j];
//             let b = input[2 * j + 1];
//             let c = constants.get_unchecked((HAS_CONSTANT_TERM as usize) + j);
//             if c.is_one() {
//                 result.add_assign_product(&a, &b);
//             } else {
//                 let mut t = a;
//                 t.mul_assign(c);
//                 result.add_assign_product(&t, &b);
//             }
//             // let mut t = a;
//             // t.mul_assign(c);
//             // result.add_assign_product(&t, &b);
//         }
//         for j in 0..NUM_LINEAR {
//             let a = input[NUM_QUADRATIC * 2 + j];
//             let c = constants.get_unchecked((HAS_CONSTANT_TERM as usize) + NUM_QUADRATIC + j);
//             if c.is_one() {
//                 result.add_assign(&a);
//             } else {
//                 result.add_assign_product(&a, c);
//             }
//             // result.add_assign_product(&a, c);
//         }
//         output[0] = result;
//     }
// }

// // seq_macro::seq!(N in 0..22 {
// //     mod false_fn_~N {
// //         use super::*;

// //         super_seq_macro::seq!(M in 0..32-N {
// //             #[inline(always)]
// //             pub(crate) fn collapse_max_quadratic_constraint_into_fixed_witness_eval_fn_~M<
// //                 F: PrimeField,
// //                 const HAS_CONSTANT_TERM: bool,
// //                 const NUM_QUADRATIC: usize,
// //                 const NUM_LINEAR: usize
// //             >(
// //                 input: WitnessGenSource<'_, F>,
// //                 output: WitnessGenDest<'_, F>,
// //                 constants: &[F],
// //                 table_driver: &TableDriver<F>,
// //                 table_type: TableType
// //             ) {

// //                 collapse_max_quadratic_constraint_into_fixed_witness_eval_fn::<F, false, N, M>(input, output, constants, table_driver, table_type)
// //             }
// //         });
// //     }
// // });

// // seq_macro::seq!(N in 0..22 {
// //     mod true_fn_~N {
// //         use super::*;

// //         super_seq_macro::seq!(M in 0..32-N {
// //             #[inline(always)]
// //             pub(crate) fn collapse_max_quadratic_constraint_into_fixed_witness_eval_fn_~M<
// //                 F: PrimeField,
// //                 const HAS_CONSTANT_TERM: bool,
// //                 const NUM_QUADRATIC: usize,
// //                 const NUM_LINEAR: usize
// //             >(
// //                 input: WitnessGenSource<'_, F>,
// //                 output: WitnessGenDest<'_, F>,
// //                 constants: &[F],
// //                 table_driver: &TableDriver<F>,
// //                 table_type: TableType
// //             ) {
// //                 collapse_max_quadratic_constraint_into_fixed_witness_eval_fn::<F, true, N, M>(input, output, constants, table_driver, table_type)
// //             }
// //         });
// //     }
// // });

// #[track_caller]
// pub(crate) fn choose_register_word_selection_value_fn<F: PrimeField>(
//     num_words: usize,
//     num_byte_pairs: usize,
// ) -> fn(WitnessGenSource<'_, F>, WitnessGenDest<'_, F>, &[F], &TableDriver<F>, TableType) {
//     super_seq_macro::seq!(N in [0, 10, 11, 12, 13] {
//         if num_words == N {
//             super_seq_macro::seq!(M in [0, 1] {
//                 if num_byte_pairs == M {
//                     return register_word_selection_value_fn::<F, N, M>;
//                 }
//             });

//             panic!("unsupported number of byte pairs: {} for {} words", num_byte_pairs, num_words);
//         }
//     });

//     panic!("unsupported number of words: {}", num_words);
// }

// // TODO: make debug and non-debug mode cases
// #[track_caller]
// fn register_word_selection_value_fn<
//     F: PrimeField,
//     const NUM_WORDS: usize,
//     const NUM_BYTE_PAIRS: usize,
// >(
//     input: WitnessGenSource<'_, F>,
//     mut output: WitnessGenDest<'_, F>,
//     _constants: &[F],
//     _table_driver: &TableDriver<F>,
//     _table_type: TableType,
// ) {
//     for j in 0..NUM_WORDS {
//         let flag = input[2 * j];
//         if flag.as_boolean() {
//             let value = input[2 * j + 1];
//             debug_assert!(value.as_u64_reduced() < 1 << 16);
//             output[0] = value;
//             return;
//         }
//     }
//     for j in 0..NUM_BYTE_PAIRS {
//         let flag = input[2 * NUM_WORDS + 3 * j];
//         if flag.as_boolean() {
//             let low = input[2 * NUM_WORDS + 3 * j + 1].as_u64_reduced();
//             let high = input[2 * NUM_WORDS + 3 * j + 2].as_u64_reduced();
//             debug_assert!(low < 1 << 8);
//             debug_assert!(high < 1 << 8);
//             output[0] = F::from_u64_unchecked((high << 8) | low);
//             return;
//         }
//     }

//     output[0] = F::ZERO;
// }
