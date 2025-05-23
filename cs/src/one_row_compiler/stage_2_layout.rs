use super::*;

impl quote::ToTokens for OptimizedOraclesForLookupWidth1 {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let Self {
            num_pairs,
            base_field_oracles,
            ext_4_field_oracles,
        } = *self;

        use quote::quote;

        let stream = quote! {
            OptimizedOraclesForLookupWidth1 {
                num_pairs: #num_pairs,
                base_field_oracles: #base_field_oracles,
                ext_4_field_oracles: #ext_4_field_oracles,
            }
        };

        tokens.extend(stream);
    }
}

#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct LookupWidth1SourceDestInformation {
    pub a_col: usize,
    pub b_col: usize,
    pub base_field_quadratic_oracle_col: usize,
    pub ext4_field_inverses_columns_start: usize,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct LookupWidth1SourceDestInformationForExpressions<F: PrimeField> {
    pub a_expr: LookupExpression<F>,
    pub b_expr: LookupExpression<F>,
    pub base_field_quadratic_oracle_col: usize,
    pub ext4_field_inverses_columns_start: usize,
}
