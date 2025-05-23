use super::*;

#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct SetupLayout {
    pub timestamp_setup_columns: ColumnSet<NUM_TIMESTAMP_COLUMNS_FOR_RAM>,
    pub range_check_16_setup_column: ColumnSet<1>,
    pub timestamp_range_check_setup_column: ColumnSet<1>,
    pub generic_lookup_setup_columns: ColumnSet<NUM_COLUMNS_FOR_COMMON_TABLE_WIDTH_SETUP>,
    pub total_width: usize,
}

impl SetupLayout {
    pub fn layout_for_lookup_size(
        lookups_total_table_len: usize,
        trace_len: usize,
        need_shuffle_ram_timestamps: bool,
    ) -> Self {
        assert!(trace_len.is_power_of_two());
        let encoding_capacity = trace_len - 1;
        let mut num_required_setup_tuples = lookups_total_table_len / encoding_capacity;
        if lookups_total_table_len % encoding_capacity != 0 {
            num_required_setup_tuples += 1;
        }
        let mut offset = 0;
        let timestamp_setup_columns = if need_shuffle_ram_timestamps {
            ColumnSet::layout_at(&mut offset, 1)
        } else {
            ColumnSet::empty()
        };

        let range_check_16_setup_column = ColumnSet::layout_at(&mut offset, 1);
        let timestamp_range_check_setup_column = ColumnSet::layout_at(&mut offset, 1);
        let generic_lookup_setup_columns =
            ColumnSet::layout_at(&mut offset, num_required_setup_tuples);
        let total_width = offset;

        Self {
            timestamp_setup_columns,
            range_check_16_setup_column,
            timestamp_range_check_setup_column,
            generic_lookup_setup_columns,
            total_width,
        }
    }
}
