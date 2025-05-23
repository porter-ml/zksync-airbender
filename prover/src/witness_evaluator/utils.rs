use super::*;

#[inline(always)]
pub(crate) fn lookup_index_into_encoding_tuple(
    lookup_row: usize,
    lookup_encoding_capacity: usize,
) -> (u32, u32) {
    let column = lookup_row / lookup_encoding_capacity;
    let row = lookup_row % lookup_encoding_capacity;

    (column as u32, row as u32)
}

#[inline(always)]
pub(crate) fn encoding_tuple_into_lookup_index(
    column: u32,
    row: u32,
    lookup_encoding_capacity: usize,
) -> usize {
    let offset = (column as usize) * lookup_encoding_capacity;
    offset + (row as usize)
}

#[inline(always)]
pub(crate) fn write_boolean_placeholder_into_memory_columns<O: Oracle<Mersenne31Field>>(
    placeholder_columns: ColumnSet<1>,
    placeholder_type: Placeholder,
    oracle: &O,
    memory_columns_view: &mut [Mersenne31Field],
    trace_step: usize,
) {
    let offset = placeholder_columns.start();

    let low = Oracle::<Mersenne31Field>::get_boolean_witness_from_placeholder(
        oracle,
        placeholder_type,
        trace_step,
    );

    debug_assert!(offset < memory_columns_view.len());
    unsafe {
        *memory_columns_view.get_unchecked_mut(offset) = Mersenne31Field(low as u32);
    }
}

#[inline(always)]
pub(crate) fn write_u16_placeholder_into_memory_columns<O: Oracle<Mersenne31Field>>(
    placeholder_columns: ColumnSet<1>,
    placeholder_type: Placeholder,
    oracle: &O,
    memory_columns_view: &mut [Mersenne31Field],
    trace_step: usize,
) {
    let offset = placeholder_columns.start();

    let low = Oracle::<Mersenne31Field>::get_u16_witness_from_placeholder(
        oracle,
        placeholder_type,
        trace_step,
    );

    debug_assert!(offset < memory_columns_view.len());
    unsafe {
        *memory_columns_view.get_unchecked_mut(offset) = Mersenne31Field(low as u32);
    }
}

#[inline(always)]
pub(crate) fn write_u32_placeholder_into_memory_columns<O: Oracle<Mersenne31Field>>(
    placeholder_columns: ColumnSet<2>,
    placeholder_type: Placeholder,
    oracle: &O,
    memory_columns_view: &mut [Mersenne31Field],
    trace_step: usize,
) {
    let offset_low = placeholder_columns.start();
    let offset_high = offset_low + 1;

    let value = Oracle::<Mersenne31Field>::get_u32_witness_from_placeholder(
        oracle,
        placeholder_type,
        trace_step,
    );
    debug_assert!(offset_low < memory_columns_view.len());
    debug_assert!(offset_high < memory_columns_view.len());

    unsafe {
        *memory_columns_view.get_unchecked_mut(offset_low) = Mersenne31Field(value & 0xffff);
        *memory_columns_view.get_unchecked_mut(offset_high) = Mersenne31Field(value >> 16);
    }
}

#[inline(always)]
pub(crate) fn write_timestamp_placeholder_into_memory_columns<O: Oracle<Mersenne31Field>>(
    placeholder_columns: ColumnSet<2>,
    placeholder_type: Placeholder,
    oracle: &O,
    memory_columns_view: &mut [Mersenne31Field],
    trace_step: usize,
) {
    let offset_low = placeholder_columns.start();
    let offset_high = offset_low + 1;

    let value = Oracle::<Mersenne31Field>::get_timestamp_witness_from_placeholder(
        oracle,
        placeholder_type,
        trace_step,
    );
    debug_assert!(offset_low < memory_columns_view.len());
    debug_assert!(offset_high < memory_columns_view.len());

    let [low, high] = timestamp_scalar_into_column_values(value);

    unsafe {
        *memory_columns_view.get_unchecked_mut(offset_low) = Mersenne31Field(low);
        *memory_columns_view.get_unchecked_mut(offset_high) = Mersenne31Field(high);
    }
}

#[inline(always)]
pub(crate) fn write_u32_value_into_memory_columns(
    columns: ColumnSet<2>,
    value: u32,
    memory_columns_view: &mut [Mersenne31Field],
) {
    let offset_low = columns.start();
    let offset_high = offset_low + 1;

    debug_assert!(offset_low < memory_columns_view.len());
    debug_assert!(offset_high < memory_columns_view.len());
    unsafe {
        *memory_columns_view.get_unchecked_mut(offset_low) = Mersenne31Field(value & 0xffff);
        *memory_columns_view.get_unchecked_mut(offset_high) = Mersenne31Field(value >> 16);
    }
}

#[inline(always)]
pub(crate) fn write_timestamp_value_into_memory_columns(
    columns: ColumnSet<2>,
    value: TimestampScalar,
    memory_columns_view: &mut [Mersenne31Field],
) {
    let offset_low = columns.start();
    let offset_high = offset_low + 1;

    debug_assert!(offset_low < memory_columns_view.len());
    debug_assert!(offset_high < memory_columns_view.len());

    let [low, high] = timestamp_scalar_into_column_values(value);
    unsafe {
        *memory_columns_view.get_unchecked_mut(offset_low) = Mersenne31Field(low);
        *memory_columns_view.get_unchecked_mut(offset_high) = Mersenne31Field(high);
    }
}
