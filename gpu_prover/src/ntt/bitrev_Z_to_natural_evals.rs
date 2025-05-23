use era_cudart::cuda_kernel;
use era_cudart::error::get_last_error;
use era_cudart::execution::{CudaLaunchConfig, KernelFunction};
use era_cudart::result::{CudaResult, CudaResultWrap};
use era_cudart::stream::CudaStream;

use crate::context::OMEGA_LOG_ORDER;
use crate::device_structures::{
    DeviceMatrixChunkImpl, DeviceMatrixChunkMutImpl, MutPtrAndStride, PtrAndStride,
};
use crate::field::BaseField;
use crate::ntt::utils::STAGE_PLANS_B2N;
use crate::utils::GetChunksCount;

cuda_kernel!(
    OneStage,
    one_stage_kernel,
    inputs_matrix: PtrAndStride<BaseField>,
    outputs_matrix: MutPtrAndStride<BaseField>,
    start_stage: u32,
    log_n: u32,
    blocks_per_ntt: u32,
    log_extension_degree: u32,
    coset_idx: u32,
);

one_stage_kernel!(bitrev_Z_to_natural_coset_evals_1_stage);

// "v" indicates a vectorized layout of BaseField columns,
// For the final output, columns represent distinct base field values.
// For intermediate outputs, each pair of columns represents the c0s and c1s
// of a single column of complex values.
cuda_kernel!(
    MultiStage,
    multi_stage_kernel,
    inputs_matrix: PtrAndStride<BaseField>,
    outputs_matrix: MutPtrAndStride<BaseField>,
    start_stage: u32,
    stages_this_launch: u32,
    log_n: u32,
    num_Z_cols: u32,
    log_extension_degree: u32,
    coset_idx: u32,
);

multi_stage_kernel!(bitrev_Z_to_natural_coset_evals_noninitial_7_or_8_stages_block);
multi_stage_kernel!(bitrev_Z_to_natural_coset_evals_initial_7_stages_warp);
multi_stage_kernel!(bitrev_Z_to_natural_coset_evals_initial_8_stages_warp);
multi_stage_kernel!(bitrev_Z_to_natural_coset_evals_initial_9_to_12_stages_block);

#[allow(non_snake_case)]
#[allow(clippy::too_many_arguments)]
fn bitrev_Z_to_natural_evals(
    inputs_matrix: &(impl DeviceMatrixChunkImpl<BaseField> + ?Sized),
    outputs_matrix: &mut (impl DeviceMatrixChunkMutImpl<BaseField> + ?Sized),
    log_n: usize,
    num_bf_cols: usize,
    log_extension_degree: usize,
    coset_idx: usize,
    stream: &CudaStream,
) -> CudaResult<()> {
    assert!(log_n >= 1);
    assert!(log_n <= OMEGA_LOG_ORDER as usize);
    assert_eq!(num_bf_cols % 2, 0);
    let n = 1 << log_n;
    let num_Z_cols = (num_bf_cols / 2) as u32;
    assert_eq!(inputs_matrix.rows(), n);
    assert_eq!(inputs_matrix.cols(), num_bf_cols);
    assert_eq!(outputs_matrix.rows(), n);
    assert_eq!(outputs_matrix.cols(), num_bf_cols);
    let log_n = log_n as u32;
    let n = n as u32;

    let inputs_matrix = inputs_matrix.as_ptr_and_stride();
    let outputs_matrix_const = outputs_matrix.as_ptr_and_stride();
    let outputs_matrix_mut = outputs_matrix.as_mut_ptr_and_stride();

    // The following bound is overly conservative, since technically the GPU-side
    // 3-layer power caches support powers as fine-grained as CIRCLE_GROUP_LOG_ORDER.
    // Therefore, the assert may fire for some sizes/LDE degrees that could technically work,
    // but are bigger than we expect. Its purpose is to remind us to revisit the logic
    // in such unexpected cases (and relax the bound if the new cases are legitimate).
    assert!(log_n + (log_extension_degree as u32) < OMEGA_LOG_ORDER);

    // The log_n < 16 path isn't performant, and is meant to unblock
    // small proofs for debugging purposes only.
    if log_n < 16 {
        let threads: u32 = 128;
        let n: u32 = 1 << log_n;
        let blocks_per_ntt: u32 = n.get_chunks_count(2 * threads);
        let blocks = blocks_per_ntt * num_Z_cols;
        let config = CudaLaunchConfig::basic(blocks, threads, stream);
        let kernel_function = OneStageFunction(bitrev_Z_to_natural_coset_evals_1_stage);
        let args = OneStageArguments::new(
            inputs_matrix,
            outputs_matrix_mut,
            0,
            log_n,
            blocks_per_ntt,
            log_extension_degree as u32,
            coset_idx as u32,
        );
        kernel_function.launch(&config, &args)?;
        for stage in 1..log_n {
            let args = OneStageArguments::new(
                outputs_matrix_const,
                outputs_matrix_mut,
                stage,
                log_n,
                blocks_per_ntt,
                log_extension_degree as u32,
                coset_idx as u32,
            );
            kernel_function.launch(&config, &args)?;
        }
        return Ok(());
    }

    use crate::ntt::utils::B2N_LAUNCH::*;
    use crate::ntt::utils::COMPLEX_COLS_PER_BLOCK;
    let plan = &STAGE_PLANS_B2N[log_n as usize - 16];
    let mut stage: u32 = 0;
    for &kernel in &plan[..] {
        let start_stage = stage;
        let num_chunks = num_Z_cols.get_chunks_count(COMPLEX_COLS_PER_BLOCK);
        if let Some((kern, stages_this_launch)) = kernel {
            stage += stages_this_launch;
            let (function, grid_dim_x, block_dim_x): (MultiStageSignature, u32, u32) = match kern {
                INITIAL_7_WARP => (
                    bitrev_Z_to_natural_coset_evals_initial_7_stages_warp,
                    n / (4 * 128),
                    128,
                ),
                INITIAL_8_WARP => (
                    bitrev_Z_to_natural_coset_evals_initial_8_stages_warp,
                    n / (4 * 256),
                    128,
                ),
                INITIAL_9_TO_12_BLOCK => (
                    bitrev_Z_to_natural_coset_evals_initial_9_to_12_stages_block,
                    n / 4096,
                    512,
                ),
                NONINITIAL_7_OR_8_BLOCK => (
                    bitrev_Z_to_natural_coset_evals_noninitial_7_or_8_stages_block,
                    n / 4096,
                    512,
                ),
            };
            let inputs = if start_stage == 0 {
                inputs_matrix
            } else {
                outputs_matrix_const
            };
            let config = CudaLaunchConfig::basic((grid_dim_x, num_chunks), block_dim_x, stream);
            let args = MultiStageArguments::new(
                inputs,
                outputs_matrix_mut,
                start_stage,
                stages_this_launch,
                log_n,
                num_Z_cols,
                log_extension_degree as u32,
                coset_idx as u32,
            );
            MultiStageFunction(function).launch(&config, &args)
        } else {
            get_last_error().wrap()
        }?;
    }
    assert_eq!(stage, log_n);
    get_last_error().wrap()
}

#[allow(non_snake_case)]
#[allow(clippy::too_many_arguments)]
pub fn bitrev_Z_to_natural_trace_coset_evals(
    inputs_matrix: &(impl DeviceMatrixChunkImpl<BaseField> + ?Sized),
    outputs_matrix: &mut (impl DeviceMatrixChunkMutImpl<BaseField> + ?Sized),
    log_n: usize,
    num_bf_cols: usize,
    stream: &CudaStream,
) -> CudaResult<()> {
    bitrev_Z_to_natural_evals(
        inputs_matrix,
        outputs_matrix,
        log_n,
        num_bf_cols,
        1,
        1,
        stream,
    )
}

#[allow(non_snake_case)]
#[allow(clippy::too_many_arguments)]
pub fn bitrev_Z_to_natural_composition_main_domain_evals(
    inputs_matrix: &(impl DeviceMatrixChunkImpl<BaseField> + ?Sized),
    outputs_matrix: &mut (impl DeviceMatrixChunkMutImpl<BaseField> + ?Sized),
    log_n: usize,
    num_bf_cols: usize,
    stream: &CudaStream,
) -> CudaResult<()> {
    bitrev_Z_to_natural_evals(
        inputs_matrix,
        outputs_matrix,
        log_n,
        num_bf_cols,
        1,
        0,
        stream,
    )
}
