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
use crate::ntt::utils::STAGE_PLANS_N2B;

cuda_kernel!(
    OneStageKernel,
    one_stage_kernel,
    inputs_matrix: PtrAndStride<BaseField>,
    outputs_matrix: MutPtrAndStride<BaseField>,
    start_stage: u32,
    log_n: u32,
    blocks_per_ntt: u32,
    evals_are_coset: bool,
);

one_stage_kernel!(evals_to_Z_one_stage);

cuda_kernel!(
    MultiStageKernel,
    multi_stage_kernel,
    inputs_matrix: PtrAndStride<BaseField>,
    outputs_matrix: MutPtrAndStride<BaseField>,
    start_stage: u32,
    stages_this_launch: u32,
    log_n: u32,
    num_Z_cols: u32,
);

multi_stage_kernel!(evals_to_Z_nonfinal_7_or_8_stages_block);
multi_stage_kernel!(main_domain_evals_to_Z_final_7_stages_warp);
multi_stage_kernel!(main_domain_evals_to_Z_final_8_stages_warp);
multi_stage_kernel!(main_domain_evals_to_Z_final_9_to_12_stages_block);
multi_stage_kernel!(coset_evals_to_Z_final_7_stages_warp);
multi_stage_kernel!(coset_evals_to_Z_final_8_stages_warp);
multi_stage_kernel!(coset_evals_to_Z_final_9_to_12_stages_block);

#[allow(non_snake_case)]
#[allow(clippy::too_many_arguments)]
fn natural_evals_to_bitrev_Z(
    inputs_matrix: &(impl DeviceMatrixChunkImpl<BaseField> + ?Sized),
    outputs_matrix: &mut (impl DeviceMatrixChunkMutImpl<BaseField> + ?Sized),
    log_n: usize,
    num_real_cols: usize,
    evals_are_coset: bool,
    stream: &CudaStream,
) -> CudaResult<()> {
    assert!(log_n >= 1);
    assert!(log_n <= OMEGA_LOG_ORDER as usize);
    assert_eq!(num_real_cols % 2, 0);
    let n = 1 << log_n;
    let num_Z_cols = (num_real_cols / 2) as u32;
    assert_eq!(inputs_matrix.rows(), n);
    assert_eq!(inputs_matrix.cols(), num_real_cols);
    assert_eq!(outputs_matrix.rows(), n);
    assert_eq!(outputs_matrix.cols(), num_real_cols);
    let log_n = log_n as u32;
    let n = n as u32;

    let inputs_matrix = inputs_matrix.as_ptr_and_stride();
    let outputs_matrix_const = outputs_matrix.as_ptr_and_stride();
    let outputs_matrix_mut = outputs_matrix.as_mut_ptr_and_stride();

    // The log_n < 16 path isn't performant, and is meant to unblock
    // small proofs for debugging purposes only.
    if log_n < 16 {
        let threads = 128;
        let blocks_per_ntt = (n + 2 * threads - 1) / (2 * threads);
        let blocks = blocks_per_ntt * num_Z_cols;
        let config = CudaLaunchConfig::basic(blocks, threads, stream);
        let kernel_function = OneStageKernelFunction(evals_to_Z_one_stage);
        let args = OneStageKernelArguments::new(
            inputs_matrix,
            outputs_matrix_mut,
            0,
            log_n,
            blocks_per_ntt,
            evals_are_coset,
        );
        kernel_function.launch(&config, &args)?;
        for stage in 1..log_n {
            let args = OneStageKernelArguments::new(
                outputs_matrix_const,
                outputs_matrix_mut,
                stage,
                log_n,
                blocks_per_ntt,
                evals_are_coset,
            );
            kernel_function.launch(&config, &args)?;
        }
        return Ok(());
    }

    use crate::ntt::utils::COMPLEX_COLS_PER_BLOCK;
    use crate::ntt::utils::N2B_LAUNCH::*;
    let plan = &STAGE_PLANS_N2B[log_n as usize - 16];
    let (kern, stages_this_launch) = plan[0].expect("plan must contain at least 1 kernel");
    assert_eq!(kern, NONFINAL_7_OR_8_BLOCK);
    let num_chunks = (num_Z_cols + COMPLEX_COLS_PER_BLOCK - 1) / COMPLEX_COLS_PER_BLOCK;
    let grid_dim_x = n / 4096;
    let block_dim_x = 512;
    let config = CudaLaunchConfig::basic((grid_dim_x, num_chunks), block_dim_x, stream);
    let args = MultiStageKernelArguments::new(
        inputs_matrix,
        outputs_matrix_mut,
        0,
        stages_this_launch,
        log_n,
        num_Z_cols,
    );
    MultiStageKernelFunction(evals_to_Z_nonfinal_7_or_8_stages_block).launch(&config, &args)?;
    let mut stage = stages_this_launch;
    for &kernel in &plan[1..] {
        let start_stage = stage;
        let num_chunks = (num_Z_cols + COMPLEX_COLS_PER_BLOCK - 1) / COMPLEX_COLS_PER_BLOCK;
        if let Some((kern, stages_this_launch)) = kernel {
            stage += stages_this_launch;
            let (function, grid_dim_x, block_dim_x): (MultiStageKernelSignature, u32, u32) =
                match kern {
                    FINAL_7_WARP => (
                        if evals_are_coset {
                            coset_evals_to_Z_final_7_stages_warp
                        } else {
                            main_domain_evals_to_Z_final_7_stages_warp
                        },
                        n / (4 * 128),
                        128,
                    ),
                    FINAL_8_WARP => (
                        if evals_are_coset {
                            coset_evals_to_Z_final_8_stages_warp
                        } else {
                            main_domain_evals_to_Z_final_8_stages_warp
                        },
                        n / (4 * 256),
                        128,
                    ),
                    FINAL_9_TO_12_BLOCK => (
                        if evals_are_coset {
                            coset_evals_to_Z_final_9_to_12_stages_block
                        } else {
                            main_domain_evals_to_Z_final_9_to_12_stages_block
                        },
                        n / 4096,
                        512,
                    ),
                    NONFINAL_7_OR_8_BLOCK => {
                        (evals_to_Z_nonfinal_7_or_8_stages_block, n / 4096, 512)
                    }
                };
            let config = CudaLaunchConfig::basic((grid_dim_x, num_chunks), block_dim_x, stream);
            let args = MultiStageKernelArguments::new(
                outputs_matrix_const,
                outputs_matrix_mut,
                start_stage,
                stages_this_launch,
                log_n,
                num_Z_cols,
            );
            MultiStageKernelFunction(function).launch(&config, &args)
        } else {
            get_last_error().wrap()
        }?;
    }
    assert_eq!(stage, log_n);
    get_last_error().wrap()
}

#[allow(non_snake_case)]
#[allow(clippy::too_many_arguments)]
pub fn natural_trace_main_domain_evals_to_bitrev_Z(
    inputs_matrix: &(impl DeviceMatrixChunkImpl<BaseField> + ?Sized),
    outputs_matrix: &mut (impl DeviceMatrixChunkMutImpl<BaseField> + ?Sized),
    log_n: usize,
    num_real_cols: usize,
    stream: &CudaStream,
) -> CudaResult<()> {
    natural_evals_to_bitrev_Z(
        inputs_matrix,
        outputs_matrix,
        log_n,
        num_real_cols,
        false,
        stream,
    )
}

#[allow(non_snake_case)]
#[allow(clippy::too_many_arguments)]
pub fn natural_composition_coset_evals_to_bitrev_Z(
    inputs_matrix: &(impl DeviceMatrixChunkImpl<BaseField> + ?Sized),
    outputs_matrix: &mut (impl DeviceMatrixChunkMutImpl<BaseField> + ?Sized),
    log_n: usize,
    num_real_cols: usize,
    stream: &CudaStream,
) -> CudaResult<()> {
    natural_evals_to_bitrev_Z(
        inputs_matrix,
        outputs_matrix,
        log_n,
        num_real_cols,
        true,
        stream,
    )
}
