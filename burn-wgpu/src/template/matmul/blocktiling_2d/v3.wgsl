@group(0)
@binding(0)
var<storage, read> lhs: array<{{ elem }}>;

@group(0)
@binding(1)
var<storage, read> rhs: array<{{ elem }}>;

@group(0)
@binding(2)
var<storage, read_write> output: array<{{ elem }}>;

@group(0)
@binding(3)
var<storage, read> info: array<u32>;

const B_M = {{b_m}}u;
const B_N = {{b_n}}u;
const B_K = {{b_k}}u;
const B_M_X_B_K = {{bm_x_bk}}u;
const B_K_X_B_N = {{bk_x_bn}}u;
const T_M = {{t_m}}u;
const T_N = {{t_n}}u;
const T_M_X_T_N = {{tm_x_tn}}u;

var<workgroup> shared_lhs: array<{{ elem }}, B_M_X_B_K>; 
var<workgroup> shared_rhs: array<{{ elem }}, B_K_X_B_N>;

@compute
@workgroup_size({{ workgroup_size_x }}, {{ workgroup_size_y }}, {{ workgroup_size_z }})
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let skip_row = workgroup_id.x * B_M;
    let skip_col = workgroup_id.y * B_N;

    let n_thread_per_row = ((B_N - 1u) / T_N) + 1u;
    let thread_row = (local_idx / n_thread_per_row) * T_M;
    let thread_col = (local_idx % n_thread_per_row) * T_N;
    
    let row = skip_row + thread_row;
    let col = skip_col + thread_col;

    let batch = global_id.z;

    // Basic information
    let dim = info[0];
    let n_rows = info[6u * dim - 1u]; 
    let n_cols = info[6u * dim]; 
    let K = info[5u * dim - 1u];

    // Calculate the corresponding offsets with support for broadcasting.
    let offset_output = batch * n_rows * n_cols; 
    var offset_lhs: u32 = skip_row * K; 
    var offset_rhs: u32 = skip_col;

    let batch_dims = dim - 2u;
    for (var b: u32 = 1u; b <= batch_dims; b++) {
        let stride_lhs = info[b];
        let stride_rhs = info[b + dim];
        let stride_output = info[b + 2u * dim];
        let shape_lhs = info[b + 3u * dim];
        let shape_rhs = info[b + 4u * dim];

        offset_lhs += offset_output / stride_output % shape_lhs * stride_lhs;
        offset_rhs += offset_output / stride_output % shape_rhs * stride_rhs;
    }
    
    var results: array<{{ elem }}, T_M_X_T_N>;
    var register_M: array<{{ elem }}, T_M>;
    var register_N: array<{{ elem }}, T_N>;

    let thread_offset = local_idx * T_M_X_T_N;

    for (var k = 0u; k < K; k += B_K) {
        // sm_limit ensures that although there are up to B_M x B_N writes to memory, 
        // shared memories remain B_M x B_K (lhs) or B_K x B_N (rhs)
        // also ensures we do not read out of matrices if M % B_M != 0 or N % B_N != 0
        let sm_limit = min(B_K, K - k);

        // Load data into shared memories
        // Each thread is responsible of loading T_M x T_N values from both lhs and rhs
        for (var load_index = 0u; load_index < T_M_X_T_N; load_index ++) {
            let lhs_sm_position = thread_offset + load_index;
            let block_row = lhs_sm_position / B_K;
            let block_col = lhs_sm_position % B_K;
            let lhs_position = offset_lhs + k + block_row * K + block_col;
            shared_lhs[lhs_sm_position] = lhs[lhs_position];
        }

        for (var load_index = 0u; load_index < T_M_X_T_N; load_index ++) {
            let rhs_sm_position = thread_offset + load_index;
            let block_row = rhs_sm_position / B_N;
            let block_col = rhs_sm_position % B_N;
            let rhs_position = offset_rhs + (k + block_row) * n_cols + block_col;
            shared_rhs[rhs_sm_position] = rhs[rhs_position];
        } 


        workgroupBarrier();

        // Compute intermediate results
        // Results are cumulated in results array and updated at each block
        // Outer loop indicates which subcolumns/subrows to read from shared memories
        for (var dot_index = 0u; dot_index < sm_limit; dot_index++) {
            // Load a subcolumn of values from lhs
            for (var tile_index = 0u; tile_index < T_M; tile_index++) {
                let lhs_sm_position = (thread_row + tile_index) * B_K + dot_index;
                register_M[tile_index] = shared_lhs[lhs_sm_position];
            }
            // Load a subrow of values from rhs
            for (var tile_index = 0u; tile_index < T_N; tile_index++) {
                let rhs_sm_position = thread_col + tile_index + dot_index * B_N;
                register_N[tile_index] = shared_rhs[rhs_sm_position];
            }
            // Multiply subcolumn and subrow and store results
            for (var res_idx_M = 0u; res_idx_M < T_M; res_idx_M++) {
                for (var res_idx_N = 0u; res_idx_N < T_N; res_idx_N++) {
                    results[res_idx_M * T_N + res_idx_N] += register_M[res_idx_M] * register_N[res_idx_N];
                }
            }
        }
        
        workgroupBarrier();
    }

    // Write output matrix
    // Each thread is responsible of writing T_M x T_N results
    for (var res_idx_M = 0u; res_idx_M < T_M; res_idx_M++) {
        for (var res_idx_N = 0u; res_idx_N < T_N; res_idx_N++) {
            let current_row = row + res_idx_M;
            let current_col = col + res_idx_N;
            // Check that we are within the bounds of output matrix
            if current_row < n_rows && current_col < n_cols { 
                let result_position = res_idx_M * T_N + res_idx_N;
                let output_position = offset_output + current_row * n_cols + current_col;
                output[output_position] = results[result_position];
            }
        }
    }
}
