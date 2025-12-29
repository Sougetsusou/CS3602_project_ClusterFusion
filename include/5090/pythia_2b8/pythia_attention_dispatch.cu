/**
 * ClusterFusion Pythia-2.8B Attention-Only Kernel Dispatch
 * 
 * This file dispatches the Attention + MLP Up + GELU kernel.
 * The MLP Down projection is assumed to be done by PyTorch.
 * 
 * Output: 
 *   - attn_output: attention output for residual [1, hidden_dim]
 *   - mlp_intermediate: MLP Up + GELU output [ffn_dim] for PyTorch MLP Down
 *   - k_new, v_new: new KV for cache
 */

#include "kernel_attention.cuh"
#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> pythia_2b8_attention_only_sm120(
    torch::Tensor input,              // [1, 1, hidden_dim]
    torch::Tensor weight_qkv,         // [hidden_dim, 3*hidden_dim]
    torch::Tensor bias_qkv,
    torch::Tensor weight_o,           // [hidden_dim, hidden_dim]
    torch::Tensor bias_o,
    torch::Tensor k_cache,            // [max_seq_len, hidden_dim]
    torch::Tensor v_cache,            // [max_seq_len, hidden_dim]
    torch::Tensor layernorm_weight,
    torch::Tensor layernorm_bias,
    torch::Tensor cos,
    torch::Tensor sin,
    torch::Tensor post_ln_weight,
    torch::Tensor post_ln_bias,
    torch::Tensor mlp_up_weight,      // [hidden_dim, ffn_dim]
    torch::Tensor mlp_up_bias,
    int64_t current_seq_len
) 
{
    cudaFuncSetAttribute(PythiaAttentionMlpUpKernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    
    uint32_t max_shmem_size = 128 * sizeof(char) + (2 * TMA_LOAD_ONCE * MAX_SMEM_DIM + DIM_PER_BLOCK + 3 * HEAD_DIM) * sizeof(half) + DIM_BLOCK_REDUCE * sizeof(float);
    cudaFuncSetAttribute(PythiaAttentionMlpUpKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shmem_size);
    
    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA, 0);
    torch::Tensor attn_output = torch::zeros({1, HIDDEN_DIM}, options);
    torch::Tensor k = torch::zeros({1, HEAD_NUM, HEAD_DIM}, options);
    torch::Tensor v = torch::zeros({1, HEAD_NUM, HEAD_DIM}, options);
    torch::Tensor mlp_intermediate = torch::empty({FFN_DIM}, options);
    torch::Tensor post_ln_buffer = torch::empty({HIDDEN_DIM}, options);
    
    // Get pointers
    half* attn_output_ptr = reinterpret_cast<half*>(attn_output.data_ptr<at::Half>());
    half* k_ptr = reinterpret_cast<half*>(k.data_ptr<at::Half>());
    half* v_ptr = reinterpret_cast<half*>(v.data_ptr<at::Half>());
    half* mlp_intermediate_ptr = reinterpret_cast<half*>(mlp_intermediate.data_ptr<at::Half>());
    half* post_ln_buffer_ptr = reinterpret_cast<half*>(post_ln_buffer.data_ptr<at::Half>());

    half* input_ptr = reinterpret_cast<half*>(input.data_ptr<at::Half>());
    half* weight_qkv_ptr = reinterpret_cast<half*>(weight_qkv.data_ptr<at::Half>());
    half* bias_qkv_ptr = reinterpret_cast<half*>(bias_qkv.data_ptr<at::Half>());
    half* weight_o_ptr = reinterpret_cast<half*>(weight_o.data_ptr<at::Half>());
    half* bias_o_ptr = reinterpret_cast<half*>(bias_o.data_ptr<at::Half>());
    half* k_cache_ptr = reinterpret_cast<half*>(k_cache.data_ptr<at::Half>());
    half* v_cache_ptr = reinterpret_cast<half*>(v_cache.data_ptr<at::Half>());
    half* layernorm_weight_ptr = reinterpret_cast<half*>(layernorm_weight.data_ptr<at::Half>());
    half* layernorm_bias_ptr = reinterpret_cast<half*>(layernorm_bias.data_ptr<at::Half>());
    float* cos_ptr = reinterpret_cast<float*>(cos.data_ptr<float>());
    float* sin_ptr = reinterpret_cast<float*>(sin.data_ptr<float>());
    half* post_ln_weight_ptr = reinterpret_cast<half*>(post_ln_weight.data_ptr<at::Half>());
    half* post_ln_bias_ptr = reinterpret_cast<half*>(post_ln_bias.data_ptr<at::Half>());
    half* mlp_up_weight_ptr = reinterpret_cast<half*>(mlp_up_weight.data_ptr<at::Half>());
    half* mlp_up_bias_ptr = reinterpret_cast<half*>(mlp_up_bias.data_ptr<at::Half>());

    const uint32_t SEQ_LEN = static_cast<uint32_t>(current_seq_len);
    const uint32_t seq_len = SEQ_LEN;
    const uint32_t KV_DIM_PER_BLOCK = ((seq_len + CLUSTER_SIZE - 1) / CLUSTER_SIZE + (TMA_LOAD_ONCE_ATTN - 1)) & ~(TMA_LOAD_ONCE_ATTN - 1);
    
    // Create TensorMaps
    constexpr uint32_t rank = 2;
    
    // QKV weight tensor map
    CUtensorMap tensor_map_weight{};
    uint64_t size[rank] = {HIDDEN_DIM, 3 * HEAD_NUM * HEAD_DIM};
    uint64_t stride[rank - 1] = {HIDDEN_DIM * sizeof(half)};
    uint32_t box_size[rank] = {TMA_LOAD_ONCE, HEAD_DIM};
    uint32_t elem_stride[rank] = {1, 1};
    cuTensorMapEncodeTiled(&tensor_map_weight, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, rank, weight_qkv_ptr, 
        size, stride, box_size, elem_stride, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    // K cache tensor map
    CUtensorMap tensor_map_k_cache{};
    uint64_t size_k_cache[rank] = {HIDDEN_DIM, SEQ_LEN};
    uint64_t stride_k_cache[rank - 1] = {HIDDEN_DIM * sizeof(half)};
    uint32_t box_size_k_cache[rank] = {HEAD_DIM, TMA_LOAD_ONCE / 2};
    uint32_t elem_stride_k_cache[rank] = {1, 1};
    cuTensorMapEncodeTiled(&tensor_map_k_cache, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, rank, k_cache_ptr,
        size_k_cache, stride_k_cache, box_size_k_cache, elem_stride_k_cache, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    // V cache tensor map
    CUtensorMap tensor_map_v_cache{};
    uint64_t size_v_cache[rank] = {HIDDEN_DIM, SEQ_LEN};
    uint64_t stride_v_cache[rank - 1] = {HIDDEN_DIM * sizeof(half)};
    uint32_t box_size_v_cache[rank] = {HEAD_DIM, TMA_LOAD_ONCE / 2};
    uint32_t elem_stride_v_cache[rank] = {1, 1};
    cuTensorMapEncodeTiled(&tensor_map_v_cache, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, rank, v_cache_ptr,
        size_v_cache, stride_v_cache, box_size_v_cache, elem_stride_v_cache, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    // Output projection weight tensor map
    CUtensorMap tensor_map_weight_o{};
    uint64_t size_weight_o[rank] = {HIDDEN_DIM, HIDDEN_DIM};
    uint64_t stride_weight_o[rank - 1] = {HIDDEN_DIM * sizeof(half)};
    uint32_t box_size_weight_o[rank] = {HEAD_DIM, TMA_LOAD_ONCE};
    uint32_t elem_stride_weight_o[rank] = {1, 1};
    cuTensorMapEncodeTiled(&tensor_map_weight_o, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, rank, weight_o_ptr,
        size_weight_o, stride_weight_o, box_size_weight_o, elem_stride_weight_o, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    // MLP up weight tensor map
    CUtensorMap tensor_map_mlp_up{};
    uint64_t size_mlp_up[rank] = {HIDDEN_DIM, FFN_DIM};
    uint64_t stride_mlp_up[rank - 1] = {HIDDEN_DIM * sizeof(half)};
    uint32_t box_size_mlp_up[rank] = {TMA_LOAD_ONCE, HEAD_DIM};
    uint32_t elem_stride_mlp_up[rank] = {1, 1};
    cuTensorMapEncodeTiled(&tensor_map_mlp_up, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, rank, mlp_up_weight_ptr,
        size_mlp_up, stride_mlp_up, box_size_mlp_up, elem_stride_mlp_up, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    dim3 grid(HEAD_NUM * CLUSTER_SIZE);
    dim3 block(BLOCK_SIZE);

    cudaDeviceSynchronize();
    
    // Launch Attention + MLP Up kernel
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = max_shmem_size;
    
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = CLUSTER_SIZE;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    
    void* kernel_args[] = {
        &attn_output_ptr,
        &k_ptr, &v_ptr,
        &mlp_intermediate_ptr,
        &input_ptr,
        &layernorm_weight_ptr, &layernorm_bias_ptr,
        &bias_qkv_ptr, &bias_o_ptr,
        &cos_ptr, &sin_ptr,
        &k_cache_ptr, &v_cache_ptr,
        &post_ln_weight_ptr, &post_ln_bias_ptr,
        &mlp_up_bias_ptr, &post_ln_buffer_ptr,
        (void*)&tensor_map_weight, (void*)&tensor_map_k_cache,
        (void*)&tensor_map_v_cache, (void*)&tensor_map_weight_o,
        (void*)&tensor_map_mlp_up,
        (void*)&seq_len, (void*)&KV_DIM_PER_BLOCK
    };
    
    cudaLaunchKernelExC(&config, (void*)PythiaAttentionMlpUpKernel, kernel_args);
    
    cudaDeviceSynchronize();
    
    // Return: attn_output, mlp_intermediate, k_new, v_new
    return std::make_tuple(attn_output, mlp_intermediate, k, v);
}

