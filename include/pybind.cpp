#include <torch/extension.h>

// ============================================================================
// Attention-Only Branch for Pythia-2.8B
// This version accelerates: LayerNorm → QKV → RoPE → Attention → Output → 
//                           Post-LN → MLP Up → GELU
// MLP Down is handled by PyTorch
// ============================================================================

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> llama_decoder_layer_sm120(
    torch::Tensor input,
    torch::Tensor weight_qkv,
    torch::Tensor weight_o,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor rms_input_weight,
    torch::Tensor cos,
    torch::Tensor sin
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> llama_decoder_layer_sglang_sm120(
    torch::Tensor input,
    torch::Tensor residual,
    torch::Tensor weight_qkv,
    torch::Tensor weight_o,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor rms_input_weight,
    float eps,
    torch::Tensor cos,
    torch::Tensor sin
);

void llama_decoder_layer_batch_sglang_sm120(
    torch::Tensor output,
    torch::Tensor residual_output,
    torch::Tensor input,
    torch::Tensor residual,
    torch::Tensor weight_qkv,
    torch::Tensor weight_o,
    torch::Tensor paged_kv_indptr,
    torch::Tensor paged_kv_indices,
    torch::Tensor k_cache_ptrs,
    torch::Tensor v_cache_ptrs,
    int layer_id,
    torch::Tensor rms_input_weight,
    float eps,
    torch::Tensor positions,
    torch::Tensor cos_sin
);

// Pythia-2.8B Attention-Only kernel
// Output: attn_output, mlp_intermediate, k_new, v_new
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> pythia_2b8_attention_only_sm120(
    torch::Tensor input,
    torch::Tensor weight_qkv,
    torch::Tensor bias_qkv,
    torch::Tensor weight_o,
    torch::Tensor bias_o,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor layernorm_weight,
    torch::Tensor layernorm_bias,
    torch::Tensor cos,
    torch::Tensor sin,
    torch::Tensor post_ln_weight,
    torch::Tensor post_ln_bias,
    torch::Tensor mlp_up_weight,
    torch::Tensor mlp_up_bias,
    int64_t current_seq_len
);

#ifdef COMPILE_SM120
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("llama_decoder_layer", &llama_decoder_layer_sm120, "");
    m.def("llama_decoder_layer_sglang", &llama_decoder_layer_sglang_sm120, "");
    m.def("llama_decoder_layer_batch_decode_sglang", &llama_decoder_layer_batch_sglang_sm120, "");
    
    // Pythia-2.8B Attention-Only
    m.def("pythia_2b8_attention_only", &pythia_2b8_attention_only_sm120, 
          "Pythia-2.8B Attention + MLP Up + GELU (MLP Down done by PyTorch)");
}
#endif
