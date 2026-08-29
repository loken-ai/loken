// Header file for custom CUDA sampling kernels
// Declares the functions that will be used for GPU-native sampling operations

#ifndef SAMPLING_KERNELS_H
#define SAMPLING_KERNELS_H

#ifdef __cplusplus
extern "C" {
#endif

// Function declarations for CUDA kernels
void launch_top_p_filtering(
    const float* probs,
    int* indices,
    int* filtered_indices,
    float* filtered_probs,
    int vocab_size,
    float top_p,
    int* num_filtered,
    int batch_size,
    void* stream
);

void launch_top_k_selection(
    const float* probs,
    int* indices,
    int* selected_indices,
    float* selected_probs,
    int vocab_size,
    int top_k,
    int batch_size,
    void* stream
);

void launch_multinomial_sampling(
    const float* filtered_probs,
    int* filtered_indices,
    int* num_filtered,
    int* sampled_tokens,
    int batch_size,
    int max_vocab_size,
    void* stream
);

#ifdef __cplusplus
}
#endif

#endif // SAMPLING_KERNELS_H