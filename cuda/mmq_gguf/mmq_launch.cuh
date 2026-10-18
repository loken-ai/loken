#pragma once

// Choosing a tile and firing the tiled quantised matmul.
//
// Every format asks the same three questions - how wide a tile fits in this card's shared
// memory, whether the row count divides it, and whether the work splits across the SMs - and
// answers them the same way. Ten translation units each held their own copy of that: 145
// lines apiece, differing in a type name and nothing else.
//
// A format still gets its own translation unit, because `mul_mat_q` is a heavy template
// expansion and compiling the ten in parallel is most of the build. What it no longer gets is
// its own copy of the decision.

// The tile widths worth trying: multiples of eight, up to the widest a warp can serve. Stated
// once so the search below and the dispatch below it cannot drift apart.
#define MMQ_TILE_WIDTHS(X)                                                                    \
    X(8) X(16) X(24) X(32) X(40) X(48) X(56) X(64)                                            \
    X(72) X(80) X(88) X(96) X(104) X(112) X(120) X(128)

/// How many weight rows one tile covers. Volta and later hold twice as many.
static int mmq_rows_per_tile(const int cc) {
    if (GGML_CUDA_CC_IS_AMD(cc)) {
        const int gen = GGML_CUDA_CC_AMD_GEN(cc);
        return gen >= 0x1010 && gen < 0x1030 ? 64 : 128;
    }
    return (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA)
               ? 128
               : 64;
}

/// Whether the card multiplies whole tiles with AMD's matrix cores, whose operand spans the wave.
static bool mmq_amd_matrix_cores(const int cc) {
    return amd_mfma_available(cc) || amd_wmma_available(cc);
}

/// The widest tile the device code of this card was built for, as `get_mmq_x_max_device`.
static int mmq_x_max_host(const int cc) {
    return turing_mma_available(cc) || mmq_amd_matrix_cores(cc) ? 128 : 64;
}

/// The row granularity of a tile `mmq_x` wide, as `mmq_get_granularity_device`: a width the
/// device code was not built for writes nothing.
static int mmq_granularity_host(const int mmq_x, const int cc) {
    if (mmq_amd_matrix_cores(cc)) {
        return mmq_x >= 128 ? 32 : 16;
    }
    if (turing_mma_available(cc)) {
        return mmq_x >= 48 ? 16 : 8;
    }
    return 8;
}

/// Launch one already-chosen tile width.
///
/// Two kernels rather than one: `need_check` guards every row index against the edge of the
/// matrix, which costs a comparison per access and is pure waste when the row count divides
/// the tile. The stream-k variant splits the k dimension across all SMs instead of giving
/// each tile its own block, and needs a fixup pass afterwards whenever the tiles do not
/// divide evenly among them.
template <ggml_type type, int mmq_x>
static void mmq_launch_tile(float *tmp_fixup, const mmq_args &args, cudaStream_t stream,
                            const int cc, const int nsm, const size_t smpbo,
                            const int warp_size_host) {
    const int mmq_y = mmq_rows_per_tile(cc);
    const int nwarps = 256 / warp_size_host;
    const int nbytes_shared = mmq_get_nbytes_shared<type>(mmq_x, mmq_y, cc, warp_size_host, nwarps);
    const int nty = (args.nrows_x + mmq_y - 1) / mmq_y;
    const int ntx = (args.ncols_max + mmq_x - 1) / mmq_x;
    const int ntzw = args.nchannels_y * args.nsamples_y;
    const dim3 block_dims(warp_size_host, nwarps, 1);

    // The kernel divides by these on every iteration; precomputing the reciprocals moves the
    // division off the device.
    constexpr int qk_t = ggml_cuda_type_traits<type>::qk;
    const uint3 blocks_per_ne00_fd = init_fastdiv_values((uint32_t) (args.ncols_x / qk_t));
    const uint3 ntx_fd = init_fastdiv_values((uint32_t) ntx);
    const uint3 nchannels_y_fd = init_fastdiv_values((uint32_t) args.nchannels_y);
    const uint3 nsamples_y_fd = init_fastdiv_values((uint32_t) args.nsamples_y);
    const uint3 channel_ratio_fd =
        init_fastdiv_values((uint32_t) (args.nchannels_y / args.nchannels_x));
    const uint3 sample_ratio_fd = init_fastdiv_values((uint32_t) (args.nsamples_y / args.nsamples_x));

    CUDA_SET_SHARED_MEMORY_LIMIT((mul_mat_q<type, mmq_x, false>), nbytes_shared);
    CUDA_SET_SHARED_MEMORY_LIMIT((mul_mat_q<type, mmq_x, true>), nbytes_shared);

    const bool ragged = args.nrows_x % mmq_y != 0;

// The argument list is long and identical in all four launches; naming it once keeps the
// difference between them - grid shape, fixup buffer, edge checking - visible.
#define MMQ_MATMUL_ARGS(fixup)                                                                \
    args.x, args.y, args.dst, fixup, blocks_per_ne00_fd,                                     \
        args.nrows_x, args.ncols_dst, args.stride_row_x, args.ncols_y, args.nrows_dst,        \
        channel_ratio_fd, nchannels_y_fd, args.stride_channel_x, args.stride_channel_y,       \
        args.stride_channel_dst, sample_ratio_fd, nsamples_y_fd, args.stride_sample_x,        \
        args.stride_sample_y, args.stride_sample_dst, ntx_fd

    if (!args.use_stream_k) {
        const dim3 grid(nty, ntx, ntzw);
        if (ragged) {
            mul_mat_q<type, mmq_x, true>
                <<<grid, block_dims, nbytes_shared, stream>>>(MMQ_MATMUL_ARGS(nullptr));
        } else {
            mul_mat_q<type, mmq_x, false>
                <<<grid, block_dims, nbytes_shared, stream>>>(MMQ_MATMUL_ARGS(nullptr));
        }
        return;
    }

    const dim3 grid_sk(nsm, 1, 1);
    const dim3 grid_sk_fixup(nsm, mmq_y / warp_size_host, 1);
    const dim3 block_dims_fixup(warp_size_host, nwarps / 2, 1);
    // Only when the tiles do not divide among the SMs does a block end up holding a partial
    // sum somebody else has to finish.
    const bool fixup_needed = ntx * nty * ntzw % nsm != 0;

#define MMQ_FIXUP_ARGS                                                                        \
    args.dst, tmp_fixup, blocks_per_ne00_fd, args.nrows_x,                                    \
        args.ncols_dst, args.nrows_dst, nchannels_y_fd, args.stride_channel_dst,              \
        nsamples_y_fd, args.stride_sample_dst, ntx_fd

    if (ragged) {
        mul_mat_q<type, mmq_x, true>
            <<<grid_sk, block_dims, nbytes_shared, stream>>>(MMQ_MATMUL_ARGS(tmp_fixup));
        if (fixup_needed) {
            mul_mat_q_stream_k_fixup<type, mmq_x, true>
                <<<grid_sk_fixup, block_dims_fixup, 0, stream>>>(MMQ_FIXUP_ARGS);
        }
    } else {
        mul_mat_q<type, mmq_x, false>
            <<<grid_sk, block_dims, nbytes_shared, stream>>>(MMQ_MATMUL_ARGS(tmp_fixup));
        if (fixup_needed) {
            mul_mat_q_stream_k_fixup<type, mmq_x, false>
                <<<grid_sk_fixup, block_dims_fixup, 0, stream>>>(MMQ_FIXUP_ARGS);
        }
    }

#undef MMQ_MATMUL_ARGS
#undef MMQ_FIXUP_ARGS
}

/// Pick the tile width, then launch it.
///
/// Wider tiles reuse a loaded weight across more activation columns, so the search wants the
/// widest one that still fits in shared memory - but only up to the point where the whole
/// output is one tile across, since past that the extra width buys nothing and costs
/// occupancy. Tensor-core tiles are laid out in groups of sixteen rather than eight, so a
/// width that is not a multiple of the granularity cannot be built at all.
template <ggml_type type>
static void mmq_launch(float *tmp_fixup, const mmq_args &args, cudaStream_t stream, const int cc,
                       const int nsm, const size_t smpbo, const int warp_size_host) {
    const int mmq_x_max = mmq_x_max_host(cc);
    const int mmq_y = mmq_rows_per_tile(cc);
    const int nwarps = 256 / warp_size_host;

    int best = 0;
    int fewest_tiles = INT_MAX;
    for (int mmq_x = 8; mmq_x <= mmq_x_max && fewest_tiles > 1; mmq_x += 8) {
        const int granularity = mmq_granularity_host(mmq_x, cc);
        if (mmq_x % granularity != 0) {
            continue;
        }
        if (mmq_get_nbytes_shared<type>(mmq_x, mmq_y, cc, warp_size_host, nwarps) > smpbo) {
            continue;
        }
        const int tiles = (args.ncols_max + mmq_x - 1) / mmq_x;
        if (tiles < fewest_tiles) {
            best = mmq_x;
            fewest_tiles = tiles;
        }
    }

    switch (best) {
#define MMQ_DISPATCH_WIDTH(w)                                                                 \
    case w:                                                                                   \
        mmq_launch_tile<type, w>(tmp_fixup, args, stream, cc, nsm, smpbo, warp_size_host);    \
        break;
        MMQ_TILE_WIDTHS(MMQ_DISPATCH_WIDTH)
#undef MMQ_DISPATCH_WIDTH
        // Nothing fitted: the caller keeps its result buffer untouched and falls back.
        default:
            break;
    }
}

/// The C entry point for one format - what `build.rs` links and Rust declares.
///
/// The shape arguments describe a plain 2-D matmul; the channel and sample strides the kernel
/// also understands are the batched form, which this path leaves at one channel, one sample.
#define MMQ_ENTRY_POINT(ggml_type_name, suffix)                                               \
    extern "C" void launch_mmq_gguf_##suffix(                                                 \
        void *tmp_fixup_ptr, const void *x, const void *y_q8_1_mmq, void *dst,                \
        int64_t ncols_x, int64_t nrows_x, int64_t ncols_y, int64_t stride_row_x,              \
        int64_t stride_col_dst, int cc, int nsm, int64_t smpbo, int warp_size_host,           \
        void *stream) {                                                                       \
        (void) stride_col_dst;                                                                \
        const bool use_stream_k = GGML_CUDA_CC_IS_NVIDIA(cc) &&                               \
                                  ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA;  \
        const mmq_args args = {                                                               \
            (const char *) x, ggml_type_name, (const int *) y_q8_1_mmq, (float *) dst,       \
            ncols_x, nrows_x, ncols_y, stride_row_x, ncols_y, nrows_x,                        \
            1, 1, 0, 0, 0, 1, 1, 0, 0, 0, use_stream_k, ncols_y};                             \
        mmq_launch<ggml_type_name>((float *) tmp_fixup_ptr, args, (cudaStream_t) stream, cc,  \
                                   nsm, smpbo, warp_size_host);                               \
    }

/// Launch the expert-batched matmul at one tile width.
template <ggml_type type, int mmq_x>
static void mmq_moe_launch_tile(const char *x, const char *x_second, int64_t dst_second,
                                const int *y, float *dst, const int *expert_bounds,
                                const int *ids_dst, int64_t stride_expert_x, int64_t ncols_x,
                                int64_t nrows_x, int64_t ncols_y, int64_t stride_row_x,
                                int64_t stride_col_dst, int n_experts, int64_t ncols_max,
                                cudaStream_t stream, const int cc, const int warp_size_host) {
    const int mmq_y = mmq_rows_per_tile(cc);
    const int nwarps = 256 / warp_size_host;
    const int nbytes_shared = mmq_get_nbytes_shared<type>(mmq_x, mmq_y, cc, warp_size_host, nwarps);
    constexpr int qk_t = ggml_cuda_type_traits<type>::qk;
    const int blocks_per_ne00 = (int) (ncols_x / qk_t);
    const int stacks = x_second ? 2 : 1;
    const dim3 grid((nrows_x + mmq_y - 1) / mmq_y, (ncols_max + mmq_x - 1) / mmq_x,
                    stacks * n_experts);
    const dim3 block_dims(warp_size_host, nwarps, 1);

    CUDA_SET_SHARED_MEMORY_LIMIT((mul_mat_q_moe<type, mmq_x, false>), nbytes_shared);
    CUDA_SET_SHARED_MEMORY_LIMIT((mul_mat_q_moe<type, mmq_x, true>), nbytes_shared);

    if (nrows_x % mmq_y != 0) {
        mul_mat_q_moe<type, mmq_x, true><<<grid, block_dims, nbytes_shared, stream>>>(
            x, x_second, y, dst, expert_bounds, ids_dst, stride_expert_x, dst_second, n_experts,
            blocks_per_ne00, (int) nrows_x, (int) stride_row_x, (int) ncols_y,
            (int) stride_col_dst);
    } else {
        mul_mat_q_moe<type, mmq_x, false><<<grid, block_dims, nbytes_shared, stream>>>(
            x, x_second, y, dst, expert_bounds, ids_dst, stride_expert_x, dst_second, n_experts,
            blocks_per_ne00, (int) nrows_x, (int) stride_row_x, (int) ncols_y,
            (int) stride_col_dst);
    }
}

/// Pick the expert-batched tile width, then launch it.
///
/// `ncols_max` is the run length the grid is sized for: a longer run is still covered whole,
/// its blocks taking several column tiles each. `ncols_hint` is the run length the width is
/// chosen for: the widest tile that still fits shared memory, cut back to the narrowest one
/// covering the run in as few tiles. `mmq_x` forces a width when it is non-zero. A non-null
/// `x_second` is a second stack of the same shape multiplied against the same columns in the
/// same launch, its results `dst_second` floats after the first's. Returns the width launched,
/// or 0 when none fits and nothing ran.
template <ggml_type type>
static int mmq_moe_launch(const char *x, const char *x_second, int64_t dst_second,
                          const int *y, float *dst, const int *expert_bounds,
                          const int *ids_dst, int64_t stride_expert_x, int64_t ncols_x,
                          int64_t nrows_x, int64_t ncols_y, int64_t stride_row_x,
                          int64_t stride_col_dst, int n_experts, int64_t ncols_max,
                          int64_t ncols_hint, int mmq_x, cudaStream_t stream, const int cc,
                          const size_t smpbo, const int warp_size_host) {
    const int mmq_x_max = mmq_x_max_host(cc);
    const int mmq_y = mmq_rows_per_tile(cc);
    const int nwarps = 256 / warp_size_host;

    int best = 0;
    int fewest_tiles = INT_MAX;
    for (int w = 8; w <= mmq_x_max; w += 8) {
        const int granularity = mmq_granularity_host(w, cc);
        if (w % granularity != 0) {
            continue;
        }
        if (mmq_get_nbytes_shared<type>(w, mmq_y, cc, warp_size_host, nwarps) > smpbo) {
            continue;
        }
        if (mmq_x != 0) {
            if (w == mmq_x) {
                best = w;
            }
            continue;
        }
        const int tiles = (int) ((ncols_hint + w - 1) / w);
        if (tiles < fewest_tiles) {
            best = w;
            fewest_tiles = tiles;
        }
    }

    switch (best) {
#define MMQ_MOE_DISPATCH_WIDTH(w)                                                             \
    case w:                                                                                   \
        mmq_moe_launch_tile<type, w>(x, x_second, dst_second, y, dst, expert_bounds, ids_dst, \
                                     stride_expert_x,                                         \
                                     ncols_x, nrows_x, ncols_y, stride_row_x, stride_col_dst, \
                                     n_experts, ncols_max, stream, cc, warp_size_host);       \
        break;
        MMQ_TILE_WIDTHS(MMQ_MOE_DISPATCH_WIDTH)
#undef MMQ_MOE_DISPATCH_WIDTH
        default:
            return 0;
    }
    return best;
}

/// The expert-batched C entry point for one format.
#define MMQ_MOE_ENTRY_POINT(ggml_type_name, suffix)                                           \
    extern "C" int launch_mmq_moe_gguf_##suffix(                                              \
        const void *x, const void *x_second, int64_t dst_second, const void *y_q8_1_mmq,      \
        void *dst, const int *expert_bounds,                                                  \
        const int *ids_dst, int64_t stride_expert_x, int64_t ncols_x, int64_t nrows_x,        \
        int64_t ncols_y, int64_t stride_row_x, int64_t stride_col_dst, int n_experts,         \
        int64_t ncols_max, int64_t ncols_hint, int mmq_x, int cc, int64_t smpbo,              \
        int warp_size_host, void *stream) {                                                   \
        return mmq_moe_launch<ggml_type_name>(                                                \
            (const char *) x, (const char *) x_second, dst_second, (const int *) y_q8_1_mmq,  \
            (float *) dst, expert_bounds,                                                     \
            ids_dst, stride_expert_x, ncols_x, nrows_x, ncols_y, stride_row_x,                \
            stride_col_dst, n_experts, ncols_max, ncols_hint, mmq_x, (cudaStream_t) stream,   \
            cc, (size_t) smpbo, warp_size_host);                                              \
    }
