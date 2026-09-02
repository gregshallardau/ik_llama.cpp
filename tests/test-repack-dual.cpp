// Checks the out-of-place repack used by the dual tensor representation (-rtrd).
//
// iqk_repack_tensor() rewrites a tensor's rows into the interleaved "_R4"/"_R8"/... layout in
// place; iqk_repack_tensor_to() performs the same shuffle into a separate destination buffer so
// the original layout survives next to it. The two must agree byte for byte - the whole feature
// rests on the repacked sibling being exactly what -rtr would have produced.

#include "ggml.h"

#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

// declared in ggml/src/iqk/iqk_quantize.h, which is not on the tests' include path
extern "C" {
void iqk_repack_tensor(struct ggml_tensor * tensor);
bool iqk_repack_tensor_to(const struct ggml_tensor * tensor, void * dst_data);
int  iqk_repacked_type(const struct ggml_tensor * tensor);
}

int main() {
    constexpr int64_t n_per_row = 512;  // multiple of QK_K and of every legacy block size
    constexpr int64_t nrows     = 32;   // multiple of 4, 8 and 16 (the interleave widths)

    std::mt19937 rng(1234);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> src(n_per_row*nrows);
    for (auto & v : src) v = dist(rng);

    // Quantize `src` into `t` one row at a time. ggml_quantize_chunk() does not cover every type
    // in the repack map (Q8_K and friends), the per-type from_float does.
    auto quantize = [&src](ggml_type type, ggml_tensor * t) {
        const auto from_float = ggml_internal_get_type_traits(type).from_float;
        const size_t row_size = ggml_row_size(type, n_per_row);
        ggml_quantize_init(type); // the IQ types need their lookup tables
        std::memset(t->data, 0, ggml_nbytes(t)); // so any per-row padding is deterministic
        for (int64_t r = 0; r < nrows; ++r) {
            from_float(src.data() + r*n_per_row, (char *) t->data + r*row_size, n_per_row);
        }
    };

    int n_tested = 0, n_failed = 0;

    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        const auto type = ggml_type(t);
        if (!ggml_is_quantized(type)) continue;      // BF16_R16 needs AVX512-BF16 hardware; skip
        if (ggml_blck_size(type) <= 0) continue;
        if (n_per_row % ggml_blck_size(type)) continue;
        if (!ggml_internal_get_type_traits(type).from_float) continue;

        ggml_init_params ip = { 3*(ggml_tensor_overhead() + ggml_row_size(type, n_per_row)*nrows), nullptr, false };
        ggml_context * ctx = ggml_init(ip);
        if (!ctx) { printf("%-12s: failed to create a context\n", ggml_type_name(type)); ++n_failed; continue; }

        ggml_tensor * in_place = ggml_new_tensor_2d(ctx, type, n_per_row, nrows);
        ggml_tensor * keep     = ggml_new_tensor_2d(ctx, type, n_per_row, nrows);
        ggml_tensor * out_of_place = ggml_new_tensor_2d(ctx, type, n_per_row, nrows);

        const auto repacked = ggml_type(iqk_repacked_type(keep));
        if (repacked == type) { ggml_free(ctx); continue; }  // no interleaved variant for this type

        // The repack rewrites rows within the tensor's own storage, so a repacked type that needs
        // more bytes per row than the original would overflow it - in -rtr as much as here. One
        // that needs fewer is merely wasteful; llm_build_repack_dual() skips those rather than
        // reason about the slack, so skip them here too.
        const size_t row_size     = ggml_row_size(type, n_per_row);
        const size_t row_size_new = ggml_row_size(repacked, n_per_row);
        if (row_size_new > row_size) {
            printf("%-12s -> %-14s FAIL: %zu -> %zu bytes per row, the repack would overflow the tensor\n",
                    ggml_type_name(type), ggml_type_name(repacked), row_size, row_size_new);
            ++n_failed;
            ggml_free(ctx);
            continue;
        }
        if (row_size_new < row_size) {
            printf("%-12s -> %-14s skipped: not size preserving (%zu -> %zu bytes per row)\n",
                    ggml_type_name(type), ggml_type_name(repacked), row_size, row_size_new);
            ggml_free(ctx);
            continue;
        }
        ++n_tested;

        quantize(type, in_place);
        std::memcpy(keep->data, in_place->data, ggml_nbytes(in_place));

        std::memset(out_of_place->data, 0xa5, ggml_nbytes(out_of_place));

        iqk_repack_tensor(in_place);
        const bool ok_out = iqk_repack_tensor_to(keep, out_of_place->data);

        bool ok = true;
        if (!ok_out) {
            printf("%-12s: iqk_repack_tensor_to() refused a tensor iqk_repacked_type() accepted\n", ggml_type_name(type));
            ok = false;
        } else if (in_place->type != repacked) {
            printf("%-12s: in-place repack produced %s, expected %s\n", ggml_type_name(type),
                    ggml_type_name(in_place->type), ggml_type_name(repacked));
            ok = false;
        } else if (ggml_nbytes(in_place) != ggml_nbytes(keep)) {
            printf("%-12s: repack changed the tensor size (%zu -> %zu)\n", ggml_type_name(type),
                    ggml_nbytes(keep), ggml_nbytes(in_place));
            ok = false;
        } else if (std::memcmp(in_place->data, out_of_place->data, ggml_nbytes(in_place)) != 0) {
            printf("%-12s: out-of-place repack differs from the in-place one\n", ggml_type_name(type));
            ok = false;
        }
        // the source must come back untouched - that is the point of the out-of-place variant
        if (ok) {
            quantize(type, out_of_place); // reuse as scratch: re-derive the source bytes
            if (std::memcmp(keep->data, out_of_place->data, ggml_nbytes(keep)) != 0) {
                printf("%-12s: iqk_repack_tensor_to() modified its source tensor\n", ggml_type_name(type));
                ok = false;
            }
        }

        if (ok) printf("%-12s -> %-14s ok\n", ggml_type_name(type), ggml_type_name(repacked));
        else    ++n_failed;

        ggml_free(ctx);
    }

    if (n_tested == 0) {
        printf("no repack-eligible types on this build - nothing to check\n");
        return 0;
    }
    printf("\n%d/%d repack-eligible types match\n", n_tested - n_failed, n_tested);
    return n_failed == 0 ? 0 : 1;
}
