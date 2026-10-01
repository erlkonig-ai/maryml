// Appended to the unchanged encoder translation unit for its RN helpers.
// Preparation and scoring never read an embedding back to the host.
extern "C" __global__ void __launch_bounds__(1) prepare_query_4096(
    const unsigned short* __restrict__ input,
    double* __restrict__ transformed,
    unsigned int* __restrict__ status
) {
    status[0] = 1;
    double sum = 0.0;
    for (int i = 0; i < 4096; ++i) {
        unsigned int bits = ((unsigned int)input[i]) << 16;
        if ((bits & 0x7f800000u) == 0x7f800000u) return;
        double value = (double)__uint_as_float(bits);
        sum = __dadd_rn(sum, __dmul_rn(value, value));
    }
    double denominator = __dsqrt_rn(sum);
    for (int i = 0; i < 4096; ++i) {
        double value = (double)__uint_as_float(((unsigned int)input[i]) << 16);
        double normalized = denominator == 0.0 ? 0.0 : __ddiv_rn(value, denominator);
        unsigned long long sign = ((unsigned long long)i) ^ 0x6B3F7B4CDA9E0673ULL;
        sign += 0x9E3779B97F4A7C15ULL;
        sign = (sign ^ (sign >> 30)) * 0xBF58476D1CE4E5B9ULL;
        sign = (sign ^ (sign >> 27)) * 0x94D049BB133111EBULL;
        sign ^= sign >> 31;
        transformed[i] = sign & 1 ? -normalized : normalized;
    }
    for (int block = 0; block < 4096; block += 256) {
        for (int width = 1; width < 256; width *= 2) {
            for (int start = 0; start < 256; start += width * 2) {
                for (int j = 0; j < width; ++j) {
                    double low = transformed[block + start + j];
                    double high = transformed[block + start + width + j];
                    transformed[block + start + j] = __dadd_rn(low, high);
                    transformed[block + start + width + j] = __dsub_rn(low, high);
                }
            }
        }
        for (int i = 0; i < 256; ++i)
            transformed[block + i] = __dmul_rn(transformed[block + i], 0.0625);
    }
    status[0] = 0;
}

extern "C" __global__ void __launch_bounds__(1) reconstructed_cosines_4096(
    const double* __restrict__ query,
    const unsigned char* __restrict__ globals0,
    const unsigned char* __restrict__ scales0,
    const unsigned char* __restrict__ codes0,
    const unsigned char* __restrict__ globals1,
    const unsigned char* __restrict__ scales1,
    const unsigned char* __restrict__ codes1,
    const unsigned char* __restrict__ norms,
    const unsigned char* __restrict__ errors,
    double* __restrict__ output,
    unsigned int* __restrict__ status
) {
    const unsigned long long row = blockIdx.x;
    const double row_norm = (double)get_f32(norms + row * 4);
    const double error = (double)get_f32(errors + row * 4);
    status[row] = 2;
    if (!isfinite(row_norm) || row_norm < 0.0 || !isfinite(error) || error < 0.0) return;
    // The canonical reader deliberately ignores row codes when its norm is zero.
    if (row_norm == 0.0) { output[row] = 0.0; status[row] = 0; return; }
    const double g0 = (double)get_f32(globals0 + row * 4);
    const double g1 = (double)get_f32(globals1 + row * 4);
    double lanes[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    for (int block = 0; block < 256; ++block) {
        const double s0 = __dmul_rn(g0, e4(scales0[row * 256 + block]));
        const double s1 = __dmul_rn(g1, e4(scales1[row * 256 + block]));
        for (int lane = 0; lane < 8; ++lane) {
            const unsigned long long code = row * 2048 + block * 8 + lane;
            unsigned char p = codes0[code], c = codes1[code];
            double low = __dadd_rn(__dmul_rn(e2(p & 15), s0), __dmul_rn(e2(c & 15), s1));
            double high = __dadd_rn(__dmul_rn(e2(p >> 4), s0), __dmul_rn(e2(c >> 4), s1));
            lanes[lane] = __dadd_rn(lanes[lane], __dmul_rn(query[block * 16 + lane * 2], low));
            lanes[lane] = __dadd_rn(lanes[lane], __dmul_rn(query[block * 16 + lane * 2 + 1], high));
        }
    }
    double sum = 0.0;
    for (int lane = 0; lane < 8; ++lane) sum = __dadd_rn(sum, lanes[lane]);
    double score = __ddiv_rn(sum, row_norm);
    status[row] = 3;
    if (!isfinite(score)) return;
    output[row] = score < -1.0 ? -1.0 : (score > 1.0 ? 1.0 : score);
    status[row] = 0;
}
