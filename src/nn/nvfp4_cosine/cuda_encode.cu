// The storage recipe, not a weight-matrix NVFP4 quantizer. One invocation
// owns one row. Every operation that affects a certificate has explicit
// round-to-nearest arithmetic; next-up steps mirror nvfp4_cosine.rs.
__device__ double up(double x) {
    if (x == 0.0) return __longlong_as_double(1LL);
    return __longlong_as_double(__double_as_longlong(x) + (x > 0.0 ? 1LL : -1LL));
}
__device__ double add_up(double a, double b) {
    return a == 0.0 && b == 0.0 ? 0.0 : up(__dadd_rn(a, b));
}
__device__ double mul_up(double a, double b) {
    return a == 0.0 || b == 0.0 ? 0.0 : up(__dmul_rn(a, b));
}
__device__ double magnitude(double x) { return x < 0.0 ? -x : x; }
__device__ double e4(unsigned char b) {
    int exponent = (b >> 3) & 15;
    int mantissa = b & 7;
    int power = exponent == 0 ? -9 : exponent - 10;
    double scale = __longlong_as_double(((long long)(power + 1023)) << 52);
    return __dmul_rn((double)(exponent == 0 ? mantissa : 8 + mantissa), scale);
}
__device__ double e2(unsigned char b) {
    const double positive[8] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};
    return b & 8 ? -positive[b & 7] : positive[b & 7];
}
__device__ unsigned char encode4(double x) {
    x = x < 0.0 ? 0.0 : (x > 448.0 ? 448.0 : x);
    int best = 0;
    double distance = __longlong_as_double(0x7ff0000000000000LL);
    for (int raw = 0; raw <= 126; ++raw) {
        double d = magnitude(__dsub_rn(e4((unsigned char)raw), x));
        if (d < distance || (d == distance &&
            (((raw & 1) == 0 && (best & 1) != 0) ||
             ((raw & 1) == (best & 1) && raw < best)))) {
            best = raw; distance = d;
        }
    }
    return (unsigned char)best;
}
__device__ unsigned char encode2(double x) {
    double a = magnitude(x);
    a = a > 6.0 ? 6.0 : a;
    int best = 0;
    double distance = __longlong_as_double(0x7ff0000000000000LL);
    for (int raw = 0; raw < 8; ++raw) {
        double d = magnitude(__dsub_rn(e2((unsigned char)raw), a));
        if (d < distance || (d == distance &&
            (((raw & 1) == 0 && (best & 1) != 0) ||
             ((raw & 1) == (best & 1) && raw < best)))) {
            best = raw; distance = d;
        }
    }
    return best == 0 ? 0 : (unsigned char)(best | (x < 0.0 ? 8 : 0));
}
__device__ void put_f32(unsigned char* out, float value) {
    unsigned int bits = __float_as_uint(value);
    for (int j = 0; j < 4; ++j) out[j] = (unsigned char)(bits >> (j * 8));
}
__device__ float get_f32(const unsigned char* in) {
    unsigned int bits = 0;
    for (int j = 0; j < 4; ++j) bits |= ((unsigned int)in[j]) << (j * 8);
    return __uint_as_float(bits);
}
__device__ float upward_f32(double x) {
    float rounded = __double2float_rn(x);
    if ((double)rounded < x) rounded = __uint_as_float(__float_as_uint(rounded) + 1u);
    return rounded;
}
__device__ double norm(const double* values) {
    double squared = 0.0;
    for (int i = 0; i < 4096; ++i) {
        if (values[i] != 0.0) {
            double a = up(magnitude(values[i]));
            squared = up(__dadd_rn(squared, up(__dmul_rn(a, a))));
        }
    }
    return squared == 0.0 ? 0.0 : up(__dsqrt_rn(squared));
}
__device__ double distance(const double* left, const double* right) {
    double squared = 0.0;
    for (int i = 0; i < 4096; ++i) {
        double d = magnitude(__dsub_rn(left[i], right[i]));
        if (d != 0.0) {
            d = up(d);
            squared = up(__dadd_rn(squared, up(__dmul_rn(d, d))));
        }
    }
    return up(__dsqrt_rn(squared));
}
__device__ bool quantize(const double* values, unsigned char* out) {
    double a = 0.0;
    for (int i = 0; i < 4096; ++i) {
        double v = magnitude(values[i]); a = v > a ? v : a;
    }
    if (a == 0.0) {
        for (int i = 0; i < 2308; ++i) out[i] = 0;
        return true;
    }
    float global = __double2float_rn(__ddiv_rn(a, 2688.0));
    if (!(global > 0.0f) || (__float_as_uint(global) & 0x7f800000u) == 0x7f800000u)
        return false;
    put_f32(out, global);
    for (int block = 0; block < 256; ++block) {
        a = 0.0;
        for (int j = 0; j < 16; ++j) {
            double v = magnitude(values[block * 16 + j]); a = v > a ? v : a;
        }
        unsigned char scale = a == 0.0 ? 0 : encode4(__ddiv_rn(a, __dmul_rn(6.0, (double)global)));
        out[4 + block] = scale;
        double decoded = __dmul_rn((double)global, e4(scale));
        for (int j = 0; j < 8; ++j) {
            int i = block * 16 + j * 2;
            unsigned char low = decoded == 0.0 ? 0 : encode2(__ddiv_rn(values[i], decoded));
            unsigned char high = decoded == 0.0 ? 0 : encode2(__ddiv_rn(values[i + 1], decoded));
            out[260 + block * 8 + j] = low | (high << 4);
        }
    }
    return true;
}
__device__ double decode(const unsigned char* stage, int i) {
    double scale = __dmul_rn((double)get_f32(stage), e4(stage[4 + i / 16]));
    unsigned char pair = stage[260 + i / 2];
    return __dmul_rn(e2(i & 1 ? pair >> 4 : pair & 15), scale);
}
__device__ float decode32(const unsigned char* out, int i) {
    float p = __fmaf_rn(get_f32(out), (float)e4(out[4 + i / 16]), 0.0f);
    float c = __fmaf_rn(get_f32(out + 2308), (float)e4(out[2312 + i / 16]), 0.0f);
    unsigned char pb = out[260 + i / 2], cb = out[2568 + i / 2];
    float first = __fmaf_rn((float)e2(i & 1 ? pb >> 4 : pb & 15), p, 0.0f);
    return __fmaf_rn((float)e2(i & 1 ? cb >> 4 : cb & 15), c, first);
}
extern "C" __global__ void __launch_bounds__(1) encode_nvfp4_4096(
    const unsigned short* __restrict__ input,
    double* __restrict__ scratch,
    unsigned char* __restrict__ out,
    unsigned int* __restrict__ status
) {
    double* normalized = scratch;
    double* transformed = scratch + 4096;
    double* residual = scratch + 8192;
    double* reconstruction = scratch + 12288;
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
        normalized[i] = denominator == 0.0 ? 0.0 : __ddiv_rn(value, denominator);
        unsigned long long sign = ((unsigned long long)i) ^ 0x6B3F7B4CDA9E0673ULL;
        sign += 0x9E3779B97F4A7C15ULL;
        sign = (sign ^ (sign >> 30)) * 0xBF58476D1CE4E5B9ULL;
        sign = (sign ^ (sign >> 27)) * 0x94D049BB133111EBULL;
        sign ^= sign >> 31;
        transformed[i] = sign & 1 ? -normalized[i] : normalized[i];
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
        for (int i = 0; i < 256; ++i) transformed[block + i] = __dmul_rn(transformed[block + i], 0.0625);
    }
    status[0] = 2;
    if (!quantize(transformed, out)) return;
    for (int i = 0; i < 4096; ++i) residual[i] = __dsub_rn(transformed[i], decode(out, i));
    if (!quantize(residual, out + 2308)) return;
    for (int i = 0; i < 4096; ++i) reconstruction[i] = __dadd_rn(decode(out, i), decode(out + 2308, i));
    double numerator = mul_up(8.0, 0x1p-53);
    double gamma_denominator = __dsub_rn(1.0, numerator);
    gamma_denominator = __longlong_as_double(__double_as_longlong(gamma_denominator) - 1LL);
    double gamma = up(__ddiv_rn(numerator, gamma_denominator));
    double allowance = mul_up(mul_up(16.0, gamma), norm(normalized));
    double source_error = add_up(distance(transformed, reconstruction), allowance);
    for (int i = 0; i < 4096; ++i) residual[i] = (double)decode32(out, i);
    double decode_error = distance(reconstruction, residual);
    put_f32(out + 4616, upward_f32(norm(reconstruction)));
    put_f32(out + 4620, upward_f32(source_error > decode_error ? source_error : decode_error));
    status[0] = 0;
}
