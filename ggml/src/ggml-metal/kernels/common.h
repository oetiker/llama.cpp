#pragma once

#include "ggml-metal-impl.h"

#include <metal_stdlib>

#ifdef GGML_METAL_HAS_TENSOR
#include <metal_tensor>

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#endif

using namespace metal;

#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define SWAP(x, y) { auto tmp = (x); (x) = (y); (y) = tmp; }

#define PAD2(x, n) (((x) + (n) - 1) & ~((n) - 1))

#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

#define N_SIMDWIDTH 32 // assuming SIMD group size is 32

// ref: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
//
// cmd:
//   .../usr/bin/metal -dM -E -c                             ggml/src/ggml-metal/kernels/<src>.metal
//   .../usr/bin/metal -dM -E -c -target air64-apple-ios14.0 ggml/src/ggml-metal/kernels/<src>.metal
//
#if __METAL_VERSION__ < 310 && defined(GGML_METAL_HAS_BF16)
#undef GGML_METAL_HAS_BF16
#endif

#if defined(GGML_METAL_HAS_BF16)
typedef matrix<bfloat, 4, 4> bfloat4x4;
typedef matrix<bfloat, 2, 4> bfloat2x4;
#endif

constexpr constant static float kvalues_iq4nl_f[16] = {
    -127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f, 1.f, 13.f, 25.f, 38.f, 53.f, 69.f, 89.f, 113.f
};

constexpr constant static float kvalues_mxfp4_f[16] = {
    0, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f
};

static inline int best_index_int8(int n, constant float * val, float x) {
    if (x <= val[0]) return 0;
    if (x >= val[n-1]) return n-1;
    int ml = 0, mu = n-1;
    while (mu-ml > 1) {
        int mav = (ml+mu)/2;
        if (x < val[mav]) mu = mav; else ml = mav;
    }
    return x - val[mu-1] < val[mu] - x ? mu-1 : mu;
}

static inline float e8m0_to_fp32(uint8_t x) {
    uint32_t bits;

    if (x == 0) {
        bits = 0x00400000;
    } else {
        bits = (uint32_t) x << 23;
    }

    return as_type<float>(bits);
}

static inline float dot(float x, float y) {
    return x*y;
}

static inline float sum(float x) {
    return x;
}

static inline float sum(float4 x) {
    return x[0] + x[1] + x[2] + x[3];
}

enum ggml_sort_order {
    GGML_SORT_ORDER_ASC,
    GGML_SORT_ORDER_DESC,
};

constant float GELU_COEF_A     = 0.044715f;
constant float GELU_QUICK_COEF = -1.702f;
constant float SQRT_2_OVER_PI  = 0.79788456080286535587989211986876f;
constant float SQRT_2_INV      = 0.70710678118654752440084436210484f;

// erf: port of FreeBSD msun s_erff.c (fdlibm s_erf.c converted to float by
// Ian Lance Taylor, Cygnus Support). The copy ported here is rust-lang/libm
// 0.2.16 src/math/erff.rs: its constants and branch structure, except that the
// [0.84375, 1.25) branch uses FreeBSD's erx + P/Q form directly instead of
// libm's 1 - (1 - erx - P/Q). rust-lang/libm is MIT-licensed; see its
// LICENSE.txt. The Sun notice below covers the algorithm itself.
// Replaces the Abramowitz-Stegun 7.1.26 approximation (up to ~6.6e-7 absolute error,
// no relative precision near 0) so that GELU_ERF on Metal tracks the CPU backend.
// CPU sweep over all finite floats: max 0.97 ulp, and max 1.56 ulp with exp
// perturbed by +-4 ulp and divide by +-1 ulp.
// Metal compiles this library with fast math on (default MTLCompileOptions), so:
//   - exp and the divisions go through precise::exp / precise::divide
//   - inf/NaN are classified on the bit pattern, not with float compares
//   - denormal inputs and results may flush to zero, which only affects |x| < 2^-126
//
// ====================================================
// Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
//
// Developed at SunPro, a Sun Microsystems, Inc. business.
// Permission to use, copy, modify, and distribute this
// software is freely granted, provided that this notice
// is preserved.
// ====================================================
constant float erx_erf  =  8.4506291151e-01f; // 0x3f58560b
// erf on [0, 0.84375]
constant float efx8_erf =  1.0270333290e+00f; // 0x3f8375d4
constant float pp0_erf  =  1.2837916613e-01f; // 0x3e0375d4
constant float pp1_erf  = -3.2504209876e-01f; // 0xbea66beb
constant float pp2_erf  = -2.8481749818e-02f; // 0xbce9528f
constant float pp3_erf  = -5.7702702470e-03f; // 0xbbbd1489
constant float pp4_erf  = -2.3763017452e-05f; // 0xb7c756b1
constant float qq1_erf  =  3.9791721106e-01f; // 0x3ecbbbce
constant float qq2_erf  =  6.5022252500e-02f; // 0x3d852a63
constant float qq3_erf  =  5.0813062117e-03f; // 0x3ba68116
constant float qq4_erf  =  1.3249473704e-04f; // 0x390aee49
constant float qq5_erf  = -3.9602282413e-06f; // 0xb684e21a
// erf on [0.84375, 1.25]
constant float pa0_erf  = -2.3621185683e-03f; // 0xbb1acdc6
constant float pa1_erf  =  4.1485610604e-01f; // 0x3ed46805
constant float pa2_erf  = -3.7220788002e-01f; // 0xbebe9208
constant float pa3_erf  =  3.1834661961e-01f; // 0x3ea2fe54
constant float pa4_erf  = -1.1089469492e-01f; // 0xbde31cc2
constant float pa5_erf  =  3.5478305072e-02f; // 0x3d1151b3
constant float pa6_erf  = -2.1663755178e-03f; // 0xbb0df9c0
constant float qa1_erf  =  1.0642088205e-01f; // 0x3dd9f331
constant float qa2_erf  =  5.4039794207e-01f; // 0x3f0a5785
constant float qa3_erf  =  7.1828655899e-02f; // 0x3d931ae7
constant float qa4_erf  =  1.2617121637e-01f; // 0x3e013307
constant float qa5_erf  =  1.3637083583e-02f; // 0x3c5f6e13
constant float qa6_erf  =  1.1984500103e-02f; // 0x3c445aa3
// erfc on [1.25, 1/0.35]
constant float ra0_erf  = -9.8649440333e-03f; // 0xbc21a093
constant float ra1_erf  = -6.9385856390e-01f; // 0xbf31a0b7
constant float ra2_erf  = -1.0558626175e+01f; // 0xc128f022
constant float ra3_erf  = -6.2375331879e+01f; // 0xc2798057
constant float ra4_erf  = -1.6239666748e+02f; // 0xc322658c
constant float ra5_erf  = -1.8460508728e+02f; // 0xc3389ae7
constant float ra6_erf  = -8.1287437439e+01f; // 0xc2a2932b
constant float ra7_erf  = -9.8143291473e+00f; // 0xc11d077e
constant float sa1_erf  =  1.9651271820e+01f; // 0x419d35ce
constant float sa2_erf  =  1.3765776062e+02f; // 0x4309a863
constant float sa3_erf  =  4.3456588745e+02f; // 0x43d9486f
constant float sa4_erf  =  6.4538726807e+02f; // 0x442158c9
constant float sa5_erf  =  4.2900814819e+02f; // 0x43d6810b
constant float sa6_erf  =  1.0863500214e+02f; // 0x42d9451f
constant float sa7_erf  =  6.5702495575e+00f; // 0x40d23f7c
constant float sa8_erf  = -6.0424413532e-02f; // 0xbd777f97
// erfc on [1/0.35, 6]
constant float rb0_erf  = -9.8649431020e-03f; // 0xbc21a092
constant float rb1_erf  = -7.9928326607e-01f; // 0xbf4c9dd4
constant float rb2_erf  = -1.7757955551e+01f; // 0xc18e104b
constant float rb3_erf  = -1.6063638306e+02f; // 0xc320a2ea
constant float rb4_erf  = -6.3756646729e+02f; // 0xc41f6441
constant float rb5_erf  = -1.0250950928e+03f; // 0xc480230b
constant float rb6_erf  = -4.8351919556e+02f; // 0xc3f1c275
constant float sb1_erf  =  3.0338060379e+01f; // 0x41f2b459
constant float sb2_erf  =  3.2579251099e+02f; // 0x43a2e571
constant float sb3_erf  =  1.5367296143e+03f; // 0x44c01759
constant float sb4_erf  =  3.1998581543e+03f; // 0x4547fdbb
constant float sb5_erf  =  2.5530502930e+03f; // 0x451f90ce
constant float sb6_erf  =  4.7452853394e+02f; // 0x43ed43a7
constant float sb7_erf  = -2.2440952301e+01f; // 0xc1b38712

static inline float erf_f32(float x) {
    const uint hx = as_type<uint>(x);
    const uint ix = hx & 0x7fffffffu;
    const bool neg = (hx >> 31) != 0;

    if (ix >= 0x7f800000u) {
        // erf(nan) = nan, erf(+-inf) = +-1
        return ix > 0x7f800000u ? x : (neg ? -1.0f : 1.0f);
    }

    if (ix < 0x3f580000u) {
        // |x| < 0.84375
        if (ix < 0x31800000u) {
            // |x| < 2^-28
            return 0.125f*(8.0f*x + efx8_erf*x);
        }
        const float z = x*x;
        const float r = pp0_erf + z*(pp1_erf + z*(pp2_erf + z*(pp3_erf + z*pp4_erf)));
        const float s = 1.0f + z*(qq1_erf + z*(qq2_erf + z*(qq3_erf + z*(qq4_erf + z*qq5_erf))));
        return x + x*precise::divide(r, s);
    }

    float y;

    if (ix < 0x3fa00000u) {
        // 0.84375 <= |x| < 1.25
        const float s = fabs(x) - 1.0f;
        const float p = pa0_erf + s*(pa1_erf + s*(pa2_erf + s*(pa3_erf + s*(pa4_erf + s*(pa5_erf + s*pa6_erf)))));
        const float q = 1.0f + s*(qa1_erf + s*(qa2_erf + s*(qa3_erf + s*(qa4_erf + s*(qa5_erf + s*qa6_erf)))));
        y = erx_erf + precise::divide(p, q);
    } else if (ix < 0x40c00000u) {
        // 1.25 <= |x| < 6: erf = 1 - erfc
        const float ax = fabs(x);
        const float s  = precise::divide(1.0f, ax*ax);
        float r;
        float S;
        if (ix < 0x4036db6du) {
            // |x| < 1/0.35
            r = ra0_erf + s*(ra1_erf + s*(ra2_erf + s*(ra3_erf + s*(ra4_erf + s*(ra5_erf + s*(ra6_erf + s*ra7_erf))))));
            S = 1.0f + s*(sa1_erf + s*(sa2_erf + s*(sa3_erf + s*(sa4_erf + s*(sa5_erf + s*(sa6_erf + s*(sa7_erf + s*sa8_erf)))))));
        } else {
            // |x| >= 1/0.35
            r = rb0_erf + s*(rb1_erf + s*(rb2_erf + s*(rb3_erf + s*(rb4_erf + s*(rb5_erf + s*rb6_erf)))));
            S = 1.0f + s*(sb1_erf + s*(sb2_erf + s*(sb3_erf + s*(sb4_erf + s*(sb5_erf + s*(sb6_erf + s*sb7_erf))))));
        }
        // split ax*ax: z keeps 11 significant bits, so z*z is exact
        const float z = as_type<float>(as_type<uint>(ax) & 0xffffe000u);
        const float erfc_x = precise::divide(precise::exp(-z*z - 0.5625f)*precise::exp((z - ax)*(z + ax) + precise::divide(r, S)), ax);
        y = 1.0f - erfc_x;
    } else {
        // |x| >= 6: 1 - 2^-120 rounds to 1
        y = 1.0f;
    }

    return neg ? -y : y;
}

// all instantiations compute in float
template<typename T> T erf_approx(T x);

template<> inline float erf_approx<float>(float x) {
    return erf_f32(x);
}

template<> inline float4 erf_approx<float4>(float4 x) {
    return float4(erf_f32(x[0]), erf_f32(x[1]), erf_f32(x[2]), erf_f32(x[3]));
}

template<> inline half erf_approx<half>(half x) {
    return (half) erf_f32((float) x);
}

template<> inline half4 erf_approx<half4>(half4 x) {
    return (half4) erf_approx<float4>((float4) x);
}

template<typename T> T elu_approx(T x);

template<> inline float elu_approx<float>(float x) {
    return (x > 0.f) ? x : (exp(x) - 1);
}

template<> inline float4 elu_approx<float4>(float4 x) {
    float4 res;

    res[0] = (x[0] > 0.0f) ? x[0] : (exp(x[0]) - 1.0f);
    res[1] = (x[1] > 0.0f) ? x[1] : (exp(x[1]) - 1.0f);
    res[2] = (x[2] > 0.0f) ? x[2] : (exp(x[2]) - 1.0f);
    res[3] = (x[3] > 0.0f) ? x[3] : (exp(x[3]) - 1.0f);

    return res;
}
