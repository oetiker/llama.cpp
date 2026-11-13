#ifndef GGML_ERF_GLSL
#define GGML_ERF_GLSL

// erf_f32: accurate f32 erf for the GELU_ERF / GEGLU_ERF shaders.
// Origin: FreeBSD msun s_erff.c (fdlibm s_erf.c converted to float by Ian Lance
// Taylor, Cygnus Support). The copy ported here is rust-lang/libm 0.2.16
// src/math/erff.rs, which carries the same algorithm and constants; the Metal
// backend (ggml-metal/kernels/common.h erf_f32) uses the same port.
// Replaces the Abramowitz-Stegun 7.1.26 approximation (up to ~6.6e-7 absolute
// error, no relative precision near 0) so that GELU_ERF on Vulkan tracks the
// CPU backend.
// GLSL notes:
//   - always computes in f32, also for the f16 shader variants
//   - inf/NaN are classified on the bit pattern, not with float compares
//   - exp and division have the Vulkan precision of the device (spec bounds:
//     exp 3+2|x| ulp, division 2.5 ulp); the error analysis behind this port
//     only needs a few ulp in either
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
const float erx_erf  =  8.4506291151e-01f; // 0x3f58560b
// erf on [0, 0.84375]
const float efx8_erf =  1.0270333290e+00f; // 0x3f8375d4
const float pp0_erf  =  1.2837916613e-01f; // 0x3e0375d4
const float pp1_erf  = -3.2504209876e-01f; // 0xbea66beb
const float pp2_erf  = -2.8481749818e-02f; // 0xbce9528f
const float pp3_erf  = -5.7702702470e-03f; // 0xbbbd1489
const float pp4_erf  = -2.3763017452e-05f; // 0xb7c756b1
const float qq1_erf  =  3.9791721106e-01f; // 0x3ecbbbce
const float qq2_erf  =  6.5022252500e-02f; // 0x3d852a63
const float qq3_erf  =  5.0813062117e-03f; // 0x3ba68116
const float qq4_erf  =  1.3249473704e-04f; // 0x390aee49
const float qq5_erf  = -3.9602282413e-06f; // 0xb684e21a
// erf on [0.84375, 1.25]
const float pa0_erf  = -2.3621185683e-03f; // 0xbb1acdc6
const float pa1_erf  =  4.1485610604e-01f; // 0x3ed46805
const float pa2_erf  = -3.7220788002e-01f; // 0xbebe9208
const float pa3_erf  =  3.1834661961e-01f; // 0x3ea2fe54
const float pa4_erf  = -1.1089469492e-01f; // 0xbde31cc2
const float pa5_erf  =  3.5478305072e-02f; // 0x3d1151b3
const float pa6_erf  = -2.1663755178e-03f; // 0xbb0df9c0
const float qa1_erf  =  1.0642088205e-01f; // 0x3dd9f331
const float qa2_erf  =  5.4039794207e-01f; // 0x3f0a5785
const float qa3_erf  =  7.1828655899e-02f; // 0x3d931ae7
const float qa4_erf  =  1.2617121637e-01f; // 0x3e013307
const float qa5_erf  =  1.3637083583e-02f; // 0x3c5f6e13
const float qa6_erf  =  1.1984500103e-02f; // 0x3c445aa3
// erfc on [1.25, 1/0.35]
const float ra0_erf  = -9.8649440333e-03f; // 0xbc21a093
const float ra1_erf  = -6.9385856390e-01f; // 0xbf31a0b7
const float ra2_erf  = -1.0558626175e+01f; // 0xc128f022
const float ra3_erf  = -6.2375331879e+01f; // 0xc2798057
const float ra4_erf  = -1.6239666748e+02f; // 0xc322658c
const float ra5_erf  = -1.8460508728e+02f; // 0xc3389ae7
const float ra6_erf  = -8.1287437439e+01f; // 0xc2a2932b
const float ra7_erf  = -9.8143291473e+00f; // 0xc11d077e
const float sa1_erf  =  1.9651271820e+01f; // 0x419d35ce
const float sa2_erf  =  1.3765776062e+02f; // 0x4309a863
const float sa3_erf  =  4.3456588745e+02f; // 0x43d9486f
const float sa4_erf  =  6.4538726807e+02f; // 0x442158c9
const float sa5_erf  =  4.2900814819e+02f; // 0x43d6810b
const float sa6_erf  =  1.0863500214e+02f; // 0x42d9451f
const float sa7_erf  =  6.5702495575e+00f; // 0x40d23f7c
const float sa8_erf  = -6.0424413532e-02f; // 0xbd777f97
// erfc on [1/0.35, 6]
const float rb0_erf  = -9.8649431020e-03f; // 0xbc21a092
const float rb1_erf  = -7.9928326607e-01f; // 0xbf4c9dd4
const float rb2_erf  = -1.7757955551e+01f; // 0xc18e104b
const float rb3_erf  = -1.6063638306e+02f; // 0xc320a2ea
const float rb4_erf  = -6.3756646729e+02f; // 0xc41f6441
const float rb5_erf  = -1.0250950928e+03f; // 0xc480230b
const float rb6_erf  = -4.8351919556e+02f; // 0xc3f1c275
const float sb1_erf  =  3.0338060379e+01f; // 0x41f2b459
const float sb2_erf  =  3.2579251099e+02f; // 0x43a2e571
const float sb3_erf  =  1.5367296143e+03f; // 0x44c01759
const float sb4_erf  =  3.1998581543e+03f; // 0x4547fdbb
const float sb5_erf  =  2.5530502930e+03f; // 0x451f90ce
const float sb6_erf  =  4.7452853394e+02f; // 0x43ed43a7
const float sb7_erf  = -2.2440952301e+01f; // 0xc1b38712

float erf_f32(float x) {
    const uint hx = floatBitsToUint(x);
    const uint ix = hx & 0x7fffffffu;
    const bool neg = (hx >> 31) != 0u;

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
        return x + x*(r/s);
    }

    float y;

    if (ix < 0x3fa00000u) {
        // 0.84375 <= |x| < 1.25
        const float s = abs(x) - 1.0f;
        const float p = pa0_erf + s*(pa1_erf + s*(pa2_erf + s*(pa3_erf + s*(pa4_erf + s*(pa5_erf + s*pa6_erf)))));
        const float q = 1.0f + s*(qa1_erf + s*(qa2_erf + s*(qa3_erf + s*(qa4_erf + s*(qa5_erf + s*qa6_erf)))));
        y = erx_erf + p/q;
    } else if (ix < 0x40c00000u) {
        // 1.25 <= |x| < 6: erf = 1 - erfc
        const float ax = abs(x);
        const float s  = 1.0f/(ax*ax);
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
        const float z = uintBitsToFloat(floatBitsToUint(ax) & 0xffffe000u);
        const float erfc_x = (exp(-z*z - 0.5625f)*exp((z - ax)*(z + ax) + r/S))/ax;
        y = 1.0f - erfc_x;
    } else {
        // |x| >= 6: 1 - 2^-120 rounds to 1
        y = 1.0f;
    }

    return neg ? -y : y;
}

#endif // GGML_ERF_GLSL
