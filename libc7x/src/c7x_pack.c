/* SPDX-License-Identifier: MIT */
/*
 * Tile-major weight packers & unpackers for C7x MMA kernels:
 *   - W4A8-g64: [n_pad/64][k_pad/64][32][64] uint8 + [n_pad/64][k_pad/64][64] f32 scales
 *   - W8A8:     [n_pad/64][k_pad/64][64][64] int8  + [n_pad] f32 scales
 */

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "c7x_rt.h"

static inline uint32_t align64_u32(uint32_t x)
{
	return (x + 63u) & ~63u;
}

static inline size_t align64_sz(size_t x)
{
	return (x + 63u) & ~((size_t)63u);
}

static inline float pack_f16_to_f32(uint16_t h)
{
	uint32_t sign = ((uint32_t)(h & 0x8000u)) << 16;
	uint32_t exp  = (h >> 10) & 0x1Fu;
	uint32_t mant = h & 0x03FFu;
	uint32_t bits;
	float f;

	if (exp == 0u) {
		if (mant == 0u) {
			bits = sign;
		} else {
			uint32_t e = 127u - 14u;
			while ((mant & 0x0400u) == 0u) {
				mant <<= 1;
				e--;
			}
			mant &= 0x03FFu;
			bits = sign | (e << 23) | (mant << 13);
		}
	} else if (exp == 31u) {
		bits = sign | 0x7F800000u | (mant << 13);
	} else {
		bits = sign | ((exp + 112u) << 23) | (mant << 13);
	}
	memcpy(&f, &bits, sizeof(f));
	return f;
}

static inline uint16_t pack_f32_to_f16(float f)
{
	uint32_t bits, sign, exp, mant;
	int32_t new_exp;

	memcpy(&bits, &f, sizeof(bits));
	sign = (bits >> 16) & 0x8000u;
	exp  = (bits >> 23) & 0xFFu;
	mant = bits & 0x007FFFFFu;
	if (exp == 255u) {
		return (uint16_t)(sign | 0x7C00u | (mant ? 0x0200u : 0u));
	}
	new_exp = (int32_t)exp - 112;
	if (new_exp >= 31) {
		return (uint16_t)(sign | 0x7C00u);
	}
	if (new_exp <= 0) {
		if (new_exp < -10) return (uint16_t)sign;
		mant = (mant | 0x00800000u) >> (uint32_t)(1 - new_exp);
		mant += 0x00000FFFu + ((mant >> 13) & 1u);
		return (uint16_t)(sign | (mant >> 13));
	}
	mant += 0x00000FFFu + ((mant >> 13) & 1u);
	if (mant & 0x00800000u) {
		mant = 0u;
		if (++new_exp >= 31) return (uint16_t)(sign | 0x7C00u);
	}
	return (uint16_t)(sign | ((uint32_t)new_exp << 10) | (mant >> 13));
}

size_t c7x_w4a8_g64_packed_bytes(uint32_t n, uint32_t k, size_t *scales_off_out)
{
	size_t n_tiles = (size_t)(align64_u32(n) >> 6);
	size_t k_tiles = (size_t)(align64_u32(k) >> 6);
	size_t w_bytes = align64_sz(n_tiles * k_tiles * 2048u);
	size_t s_bytes = align64_sz(n_tiles * k_tiles * 64u * sizeof(float));

	if (scales_off_out) {
		*scales_off_out = w_bytes;
	}
	return w_bytes + s_bytes;
}

size_t c7x_w8a8_packed_bytes(uint32_t n, uint32_t k, size_t *scales_off_out)
{
	size_t n_pad   = (size_t)align64_u32(n);
	size_t n_tiles = n_pad >> 6;
	size_t k_tiles = (size_t)(align64_u32(k) >> 6);
	size_t w_bytes = align64_sz(n_tiles * k_tiles * 4096u);
	size_t s_bytes = align64_sz(n_pad * sizeof(float));

	if (scales_off_out) {
		*scales_off_out = w_bytes;
	}
	return w_bytes + s_bytes;
}

/* Stream-pack one 64-row output tile (`valid_rows` in 1..64) into W4A8-g64. */
void c7x_pack_f32_ntile_to_w4a8_g64(const float *rows_f32, uint32_t valid_rows,
				    uint32_t k, uint32_t nt,
				    uint8_t *w_packed, float *w_scales)
{
	uint32_t k_pad = align64_u32(k);
	uint32_t kt_cnt = k_pad >> 6;
	uint32_t kt, r, c;

	if (valid_rows > 64u) {
		valid_rows = 64u;
	}
	for (kt = 0u; kt < kt_cnt; kt++) {
		uint8_t *tile = w_packed + ((size_t)nt * kt_cnt + kt) * 2048u;
		float *ts     = w_scales + ((size_t)nt * kt_cnt + kt) * 64u;

		memset(tile, 0, 2048u);
		for (r = 0u; r < 64u; r++) {
			float amax = 0.0f;
			float sc, inv;
			uint8_t *vec = tile + (r >> 1) * 64u;

			if (r < valid_rows) {
				const float *src = rows_f32 + (size_t)r * k;
				for (c = 0u; c < 64u; c++) {
					uint32_t col = kt * 64u + c;
					if (col < k) {
						float v = fabsf(src[col]);
						if (v > amax) amax = v;
					}
				}
			}
			sc = amax / 7.0f;
			ts[r] = sc;
			inv = (sc > 0.0f) ? (1.0f / sc) : 0.0f;

			if (r < valid_rows && sc > 0.0f) {
				const float *src = rows_f32 + (size_t)r * k;
				for (c = 0u; c < 64u; c++) {
					uint32_t col = kt * 64u + c;
					int q = 0;
					uint8_t nib;
					if (col < k) {
						q = (int)lrintf(src[col] * inv);
						if (q > 7) q = 7;
						if (q < -7) q = -7;
					}
					nib = (uint8_t)(q & 0x0F);
					if ((r & 1u) == 0u) {
						vec[c] = nib;
					} else {
						vec[c] |= (uint8_t)(nib << 4);
					}
				}
			}
		}
	}
}

void c7x_pack_f32_to_w4a8_g64(const float *w_f32, uint32_t n, uint32_t k,
			      uint8_t *w_packed, float *w_scales)
{
	uint32_t n_pad = align64_u32(n);
	uint32_t nt_cnt = n_pad >> 6;
	uint32_t nt;

	for (nt = 0u; nt < nt_cnt; nt++) {
		uint32_t row0 = nt * 64u;
		uint32_t valid = (n > row0) ? (n - row0) : 0u;
		if (valid > 64u) valid = 64u;
		c7x_pack_f32_ntile_to_w4a8_g64(w_f32 + (size_t)row0 * k, valid,
					       k, nt, w_packed, w_scales);
	}
}

/* Stream-pack one 64-row output tile (`valid_rows` in 1..64) into W8A8. */
void c7x_pack_f32_ntile_to_w8a8(const float *rows_f32, uint32_t valid_rows,
				uint32_t k, uint32_t nt,
				int8_t *w_tiled, float *w_scales)
{
	uint32_t k_pad = align64_u32(k);
	uint32_t kt_cnt = k_pad >> 6;
	uint32_t r, kt, c;

	if (valid_rows > 64u) {
		valid_rows = 64u;
	}
	for (kt = 0u; kt < kt_cnt; kt++) {
		memset(w_tiled + ((size_t)nt * kt_cnt + kt) * 4096u, 0, 4096u);
	}
	for (r = 0u; r < 64u; r++) {
		uint32_t row = nt * 64u + r;
		float amax = 0.0f;
		float sc, inv;

		if (r < valid_rows) {
			const float *src = rows_f32 + (size_t)r * k;
			for (c = 0u; c < k; c++) {
				float v = fabsf(src[c]);
				if (v > amax) amax = v;
			}
		}
		sc = amax / 127.0f;
		w_scales[row] = sc;
		inv = (sc > 0.0f) ? (1.0f / sc) : 0.0f;

		if (r < valid_rows && sc > 0.0f) {
			const float *src = rows_f32 + (size_t)r * k;
			for (kt = 0u; kt < kt_cnt; kt++) {
				int8_t *tile = w_tiled + ((size_t)nt * kt_cnt + kt) * 4096u;
				for (c = 0u; c < 64u; c++) {
					uint32_t col = kt * 64u + c;
					if (col < k) {
						int q = (int)lrintf(src[col] * inv);
						if (q > 127) q = 127;
						if (q < -127) q = -127;
						tile[r * 64u + c] = (int8_t)q;
					}
				}
			}
		}
	}
}

void c7x_pack_f32_to_w8a8(const float *w_f32, uint32_t n, uint32_t k,
			  int8_t *w_tiled, float *w_scales)
{
	uint32_t n_pad = align64_u32(n);
	uint32_t nt_cnt = n_pad >> 6;
	uint32_t nt;

	for (nt = 0u; nt < nt_cnt; nt++) {
		uint32_t row0 = nt * 64u;
		uint32_t valid = (n > row0) ? (n - row0) : 0u;
		if (valid > 64u) valid = 64u;
		c7x_pack_f32_ntile_to_w8a8(w_f32 + (size_t)row0 * k, valid,
					   k, nt, w_tiled, w_scales);
	}
}

void c7x_pack_q4_0_to_w4a8_g64(const void *q4_0_data, uint32_t n, uint32_t k,
			       uint8_t *w_packed, float *w_scales)
{
	const uint8_t *src = (const uint8_t *)q4_0_data;
	uint32_t n_pad = align64_u32(n);
	uint32_t k_pad = align64_u32(k);
	uint32_t nt_cnt = n_pad >> 6;
	uint32_t kt_cnt = k_pad >> 6;
	size_t row_bytes = (size_t)(k / 32u) * 18u;
	uint32_t nt, kt, r, j;
	float grp[64];

	for (nt = 0u; nt < nt_cnt; nt++) {
		for (kt = 0u; kt < kt_cnt; kt++) {
			uint8_t *tile = w_packed + ((size_t)nt * kt_cnt + kt) * 2048u;
			float *ts     = w_scales + ((size_t)nt * kt_cnt + kt) * 64u;

			memset(tile, 0, 2048u);
			for (r = 0u; r < 64u; r++) {
				uint32_t row = nt * 64u + r;
				float amax = 0.0f;
				float sc, inv;
				uint8_t *vec = tile + (r >> 1) * 64u;
				uint32_t blk0 = kt * 2u;
				uint32_t b;

				if (row >= n) {
					ts[r] = 0.0f;
					continue;
				}
				memset(grp, 0, sizeof(grp));
				for (b = 0u; b < 2u; b++) {
					uint32_t bi = blk0 + b;
					if ((bi + 1u) * 32u <= k) {
						const uint8_t *blk = src + (size_t)row * row_bytes + (size_t)bi * 18u;
						uint16_t dh = (uint16_t)blk[0] | ((uint16_t)blk[1] << 8);
						float d = pack_f16_to_f32(dh);
						const uint8_t *qs = blk + 2u;
						for (j = 0u; j < 16u; j++) {
							float v0 = (float)((int)(qs[j] & 0x0Fu) - 8) * d;
							float v1 = (float)((int)(qs[j] >> 4) - 8) * d;
							grp[b * 32u + j]       = v0;
							grp[b * 32u + j + 16u] = v1;
							if (fabsf(v0) > amax) amax = fabsf(v0);
							if (fabsf(v1) > amax) amax = fabsf(v1);
						}
					}
				}
				sc = amax / 7.0f;
				ts[r] = sc;
				inv = (sc > 0.0f) ? (1.0f / sc) : 0.0f;
				if (sc > 0.0f) {
					for (j = 0u; j < 64u; j++) {
						int q = (int)lrintf(grp[j] * inv);
						uint8_t nib;
						if (q > 7) q = 7;
						if (q < -7) q = -7;
						nib = (uint8_t)(q & 0x0F);
						if ((r & 1u) == 0u) {
							vec[j] = nib;
						} else {
							vec[j] |= (uint8_t)(nib << 4);
						}
					}
				}
			}
		}
	}
}

/* Zero-malloc 2-pass Q8_0 -> W8A8 tile packer */
void c7x_pack_q8_0_to_w8a8(const void *q8_0_data, uint32_t n, uint32_t k,
			   int8_t *w_tiled, float *w_scales)
{
	const uint8_t *src = (const uint8_t *)q8_0_data;
	uint32_t n_pad = align64_u32(n);
	uint32_t k_pad = align64_u32(k);
	uint32_t kt_cnt = k_pad >> 6;
	uint32_t nb = k / 32u;
	size_t row_bytes = (size_t)nb * 34u;
	uint32_t row, b, j;

	memset(w_tiled, 0, (size_t)(n_pad >> 6) * kt_cnt * 4096u);
	for (row = 0u; row < n_pad; row++) {
		float amax = 0.0f, sc, inv;
		uint32_t nt = row >> 6;
		uint32_t r  = row & 63u;

		if (row >= n) {
			w_scales[row] = 0.0f;
			continue;
		}
		/* Pass 1: find row max */
		for (b = 0u; b < nb; b++) {
			const uint8_t *blk = src + (size_t)row * row_bytes + (size_t)b * 34u;
			uint16_t dh = (uint16_t)blk[0] | ((uint16_t)blk[1] << 8);
			float d = fabsf(pack_f16_to_f32(dh));
			const int8_t *qs = (const int8_t *)(blk + 2u);
			int qmax = 0;
			for (j = 0u; j < 32u; j++) {
				int a = qs[j] < 0 ? -qs[j] : qs[j];
				if (a > qmax) qmax = a;
			}
			if ((float)qmax * d > amax) amax = (float)qmax * d;
		}
		sc = amax / 127.0f;
		w_scales[row] = sc;
		inv = (sc > 0.0f) ? (1.0f / sc) : 0.0f;
		/* Pass 2: requantize directly into tiles */
		if (sc > 0.0f) {
			for (b = 0u; b < nb; b++) {
				const uint8_t *blk = src + (size_t)row * row_bytes + (size_t)b * 34u;
				uint16_t dh = (uint16_t)blk[0] | ((uint16_t)blk[1] << 8);
				float ratio = pack_f16_to_f32(dh) * inv;
				const int8_t *qs = (const int8_t *)(blk + 2u);
				uint32_t kt = b >> 1;
				uint32_t col_base = (b & 1u) * 32u;
				int8_t *tile = w_tiled + ((size_t)nt * kt_cnt + kt) * 4096u;
				for (j = 0u; j < 32u; j++) {
					int q = (int)lrintf((float)qs[j] * ratio);
					if (q > 127) q = 127;
					if (q < -127) q = -127;
					tile[r * 64u + col_base + j] = (int8_t)q;
				}
			}
		}
	}
}

/* Zero-malloc 2-pass F16 -> W8A8 tile packer */
void c7x_pack_f16_to_w8a8(const uint16_t *f16_data, uint32_t n, uint32_t k,
			  int8_t *w_tiled, float *w_scales)
{
	uint32_t n_pad = align64_u32(n);
	uint32_t k_pad = align64_u32(k);
	uint32_t kt_cnt = k_pad >> 6;
	uint32_t row, kt, c;

	memset(w_tiled, 0, (size_t)(n_pad >> 6) * kt_cnt * 4096u);
	for (row = 0u; row < n_pad; row++) {
		float amax = 0.0f, sc, inv;
		uint32_t nt = row >> 6;
		uint32_t r  = row & 63u;

		if (row >= n) {
			w_scales[row] = 0.0f;
			continue;
		}
		for (c = 0u; c < k; c++) {
			float v = fabsf(pack_f16_to_f32(f16_data[(size_t)row * k + c]));
			if (v > amax) amax = v;
		}
		sc = amax / 127.0f;
		w_scales[row] = sc;
		inv = (sc > 0.0f) ? (1.0f / sc) : 0.0f;
		if (sc > 0.0f) {
			for (kt = 0u; kt < kt_cnt; kt++) {
				int8_t *tile = w_tiled + ((size_t)nt * kt_cnt + kt) * 4096u;
				for (c = 0u; c < 64u; c++) {
					uint32_t col = kt * 64u + c;
					if (col < k) {
						float v = pack_f16_to_f32(f16_data[(size_t)row * k + col]);
						int q = (int)lrintf(v * inv);
						if (q > 127) q = 127;
						if (q < -127) q = -127;
						tile[r * 64u + c] = (int8_t)q;
					}
				}
			}
		}
	}
}

/* -------------------------------------------------------------------------- */
/* Unpackers (for lossless get_tensor / partial set_tensor on repacked weights)*/
/* -------------------------------------------------------------------------- */

void c7x_unpack_w4a8_g64_row_f32(const uint8_t *w_packed, const float *w_scales,
				 uint32_t row, uint32_t k, float *out_f32)
{
	uint32_t k_pad = align64_u32(k);
	uint32_t kt_cnt = k_pad >> 6;
	uint32_t nt = row >> 6;
	uint32_t r  = row & 63u;
	uint32_t kt, c;

	for (kt = 0u; kt < kt_cnt; kt++) {
		const uint8_t *tile = w_packed + ((size_t)nt * kt_cnt + kt) * 2048u;
		const uint8_t *vec  = tile + (r >> 1) * 64u;
		float sc = w_scales[((size_t)nt * kt_cnt + kt) * 64u + r];
		uint32_t col0 = kt << 6;

		for (c = 0u; c < 64u; c++) {
			if (col0 + c < k) {
				uint8_t byte = vec[c];
				int8_t w = (r & 1u)
					   ? ((int8_t)byte >> 4)
					   : ((int8_t)(byte << 4) >> 4);
				out_f32[col0 + c] = (float)w * sc;
			}
		}
	}
}

void c7x_unpack_w8a8_row_f32(const int8_t *w_tiled, const float *w_scales,
			     uint32_t row, uint32_t k, float *out_f32)
{
	uint32_t k_pad = align64_u32(k);
	uint32_t kt_cnt = k_pad >> 6;
	uint32_t nt = row >> 6;
	uint32_t r  = row & 63u;
	float sc = w_scales[row];
	uint32_t kt, c;

	for (kt = 0u; kt < kt_cnt; kt++) {
		const int8_t *tile = w_tiled + ((size_t)nt * kt_cnt + kt) * 4096u;
		const int8_t *src  = tile + r * 64u;
		uint32_t col0 = kt << 6;
		for (c = 0u; c < 64u; c++) {
			if (col0 + c < k) {
				out_f32[col0 + c] = (float)src[c] * sc;
			}
		}
	}
}

void c7x_unpack_w4a8_g64_to_q4_0(const uint8_t *w_packed, const float *w_scales,
				 uint32_t n, uint32_t k, void *q4_0_out)
{
	uint8_t *dst = (uint8_t *)q4_0_out;
	uint32_t nb = k / 32u;
	size_t row_bytes = (size_t)nb * 18u;
	uint32_t k_pad = align64_u32(k);
	uint32_t kt_cnt = k_pad >> 6;
	uint32_t row, b, j;

	for (row = 0u; row < n; row++) {
		uint32_t nt = row >> 6;
		uint32_t r  = row & 63u;
		for (b = 0u; b < nb; b++) {
			uint32_t kt = b >> 1;
			uint32_t off = (b & 1u) * 32u;
			const uint8_t *tile = w_packed + ((size_t)nt * kt_cnt + kt) * 2048u;
			const uint8_t *vec  = tile + (r >> 1) * 64u;
			float sc = w_scales[((size_t)nt * kt_cnt + kt) * 64u + r];
			uint16_t dh = pack_f32_to_f16(sc);
			uint8_t *blk = dst + (size_t)row * row_bytes + (size_t)b * 18u;

			blk[0] = (uint8_t)(dh & 0xFFu);
			blk[1] = (uint8_t)(dh >> 8);
			for (j = 0u; j < 16u; j++) {
				uint8_t b0 = vec[off + j];
				uint8_t b1 = vec[off + j + 16u];
				int w0 = (r & 1u) ? ((int8_t)b0 >> 4) : ((int8_t)(b0 << 4) >> 4);
				int w1 = (r & 1u) ? ((int8_t)b1 >> 4) : ((int8_t)(b1 << 4) >> 4);
				int q0 = w0 + 8;
				int q1 = w1 + 8;
				if (q0 < 0) {
					q0 = 0;
				}
				if (q0 > 15) {
					q0 = 15;
				}
				if (q1 < 0) {
					q1 = 0;
				}
				if (q1 > 15) {
					q1 = 15;
				}
				blk[2u + j] = (uint8_t)((q0 & 0x0F) | ((q1 & 0x0F) << 4));
			}
		}
	}
}

void c7x_unpack_w8a8_to_q8_0(const int8_t *w_tiled, const float *w_scales,
			     uint32_t n, uint32_t k, void *q8_0_out)
{
	uint8_t *dst = (uint8_t *)q8_0_out;
	uint32_t nb = k / 32u;
	size_t row_bytes = (size_t)nb * 34u;
	uint32_t k_pad = align64_u32(k);
	uint32_t kt_cnt = k_pad >> 6;
	uint32_t row, b, j;

	for (row = 0u; row < n; row++) {
		uint32_t nt = row >> 6;
		uint32_t r  = row & 63u;
		uint16_t dh = pack_f32_to_f16(w_scales[row]);

		for (b = 0u; b < nb; b++) {
			uint32_t kt = b >> 1;
			uint32_t off = (b & 1u) * 32u;
			const int8_t *tile = w_tiled + ((size_t)nt * kt_cnt + kt) * 4096u;
			const int8_t *src  = tile + r * 64u + off;
			uint8_t *blk = dst + (size_t)row * row_bytes + (size_t)b * 34u;

			blk[0] = (uint8_t)(dh & 0xFFu);
			blk[1] = (uint8_t)(dh >> 8);
			for (j = 0u; j < 32u; j++) {
				((int8_t *)(blk + 2u))[j] = src[j];
			}
		}
	}
}
