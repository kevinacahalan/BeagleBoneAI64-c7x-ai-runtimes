/* SPDX-License-Identifier: MIT */
/*
 * End-to-end verification and benchmark suite for ort-c7x-ep:
 *   1. MobileNetV2-style INT8/UINT8 QLinearConv (3x3 s=2, DW 3x3, PW 1x1, 3x3 s=2)
 *      + Dequant + Relu + GlobalAveragePool + Gemm (.onnx binary model)
 *   2. Whisper/Audio-Encoder-style Conv + GELU + LayerNorm + MatMul FFN + Add (.onnx)
 *   3. Standalone QLinearMatMul asymmetric UINT8 x INT8 accuracy & MMA timing
 */

#include "c7x_ort_ep.h"
#include "c7x_rt.h"
#include "c7x_cmd_proto.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <time.h>
#include <vector>

static double now_us(void) {
	struct timespec ts {};
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec * 1e-3;
}

static bool load_bin(const std::string &path, std::vector<uint8_t> &out) {
	std::ifstream f(path, std::ios::binary | std::ios::ate);
	if (!f) return false;
	std::streamsize sz = f.tellg();
	if (sz <= 0) return false;
	f.seekg(0, std::ios::beg);
	out.resize((size_t)sz);
	return (bool)f.read(reinterpret_cast<char *>(out.data()), sz);
}

static void calc_metrics(const float *got, const float *ref, size_t n,
			 float &max_abs, float &cos_sim) {
	double dot = 0.0, na = 0.0, nb = 0.0;
	max_abs = 0.0f;
	for (size_t i = 0; i < n; i++) {
		float d = std::fabs(got[i] - ref[i]);
		if (d > max_abs) max_abs = d;
		dot += (double)got[i] * ref[i];
		na  += (double)got[i] * got[i];
		nb  += (double)ref[i] * ref[i];
	}
	cos_sim = (na > 0.0 && nb > 0.0) ? (float)(dot / (std::sqrt(na) * std::sqrt(nb))) : 1.0f;
}

static int run_onnx_model_test(const std::string &dir, const char *label,
			       const char *onnx_file, const char *in_file, const char *ref_file,
			       float max_err_tol, float min_cos, int bench_iters,
			       int allow_cpu_fallback = 0,
			       uint32_t expect_subgraphs = 1u,
			       uint32_t expect_cpu_nodes = 0u) {
	std::vector<uint8_t> in_bytes, ref_bytes;
	if (!load_bin(dir + "/" + in_file, in_bytes) ||
	    !load_bin(dir + "/" + ref_file, ref_bytes)) {
		fprintf(stderr, "[ort-c7x-ep] missing test files for %s in %s\n", label, dir.c_str());
		return 1;
	}

	/* Verify strict C7x-only mode rejects graphs requiring CPU fallback. */
	if (allow_cpu_fallback && expect_cpu_nodes > 0u) {
		struct c7x_ort_ep_options strict_opts {};
		strict_opts.use_msmc_weights   = 1;
		strict_opts.emu_ddr_bytes      = 64ull * 1024ull * 1024ull;
		strict_opts.allow_cpu_fallback = 0;
		c7x_ort_session_t *bad = c7x_ort_session_create_from_file((dir + "/" + onnx_file).c_str(), &strict_opts);
		if (bad != nullptr) {
			fprintf(stderr, "[ort-c7x-ep] expected strict mode to reject %s\n", onnx_file);
			c7x_ort_session_destroy(bad);
			return 1;
		}
	}

	struct c7x_ort_ep_options opts {};
	opts.use_msmc_weights   = 1;
	opts.enable_profiling   = 0;
	opts.force_hostemu      = 0;
	opts.emu_ddr_bytes      = 64ull * 1024ull * 1024ull;
	opts.allow_cpu_fallback = allow_cpu_fallback;

	c7x_ort_session_t *sess = c7x_ort_session_create_from_file((dir + "/" + onnx_file).c_str(), &opts);
	if (!sess) {
		fprintf(stderr, "[ort-c7x-ep] failed to compile %s\n", onnx_file);
		return 1;
	}

	uint32_t sg = 0u, c7x_n = 0u, cpu_n = 0u;
	c7x_ort_session_partition_info(sess, &sg, &c7x_n, &cpu_n);

	struct c7x_ort_tensor_view iv {}, ov {};
	if (c7x_ort_session_input_info(sess, 0, &iv) != 0 ||
	    c7x_ort_session_output_info(sess, 0, &ov) != 0) {
		fprintf(stderr, "[ort-c7x-ep] failed to query IO info for %s\n", label);
		c7x_ort_session_destroy(sess);
		return 1;
	}
	std::vector<uint8_t> out_bytes(ov.nbytes, 0u);
	iv.data   = in_bytes.data();
	iv.nbytes = in_bytes.size();
	ov.data   = out_bytes.data();
	ov.nbytes = out_bytes.size();

	struct c7x_ort_run_stats st {};
	if (c7x_ort_session_run(sess, &iv, 1, &ov, 1, &st) != 0) {
		fprintf(stderr, "[ort-c7x-ep] inference failed for %s\n", label);
		c7x_ort_session_destroy(sess);
		return 1;
	}

	size_t n_f32 = ref_bytes.size() / sizeof(float);
	float max_abs = 0.0f, cos_sim = 0.0f;
	calc_metrics(reinterpret_cast<const float *>(out_bytes.data()),
		     reinterpret_cast<const float *>(ref_bytes.data()),
		     n_f32, max_abs, cos_sim);

	/* Warm benchmark runs */
	double total_wall_us = 0.0, total_cpu_us = 0.0;
	uint64_t total_dsp_cyc = 0u;
	for (int it = 0; it < bench_iters; it++) {
		struct c7x_ort_run_stats bst {};
		if (c7x_ort_session_run(sess, &iv, 1, &ov, 1, &bst) != 0) {
			c7x_ort_session_destroy(sess);
			return 1;
		}
		total_wall_us += bst.wall_us;
		total_cpu_us  += bst.cpu_fallback_us;
		total_dsp_cyc += bst.dsp_cycles;
	}

	bool ok = (max_abs <= max_err_tol && cos_sim >= min_cos &&
		   sg == expect_subgraphs && cpu_n == expect_cpu_nodes);
	printf("[ort-c7x-ep] %-20s sg=%u cmds=%u cpu=%u max_err=%.6f cos=%.6f dsp=%.3f ms cpu=%.3f ms wall=%.3f ms: %s\n",
	       label, sg, c7x_ort_session_num_cmds(sess), cpu_n, max_abs, cos_sim,
	       (double)total_dsp_cyc / (1e6 * bench_iters),
	       total_cpu_us / (1e3 * bench_iters),
	       total_wall_us / (1e3 * bench_iters),
	       ok ? "PASS" : "FAIL");

	c7x_ort_session_destroy(sess);
	return ok ? 0 : 1;
}

static int test_qlinear_matmul_asymmetric(int bench_iters) {
	/* Direct asymmetric UINT8 x INT8 -> UINT8 QLinearMatMul MMA verification + CPU baseline */
	const uint32_t M = 60, K = 256, N = 256;
	const float a_scale = 0.02f, y_scale = 0.05f;
	const int32_t a_zp = 133, b_zp = -7, y_zp = 120;

	std::vector<uint8_t> A(M * K);
	std::vector<int8_t>  B_nk(N * K);
	std::vector<float>   b_scales(N);
	std::vector<int32_t> bias(N);
	uint32_t rng = 0x55AA1234u;
	auto next_r = [&]() { rng = rng * 1664525u + 1013904223u; return rng; };

	for (auto &v : A) v = (uint8_t)(80u + (next_r() % 96u));
	for (auto &v : B_nk) v = (int8_t)((int)(next_r() % 81u) - 40);
	for (uint32_t n = 0; n < N; n++) {
		b_scales[n] = 0.008f + 0.00003f * (float)n;
		bias[n] = (int32_t)(next_r() % 41u) - 20;
	}

	/* CPU reference + timing */
	std::vector<uint8_t> ref(M * N, 0u);
	double t0 = now_us();
	for (uint32_t m = 0; m < M; m++) {
		for (uint32_t n = 0; n < N; n++) {
			int32_t acc = bias[n];
			for (uint32_t k = 0; k < K; k++) {
				acc += ((int32_t)A[m * K + k] - a_zp) * ((int32_t)B_nk[n * K + k] - b_zp);
			}
			float v = (float)acc * (a_scale * b_scales[n]);
			int32_t q = (int32_t)std::rintf(v / y_scale + (float)y_zp);
			if (q < 0) q = 0;
			if (q > 255) q = 255;
			ref[m * N + n] = (uint8_t)q;
		}
	}
	double cpu_us = now_us() - t0;

	/* C7x MMA via libc7x */
	struct c7x_open_opts ro {};
	ro.transport = C7X_TRANSPORT_AUTO;
	ro.emu_ddr_bytes = 32ull * 1024ull * 1024ull;
	c7x_ctx_t *rt = c7x_open(&ro);
	if (!rt) return 1;

	size_t bsum_off = 0;
	size_t pk_sz = c7x_qlinear_packed_bytes(N, K, &bsum_off);
	std::vector<uint8_t> pk(pk_sz, 0u);
	c7x_pack_qlinear_weights(B_nk.data(), 0, N, K,
				 reinterpret_cast<int8_t *>(pk.data()),
				 reinterpret_cast<int32_t *>(pk.data() + bsum_off));

	c7x_buf_t *b_w  = c7x_alloc(rt, pk_sz, C7X_MEM_MSMC);
	if (!b_w) b_w   = c7x_alloc(rt, pk_sz, C7X_MEM_DDR);
	c7x_buf_t *b_s  = c7x_alloc(rt, N * sizeof(float), C7X_MEM_DDR);
	c7x_buf_t *b_bi = c7x_alloc(rt, N * sizeof(int32_t), C7X_MEM_DDR);
	c7x_buf_t *b_a  = c7x_alloc(rt, M * K, C7X_MEM_DDR);
	c7x_buf_t *b_y  = c7x_alloc(rt, M * N, C7X_MEM_DDR);
	c7x_write(rt, b_w, 0, pk.data(), pk_sz);
	c7x_write(rt, b_s, 0, b_scales.data(), N * sizeof(float));
	c7x_write(rt, b_bi, 0, bias.data(), N * sizeof(int32_t));
	c7x_write(rt, b_a, 0, A.data(), M * K);

	c7x_cmdbuf_t *cb = c7x_cmdbuf_create(rt, 8u, 4096u);
	struct c7x_qlinear_matmul_desc md {};
	md.M = M; md.K = K; md.N = N;
	md.k_pad = K; md.n_pad = N;
	md.a_zp = a_zp - 128; md.b_zp = b_zp; md.y_zp = y_zp;
	md.a_scale = a_scale; md.y_scale = y_scale;
	md.per_channel_b = 1u;
	md.b_sum_addr    = c7x_buf_phys(b_w) + bsum_off;
	md.b_scales_addr = c7x_buf_phys(b_s);

	struct c7x_cmd cmd {};
	cmd.op    = C7X_CMD_QLINEAR_MATMUL;
	cmd.flags = C7X_CMD_FLAG_A_UINT8 | C7X_CMD_FLAG_Y_UINT8 | C7X_CMD_FLAG_BIAS_I32;
	cmd.src0  = c7x_buf_phys(b_w);
	cmd.src1  = c7x_buf_phys(b_a);
	cmd.src2  = c7x_buf_phys(b_bi);
	cmd.src3  = c7x_cmdbuf_push_desc(cb, &md, sizeof(md));
	cmd.dst   = c7x_buf_phys(b_y);
	c7x_cmdbuf_push(cb, &cmd);

	struct c7x_exec_stats st {};
	int rc = c7x_cmdbuf_submit(rt, cb, &st);
	std::vector<uint8_t> got(M * N, 0u);
	if (rc == 0) c7x_read(rt, got.data(), b_y, 0, M * N);

	int max_diff = 0;
	for (size_t i = 0; i < got.size(); i++) {
		int d = std::abs((int)got[i] - (int)ref[i]);
		if (d > max_diff) max_diff = d;
	}

	uint64_t cyc_sum = 0;
	double wall_sum = 0.0;
	for (int it = 0; it < bench_iters; it++) {
		struct c7x_exec_stats bst {};
		c7x_cmdbuf_submit(rt, cb, &bst);
		cyc_sum += bst.cycles;
		wall_sum += bst.wall_us;
	}
	double dsp_ms  = (double)cyc_sum / (1e6 * bench_iters);
	double wall_ms = wall_sum / (1e3 * bench_iters);
	double gops    = (dsp_ms > 0.0) ? ((2.0 * M * K * N) / (dsp_ms * 1e6)) : 0.0;

	bool ok = (rc == 0 && max_diff <= 1);
	printf("[ort-c7x-ep] QLinearMatMul 60x256x256 max_qdiff=%d dsp=%.3f ms (%.1f GOPS) wall=%.3f ms cpu=%.2f ms: %s\n",
	       max_diff, dsp_ms, gops, wall_ms, cpu_us / 1e3, ok ? "PASS" : "FAIL");

	c7x_cmdbuf_destroy(cb);
	c7x_free(rt, b_w); c7x_free(rt, b_s); c7x_free(rt, b_bi);
	c7x_free(rt, b_a); c7x_free(rt, b_y);
	c7x_close(rt);
	return ok ? 0 : 1;
}

static int test_qlinear_unaligned_f32_and_ties(void) {
	/*
	 * Verify non-16-multiple N=70 Y_F32 (unaligned row-1+ vector store on silicon!)
	 * and exact round-to-nearest-even consistency between lanes 0..15 and tail 16..17.
	 */
	const uint32_t M = 5, K = 70, N = 70;
	const uint32_t k_pad = 128, n_pad = 128;
	std::vector<uint8_t> A(M * K);
	std::vector<int8_t>  B_nk(N * K);
	std::vector<float>   b_scales(N, 0.01f);
	uint32_t rng = 0x1234ABCDu;
	auto next_r = [&]() { rng = rng * 1664525u + 1013904223u; return rng; };
	for (auto &v : A) v = (uint8_t)(100u + (next_r() % 55u));
	for (auto &v : B_nk) v = (int8_t)((int)(next_r() % 61u) - 30);

	const int32_t a_zp = 125, b_zp = 4;
	const float a_scale = 0.02f;
	std::vector<float> ref(M * N, 0.0f);
	for (uint32_t m = 0; m < M; m++) {
		for (uint32_t n = 0; n < N; n++) {
			int32_t acc = 0;
			for (uint32_t k = 0; k < K; k++) {
				acc += ((int32_t)A[m * K + k] - a_zp) * ((int32_t)B_nk[n * K + k] - b_zp);
			}
			ref[m * N + n] = (float)acc * (a_scale * b_scales[n]);
		}
	}

	struct c7x_open_opts ro {};
	ro.transport = C7X_TRANSPORT_AUTO;
	ro.emu_ddr_bytes = 32ull * 1024ull * 1024ull;
	c7x_ctx_t *rt = c7x_open(&ro);
	if (!rt) return 1;

	size_t bsum_off = 0;
	size_t pk_sz = c7x_qlinear_packed_bytes(N, K, &bsum_off);
	std::vector<uint8_t> pk(pk_sz, 0u);
	c7x_pack_qlinear_weights(B_nk.data(), 0, N, K,
				 reinterpret_cast<int8_t *>(pk.data()),
				 reinterpret_cast<int32_t *>(pk.data() + bsum_off));

	c7x_buf_t *b_w = c7x_alloc(rt, pk_sz, C7X_MEM_DDR);
	c7x_buf_t *b_s = c7x_alloc(rt, N * sizeof(float), C7X_MEM_DDR);
	c7x_buf_t *b_a = c7x_alloc(rt, M * K, C7X_MEM_DDR);
	c7x_buf_t *b_y = c7x_alloc(rt, M * N * sizeof(float), C7X_MEM_DDR);
	c7x_write(rt, b_w, 0, pk.data(), pk_sz);
	c7x_write(rt, b_s, 0, b_scales.data(), N * sizeof(float));
	c7x_write(rt, b_a, 0, A.data(), M * K);

	c7x_cmdbuf_t *cb = c7x_cmdbuf_create(rt, 8u, 4096u);
	struct c7x_qlinear_matmul_desc md {};
	md.M = M; md.K = K; md.N = N;
	md.k_pad = k_pad; md.n_pad = n_pad;
	md.a_zp = a_zp - 128; md.b_zp = b_zp; md.y_zp = 0;
	md.a_scale = a_scale; md.y_scale = 1.0f;
	md.per_channel_b = 1u;
	md.b_sum_addr    = c7x_buf_phys(b_w) + bsum_off;
	md.b_scales_addr = c7x_buf_phys(b_s);

	struct c7x_cmd cmd {};
	cmd.op    = C7X_CMD_QLINEAR_MATMUL;
	cmd.flags = C7X_CMD_FLAG_A_UINT8 | C7X_CMD_FLAG_Y_F32;
	cmd.src0  = c7x_buf_phys(b_w);
	cmd.src1  = c7x_buf_phys(b_a);
	cmd.src3  = c7x_cmdbuf_push_desc(cb, &md, sizeof(md));
	cmd.dst   = c7x_buf_phys(b_y);
	c7x_cmdbuf_push(cb, &cmd);

	struct c7x_exec_stats st {};
	int rc = c7x_cmdbuf_submit(rt, cb, &st);
	std::vector<float> got(M * N, 0.0f);
	if (rc == 0) c7x_read(rt, got.data(), b_y, 0, M * N * sizeof(float));

	float max_abs = 0.0f, cos_sim = 0.0f;
	calc_metrics(got.data(), ref.data(), M * N, max_abs, cos_sim);
	bool ok = (rc == 0 && max_abs < 1e-5f);

	/* Also verify exact IEEE round-to-nearest-even ties (8.5->8, 9.5->10, -8.5->-8, -9.5->-10)
	 * across vector lanes 0..15 AND scalar tail lanes 16..17 on quantized INT8 output. */
	{
		const uint32_t TM = 2, TK = 1, TN = 18;
		int8_t A_tie[2] = { 1, -1 };
		int8_t B_tie[18] = { 17, 19, 1, 3, 5, 7, -17, -19, 17, 19, 1, 3, 5, 7, -17, -19, 17, 19 };
		int8_t want_tie[2][18] = {
			{  8,  10,  0,  2,  2,  4,  -8, -10,  8,  10,  0,  2,  2,  4,  -8, -10,  8,  10 },
			{ -8, -10,  0, -2, -2, -4,   8,  10, -8, -10,  0, -2, -2, -4,   8,  10, -8, -10 }
		};
		size_t tbsum = 0;
		size_t tpk_sz = c7x_qlinear_packed_bytes(TN, TK, &tbsum);
		std::vector<uint8_t> tpk(tpk_sz, 0u);
		c7x_pack_qlinear_weights(B_tie, 0, TN, TK,
					 reinterpret_cast<int8_t *>(tpk.data()),
					 reinterpret_cast<int32_t *>(tpk.data() + tbsum));
		float one_scale = 1.0f;
		c7x_buf_t *tb_w = c7x_alloc(rt, tpk_sz, C7X_MEM_DDR);
		c7x_buf_t *tb_s = c7x_alloc(rt, sizeof(float), C7X_MEM_DDR);
		c7x_buf_t *tb_a = c7x_alloc(rt, TM * TK, C7X_MEM_DDR);
		c7x_buf_t *tb_y = c7x_alloc(rt, TM * TN, C7X_MEM_DDR);
		c7x_write(rt, tb_w, 0, tpk.data(), tpk_sz);
		c7x_write(rt, tb_s, 0, &one_scale, sizeof(float));
		c7x_write(rt, tb_a, 0, A_tie, sizeof(A_tie));

		c7x_cmdbuf_t *tcb = c7x_cmdbuf_create(rt, 4u, 2048u);
		struct c7x_qlinear_matmul_desc tmd {};
		tmd.M = TM; tmd.K = TK; tmd.N = TN;
		tmd.k_pad = 64; tmd.n_pad = 64;
		tmd.a_scale = 1.0f; tmd.y_scale = 2.0f;
		tmd.per_channel_b = 0u;
		tmd.b_sum_addr = c7x_buf_phys(tb_w) + tbsum;
		tmd.b_scales_addr = c7x_buf_phys(tb_s);

		struct c7x_cmd tcmd {};
		tcmd.op   = C7X_CMD_QLINEAR_MATMUL;
		tcmd.src0 = c7x_buf_phys(tb_w);
		tcmd.src1 = c7x_buf_phys(tb_a);
		tcmd.src3 = c7x_cmdbuf_push_desc(tcb, &tmd, sizeof(tmd));
		tcmd.dst  = c7x_buf_phys(tb_y);
		c7x_cmdbuf_push(tcb, &tcmd);

		struct c7x_exec_stats tst {};
		int trc = c7x_cmdbuf_submit(rt, tcb, &tst);
		int8_t got_tie[2][18] = {};
		if (trc == 0) c7x_read(rt, got_tie, tb_y, 0, sizeof(got_tie));
		if (trc != 0 || std::memcmp(got_tie, want_tie, sizeof(got_tie)) != 0) ok = false;

		c7x_cmdbuf_destroy(tcb);
		c7x_free(rt, tb_w); c7x_free(rt, tb_s); c7x_free(rt, tb_a); c7x_free(rt, tb_y);
	}

	printf("[ort-c7x-ep] QLinearMatMul 5x70x70 Y_F32 unaligned (err=%.6f) + 18-lane ties: %s\n",
	       max_abs, ok ? "PASS" : "FAIL");

	c7x_cmdbuf_destroy(cb);
	c7x_free(rt, b_w); c7x_free(rt, b_s); c7x_free(rt, b_a); c7x_free(rt, b_y);
	c7x_close(rt);
	return ok ? 0 : 1;
}

int main(int argc, char **argv) {
	std::string model_dir = (argc > 1) ? argv[1] : "build-hostemu/models";
	int bench_iters = (argc > 2) ? std::atoi(argv[2]) : 3;
	if (bench_iters < 1) bench_iters = 1;

	int fail = 0;
	/* Note: Python ref uses float64 intermediate rounding vs C7x float32 -> rare 1-LSB INT8 boundary (~4.3e-3 on 4-layer cascade; exact <1e-6 vs fp32 ref). */
	fail |= run_onnx_model_test(model_dir, "mobilenet_qlinear",
				    "mobilenet_qlinear.onnx", "mobilenet_input.bin", "mobilenet_ref.bin",
				    1e-2f, 0.99999f, bench_iters, 0, 1u, 0u);
	fail |= run_onnx_model_test(model_dir, "audio_enc_block",
				    "audio_enc_block.onnx", "audio_input.bin", "audio_ref.bin",
				    1e-4f, 0.99999f, bench_iters, 0, 1u, 0u);
	fail |= run_onnx_model_test(model_dir, "mobilenet_e2e_hyb",
				    "mobilenet_e2e_hybrid.onnx", "mobilenet_e2e_input.bin", "mobilenet_e2e_ref.bin",
				    1e-3f, 0.99999f, bench_iters, 1, 1u, 2u);
	fail |= run_onnx_model_test(model_dir, "whisper_erf_hyb",
				    "whisper_erf_hybrid.onnx", "whisper_erf_input.bin", "whisper_erf_ref.bin",
				    1e-4f, 0.99999f, bench_iters, 1, 3u, 3u);
	fail |= test_qlinear_matmul_asymmetric(bench_iters);
	fail |= test_qlinear_unaligned_f32_and_ties();

	printf("\n=== ORT C7x EP Suite: %s ===\n", fail ? "FAIL" : "ALL PASSED");
	return fail;
}
