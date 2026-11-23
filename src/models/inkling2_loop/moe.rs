//! Inkling 2 Mini's routed experts as one grouped call per projection, with
//! the routing decided and the rows grouped on the device.
//!
//! The per-expert loop this replaces ran, per layer visit, a router GEMM whose
//! `[t, 128]` logits were downloaded so the host could pick the top 8 of 128
//! and build a `BTreeMap` of rows per expert, then about 120 iterations of
//! `select`, gate, up, the softcapped SwiGLU, down, a weight multiply and a
//! `select_assign`, each a handful of launches. Profiled on sky at a
//! 4,096-token chunk (a profile of 2026-11-23): 7-8 s of the loop, 3.9 s
//! of router download and 2.0 s of host routing per chunk.
//!
//! Here one visit is a fixed sequence of launches, whatever the routing:
//!
//! ```text
//! logits  = router(x)                              one BF16 GEMM, stays on the device
//! topk    = router_topk(logits, bias, route_scale) inkling::routetopk, the same rule as
//!                                                  route_from_logits (sigmoid + bias picks,
//!                                                  the normalised sigmoid of the picks
//!                                                  times route_scale weighs)
//! plan    = select, rank, place                    three small kernels, below
//! a       = gather(x, plan.row_src)                [m_max, H] BF16, padding rows zero
//! g, u    = grouped(a, gate), grouped(a, up)       one launch each over every active expert
//! s       = softcapped_swiglu(g, u)                [m_max, I] BF16
//! y       = grouped(s, down)                       one launch
//! out     = scatter(y, plan)                       per row, its picks in ascending expert id
//! ```
//!
//! # The row contract (for a caller batching several contexts)
//!
//! [`Bank::forward`] takes rows, not tokens: `x` is `[rows, H]`, and nothing
//! below knows where a row came from. Each routed row carries the index of its
//! source row in `x` (`row_src`), and the scatter writes each source row's
//! weighted sum back to that row of the `[rows, H]` output. A caller that
//! concatenates the rows of several contexts gets each context's rows back in
//! the same places; the kernels see no context identity.
//!
//! # The plan
//!
//! Rows are stacked by ascending expert id, each expert's rows in ascending
//! source row, and each expert padded to whole 16-row MMA tiles. Those are the
//! order and padding the per-expert loop had (`BTreeMap` order, then
//! `select` in row order), and ascending expert id per source row is also the
//! order the loop's chain of `select_assign(Add)` summed in. The count of
//! active tiles is data, so the GEMMs launch over an upper bound
//! (`max_tiles`) and a tile with no expert (`u32::MAX` in `tile_slot`) ends at
//! once. Nothing about the plan crosses to the host.
//!
//! * `select`: one unit per (row, expert) writes `pick + 1` or 0 into a
//!   `[rows, experts]` table, and the stacked buffer's padding and every
//!   scatter slot (`tok_rows`, to stacked row 0) are initialised.
//! * `rank`: one cube, one unit per expert, walks the rows in order and gives
//!   each pick its place among that expert's rows; then the tile prefix sum
//!   and the per-tile expert table.
//! * `place`: one unit per pick writes its stacked row's source and weight,
//!   and its place in its source row's ascending-id list for the scatter.
//!
//! # Weights
//!
//! A layer's experts are bound as one buffer per projection, experts stacked
//! `[E, n, k]`, so expert `e`'s weight row `r` is global row `e * n + r` and
//! needs no offset table. One copy on the device.
//!
//! # Precision
//!
//! The experts run at the precision they were trained in, and there is no
//! other arm. Stated by the release's `hf_quant_config.json` (layers 0..19):
//! NVFP4 weights, and NVFP4 activations into both products (E2M1, an E4M3
//! scale per 16, round to nearest, an outer scale per token, Four Over Six
//! with the MAE rule). Stated (layers 20..23): MXFP8 weights and activations
//! (E4M3, a UE8M0 scale per 32, round to nearest). Stated by the Four Over Six
//! paper or its reference code: the outer scale `amax / (6 * 256)`, errors
//! summed in f32, a tie keeping 6, E2M1 ties to even.
//!
//! Inferred, with no source that states it for this model: the MXFP8 activation
//! scale (the OCP MX v1.0 rule), BF16 for the values that enter a quantizer,
//! f32 for the MMA accumulator, the SwiGLU and the weighted scatter.
//!
//! Not reproduced: an FP8 MMA (layers 20..23 multiply MXFP8-rounded operands
//! in a BF16 GEMM, whose accumulator and summation order are not shown to
//! match one), and the compressed transport the training run's combine used.

use burn::tensor::Tensor;
use cubecl::prelude::*;
use cubecl::server::Handle;
use half::bf16;

use crate::models::inkling::bf16gemm::{Bf16W, linear_bf16 as linear_bf16_handle, linear_bf16_narrow};
use crate::models::inkling::fp4quant::{qdq_mxfp8_bf16, quantize_nvfp4_46_bf16};
use crate::models::inkling::moegroup::{
    BlockPlanDev, bf16_linear_grouped_launch, fp4_linear_grouped_planes_launch_as, gather_grouped_bf16,
    gather_grouped_bf16_from_bf16, scatter_weighted,
};
use crate::models::inkling::routetopk::router_topk_launch;
use crate::models::inkling::seam::{Bk, client_of, handle_of_any, tensor_of};
use crate::models::inkling::w4a16gemm::{GROUP, MTILE};

type Client = ComputeClient<cubecl::cuda::CudaRuntime>;
type Dev = burn::backend::cuda::CudaDevice;

/// The tile-table value of a tile no expert owns (`0xFFFF_FFFF` in the kernels).
pub const NO_EXPERT: u32 = u32::MAX;
const CUBE: u32 = 256;

/// One projection of every expert of a layer, stacked `[E, n, k]`.
pub(super) enum Stack {
    /// NVFP4: codes `[E * n, k / 2]` bytes, E4M3 scales `[E * n, k / 16]`,
    /// and each expert's second-level scale `[E]` f32. `off` is `[2E]` u64:
    /// expert `e`'s first code byte and first scale byte, the table
    /// [`fp4_linear_grouped_planes_launch_as`] reads.
    Nvfp4 { codes: Handle, scales: Handle, scale2: Handle, off: Handle },
    /// BF16 (MXFP8 decoded exactly): `[E * n, k]`, and the per-expert element
    /// offsets `[E]` u64 that [`bf16_linear_grouped_launch`] indexes.
    Bf16 { w: Handle, off: Handle },
}

pub(super) struct Proj {
    pub stack: Stack,
    pub n: usize,
    pub k: usize,
}

/// A layer's router and routed experts.
pub struct Bank {
    router: Bf16W,
    /// `[E]` f32 on the device.
    bias: Handle,
    #[cfg(test)]
    bias_host: Vec<f32>,
    gate: Proj,
    up: Proj,
    down: Proj,
    experts: usize,
    top_k: usize,
    hidden: usize,
    inter: usize,
    route_scale: f32,
    gate_softcap: f32,
    up_softcap: f32,
}

/// The routing a visit used, on the device; read it only if you want it.
pub struct Route {
    /// `[rows, E]` f32 router logits.
    pub logits: Tensor<Bk, 2>,
    topk: Handle,
    rows: usize,
    top_k: usize,
}

impl Route {
    pub fn logits(&self) -> Vec<f32> {
        self.logits.clone().into_data().to_vec().expect("router logits are f32")
    }

    /// Each row's picks, highest selection score first, and their weights
    /// (route_scale included), as [`crate::models::inkling::block::route_from_logits`]
    /// orders them.
    pub fn picks(&self) -> Vec<(Vec<usize>, Vec<f32>)> {
        let client = client_of(&self.logits);
        let w = 2 * self.top_k + 1;
        let bytes = client.read_one(self.topk.clone()).expect("read the router's top-k");
        let v = f32::from_bytes(&bytes);
        (0..self.rows)
            .map(|r| {
                let row = &v[r * w..(r + 1) * w];
                let flag = row[w - 1];
                assert!(flag == 0.0, "router logit is non-finite at row {r}, logit {}", flag as usize - 1);
                ((0..self.top_k).map(|j| row[j] as usize).collect(), row[self.top_k..2 * self.top_k].to_vec())
            })
            .collect()
    }
}

/// The device plan of one visit. `m_max = 16 * max_tiles` stacked rows.
pub struct Plan {
    pub row_src: Handle,
    pub row_w: Handle,
    pub tile_slot: Handle,
    pub tok_rows: Handle,
    pub tok_cnt: Handle,
    pub blk: BlockPlanDev,
    pub max_tiles: usize,
}

/// The most 16-row tiles `rows * top_k` picks over `experts` experts can
/// occupy: each active expert pads to a whole tile, and every tile holds at
/// least one pick.
pub fn max_tiles(rows: usize, top_k: usize, experts: usize) -> usize {
    let picks = rows * top_k;
    let active = experts.min(picks);
    picks.min((picks + (MTILE - 1) * active) / MTILE)
}

#[cube(launch_unchecked)]
fn route_select(
    topk: &Array<f32>,
    sel: &mut Array<u32>,
    row_src: &mut Array<i32>,
    row_w: &mut Array<f32>,
    tok_rows: &mut Array<u32>,
    rows: u32,
    m_max: u32,
    #[comptime] experts: u32,
    #[comptime] top_k: u32,
) {
    let p = ABSOLUTE_POS as u32;
    let width = comptime!(2 * top_k + 1);
    // A row with at least `top_k` non-NaN logits has distinct picks, so
    // `route_place` writes every one of its `top_k` scatter slots. A row with
    // fewer repeats expert 0 (`router_topk` never picks a NaN) and leaves
    // slots unwritten; stacked row 0 always exists, so those slots read an
    // in-range row. Such a row is a router fault either way.
    if p < rows * top_k {
        tok_rows[p as usize] = 0u32;
    }
    if p < rows * experts {
        let r = p / experts;
        let e = p % experts;
        let mut s = 0u32;
        for j in 0..top_k {
            if u32::cast_from(topk[(r * width + j) as usize]) == e {
                s = j + 1u32;
            }
        }
        sel[p as usize] = s;
    }
    if p < m_max {
        row_src[p as usize] = -1i32;
        row_w[p as usize] = 0.0f32;
    }
}

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn route_rank(
    sel: &Array<u32>,
    rank: &mut Array<u32>,
    tile_base: &mut Array<u32>,
    tile_slot: &mut Array<u32>,
    blk_slot: &mut Array<u32>,
    blk_tile0: &mut Array<u32>,
    blk_cnt: &mut Array<u32>,
    rows: u32,
    max_tiles: u32,
    #[comptime] experts: u32,
) {
    let e = UNIT_POS_X;
    let mut tiles = SharedMemory::<u32>::new(comptime!(experts as usize + 1));
    let mut c = 0u32;
    for r in 0..rows {
        let i = (r * experts + e) as usize;
        if sel[i] != 0u32 {
            rank[i] = c;
            c += 1u32;
        }
    }
    tiles[e as usize] = (c + 15u32) / 16u32;
    sync_cube();
    if e == 0u32 {
        let mut acc = 0u32;
        for x in 0..experts {
            let n = tiles[x as usize];
            tiles[x as usize] = acc;
            acc += n;
        }
        tiles[experts as usize] = acc;
    }
    sync_cube();
    let base = tiles[e as usize];
    let end = tiles[(e + 1u32) as usize];
    tile_base[e as usize] = base;
    for q in base..end {
        tile_slot[q as usize] = e;
        blk_slot[q as usize] = e;
        blk_tile0[q as usize] = q;
        blk_cnt[q as usize] = 1u32;
    }
    let total = tiles[experts as usize];
    let iters = (max_tiles - total + experts - 1u32) / experts;
    for i in 0..iters {
        let q = total + e + i * experts;
        if q < max_tiles {
            tile_slot[q as usize] = 0xFFFF_FFFFu32;
            blk_slot[q as usize] = 0u32;
            blk_tile0[q as usize] = q;
            blk_cnt[q as usize] = 0u32;
        }
    }
}

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn route_place(
    topk: &Array<f32>,
    rank: &Array<u32>,
    tile_base: &Array<u32>,
    row_src: &mut Array<i32>,
    row_w: &mut Array<f32>,
    tok_rows: &mut Array<u32>,
    tok_cnt: &mut Array<u32>,
    fault: &mut Array<u32>,
    rows: u32,
    #[comptime] experts: u32,
    #[comptime] top_k: u32,
) {
    let p = ABSOLUTE_POS as u32;
    let width = comptime!(2 * top_k + 1);
    if p < rows * top_k {
        let r = p / top_k;
        let j = p % top_k;
        let rb = r * width;
        let e = u32::cast_from(topk[(rb + j) as usize]);
        let row = tile_base[e as usize] * 16u32 + rank[(r * experts + e) as usize];
        row_src[row as usize] = i32::cast_from(r);
        row_w[row as usize] = topk[(rb + top_k + j) as usize];
        // this pick's place among its row's picks in ascending expert id: the
        // order the per-expert loop's select_assign chain added them in
        let mut q = 0u32;
        for o in 0..top_k {
            if u32::cast_from(topk[(rb + o) as usize]) < e {
                q += 1u32;
            }
        }
        tok_rows[(r * top_k + q) as usize] = row;
        if j == 0u32 {
            tok_cnt[r as usize] = top_k;
            let flag = u32::cast_from(topk[(rb + width - 1u32) as usize]);
            if flag != 0u32 {
                fault[0] = flag;
            }
        }
    }
}

/// `out = act(gate) * up` per stacked row, Mini's softcapped SwiGLU, into BF16
/// for the down projection. Rows of tiles no expert owns are zero: no GEMM
/// wrote their `g` and `u`, so those are not read, but the trained precision
/// quantizes every row of `out`, so each must hold a value. When
/// `scaled`, `g` and `u` are first multiplied by their row's `alpha` (the
/// per-token outer scale the W4A4 GEMM leaves out).
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn softcapped_swiglu(
    g: &Array<f32>,
    u: &Array<f32>,
    tile_slot: &Array<u32>,
    alpha: &Array<f32>,
    out: &mut Array<bf16>,
    total: u32,
    gate_cap: f32,
    up_cap: f32,
    #[comptime] inter: u32,
    #[comptime] scaled: bool,
) {
    let p = ABSOLUTE_POS as u32;
    if p < total {
        let tile = p / comptime!(inter * 16);
        if tile_slot[tile as usize] != 0xFFFF_FFFFu32 {
            let mut gv = g[p as usize];
            let mut uv = u[p as usize];
            if comptime!(scaled) {
                // the row's per-token outer scale of its NVFP4 activation
                let a = alpha[(p / comptime!(inter)) as usize];
                gv *= a;
                uv *= a;
            }
            let cg = Tanh::tanh(gv / gate_cap) * gate_cap;
            let cu = Tanh::tanh(uv / up_cap) * up_cap;
            // sigmoid without overflow: e = exp(-|g|), 1 / (1 + e) or e / (1 + e)
            let e = Exp::exp(-Abs::abs(gv));
            let mut s = e / (1.0f32 + e);
            if gv >= 0.0f32 {
                s = 1.0f32 / (1.0f32 + e);
            }
            out[p as usize] = bf16::cast_from(cg * s * cu);
        } else {
            out[p as usize] = bf16::cast_from(0.0f32);
        }
    }
}

fn u32s(client: &Client, v: &[u32]) -> Handle {
    client.create_from_slice(u32::as_bytes(v))
}

impl Bank {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        client: &Client,
        router: Bf16W,
        bias: Vec<f32>,
        [gate, up, down]: [Proj; 3],
        top_k: usize,
        route_scale: f32,
        gate_softcap: f32,
        up_softcap: f32,
    ) -> Bank {
        let experts = bias.len();
        Bank {
            router,
            bias: client.create_from_slice(f32::as_bytes(&bias)),
            #[cfg(test)]
            bias_host: bias,
            hidden: gate.k,
            inter: gate.n,
            gate,
            up,
            down,
            experts,
            top_k,
            route_scale,
            gate_softcap,
            up_softcap,
        }
    }

    /// The router, then the experts: `[rows, H]` f32, and the routing on the
    /// device.
    ///
    /// `fault` is the caller's cell ([`fault_cell`]): the plan writes
    /// `1 + logit` into it when a row's router logits are not finite, its form
    /// of `route_from_logits`' panic, and nothing else reads or clears it. The
    /// read is a sync, so it is the caller's to place ([`read_fault`]). The cell belongs to
    /// whoever owns the call's state, never to the bank, so one context's
    /// fault cannot reach another context sharing these weights.
    pub fn forward(&self, x: &Tensor<Bk, 2>, fault: &FaultCell) -> (Tensor<Bk, 2>, Route) {
        let dev = x.device();
        assert!(fault.dev == dev, "moe: a fault cell on {:?} for x on {dev:?}", fault.dev);
        let route = super::prof::span("router", &dev, || self.route(x));
        let out = super::prof::span("experts", &dev, || self.experts_grouped(x, &route, fault));
        (out, route)
    }

    pub fn experts(&self) -> usize {
        self.experts
    }

    #[cfg(test)]
    fn client(&self) -> Client {
        client_of(&Tensor::<Bk, 1>::zeros([1], &burn::backend::cuda::CudaDevice::default()))
    }

    /// The router logits and the device top-k of `x` (`[rows, H]`).
    pub fn route(&self, x: &Tensor<Bk, 2>) -> Route {
        let [rows, h] = x.dims();
        assert_eq!(h, self.hidden);
        let client = client_of(x);
        let dev = x.device();
        let (xh, dt) = handle_of_any(x.clone());
        let lh = match dt {
            burn::tensor::DType::BF16 => linear_bf16_narrow(&client, &xh, &self.router, rows),
            burn::tensor::DType::F32 => linear_bf16_handle(&client, &xh, &self.router, rows),
            other => panic!("router: no lane for a {other:?} activation"),
        };
        let stride = self.router.n;
        let lrows = crate::models::inkling::bf16gemm::rows_for(self.router.align, rows);
        let logits = tensor_of(client.clone(), dev, lh.clone(), lrows, stride).slice([0..rows, 0..self.experts]);
        let topk = router_topk_launch(&client, &lh, &self.bias, rows, stride, self.experts, 0, self.top_k, self.route_scale);
        Route { logits, topk, rows, top_k: self.top_k }
    }

    /// The device plan for `route` (see the module doc). `fault` (one u32)
    /// receives `1 + logit` of a row whose router logits were not finite.
    fn plan(&self, client: &Client, route: &Route, fault: &FaultCell) -> Plan {
        assert_eq!(route.top_k, self.top_k, "plan: a route of top {} on a bank of top {}", route.top_k, self.top_k);
        let (rows, k, e) = (route.rows, self.top_k, self.experts);
        let max_tiles = max_tiles(rows, k, e);
        let m_max = max_tiles * MTILE;
        let sel = client.empty(rows * e * 4);
        let rank = client.empty(rows * e * 4);
        let row_src = client.empty(m_max * 4);
        let row_w = client.empty(m_max * 4);
        let tile_base = client.empty(e * 4);
        let tile_slot = client.empty(max_tiles * 4);
        let (bs, bt, bc) = (client.empty(max_tiles * 4), client.empty(max_tiles * 4), client.empty(max_tiles * 4));
        let tok_rows = client.empty(rows * k * 4);
        let tok_cnt = client.empty(rows * 4);
        let width = 2 * k + 1;
        let span = (rows * e).max(m_max);
        unsafe {
            route_select::launch_unchecked::<cubecl::cuda::CudaRuntime>(
                client,
                CubeCount::new_1d(span.div_ceil(CUBE as usize) as u32),
                CubeDim::new_1d(CUBE),
                ArrayArg::from_raw_parts(route.topk.clone(), rows * width),
                ArrayArg::from_raw_parts(sel.clone(), rows * e),
                ArrayArg::from_raw_parts(row_src.clone(), m_max),
                ArrayArg::from_raw_parts(row_w.clone(), m_max),
                ArrayArg::from_raw_parts(tok_rows.clone(), rows * k),
                rows as u32,
                m_max as u32,
                e as u32,
                k as u32,
            );
            route_rank::launch_unchecked::<cubecl::cuda::CudaRuntime>(
                client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(e as u32),
                ArrayArg::from_raw_parts(sel.clone(), rows * e),
                ArrayArg::from_raw_parts(rank.clone(), rows * e),
                ArrayArg::from_raw_parts(tile_base.clone(), e),
                ArrayArg::from_raw_parts(tile_slot.clone(), max_tiles),
                ArrayArg::from_raw_parts(bs.clone(), max_tiles),
                ArrayArg::from_raw_parts(bt.clone(), max_tiles),
                ArrayArg::from_raw_parts(bc.clone(), max_tiles),
                rows as u32,
                max_tiles as u32,
                e as u32,
            );
            route_place::launch_unchecked::<cubecl::cuda::CudaRuntime>(
                client,
                CubeCount::new_1d((rows * k).div_ceil(CUBE as usize) as u32),
                CubeDim::new_1d(CUBE),
                ArrayArg::from_raw_parts(route.topk.clone(), rows * width),
                ArrayArg::from_raw_parts(rank.clone(), rows * e),
                ArrayArg::from_raw_parts(tile_base.clone(), e),
                ArrayArg::from_raw_parts(row_src.clone(), m_max),
                ArrayArg::from_raw_parts(row_w.clone(), m_max),
                ArrayArg::from_raw_parts(tok_rows.clone(), rows * k),
                ArrayArg::from_raw_parts(tok_cnt.clone(), rows),
                ArrayArg::from_raw_parts(fault.handle.clone(), 1),
                rows as u32,
                e as u32,
                k as u32,
            );
        }
        let blk = BlockPlanDev { slot: bs, tile0: bt, cnt: bc, blocks: max_tiles, planes: 1, rows_real: rows * k };
        Plan { row_src, row_w, tile_slot, tok_rows, tok_cnt, blk, max_tiles }
    }

    /// One MXFP8 projection as the BF16 GEMM multiplies it: `[m, n]` f32 from
    /// the MXFP8-rounded activations `a` (`[m, k]` BF16).
    fn project_mxfp8(&self, client: &Client, p: &Proj, a: &Handle, plan: &Plan) -> Handle {
        let Stack::Bf16 { w, off } = &p.stack else {
            panic!("project_mxfp8: a projection decoded to BF16 only");
        };
        let bytes = self.experts * p.n * p.k * 2;
        bf16_linear_grouped_launch(
            client, a, w, bytes, &plan.blk, off, self.experts, plan.max_tiles * MTILE, p.k, p.n,
        )
    }

    /// The grouped experts over `x` (`[rows, H]`) routed by `route`:
    /// `[rows, H]` f32, each row the weighted sum of its picks' outputs.
    /// `fault` as in [`Bank::forward`]. The precision is the module doc's. The
    /// kernels below launch unchecked, so the shapes are checked here.
    fn experts_grouped(&self, x: &Tensor<Bk, 2>, route: &Route, fault: &FaultCell) -> Tensor<Bk, 2> {
        let [rows, h] = x.dims();
        assert_eq!(h, self.hidden, "experts: x is {h} wide, the bank's hidden size is {}", self.hidden);
        assert_eq!(route.rows, rows, "experts: a route of {} rows for {rows} rows of x", route.rows);
        assert_eq!(route.top_k, self.top_k, "experts: a route of top {} on a bank of top {}", route.top_k, self.top_k);
        let client = client_of(x);
        let dev = x.device();
        let plan = self.plan(&client, route, fault);
        let m = plan.max_tiles * MTILE;
        let (xh, dt) = handle_of_any(x.clone());
        let a = match dt {
            burn::tensor::DType::F32 => gather_grouped_bf16(&client, &xh, &plan.row_src, rows, m, h),
            burn::tensor::DType::BF16 => gather_grouped_bf16_from_bf16(&client, &xh, &plan.row_src, rows, m, h),
            other => panic!("experts: no lane for a {other:?} activation"),
        };
        let s = client.empty(m * self.inter * 2);
        let nvfp4 = matches!(self.gate.stack, Stack::Nvfp4 { .. });
        let (y, row_w) = if nvfp4 {
            // W4A4: both operands NVFP4, the activations per token with Four
            // Over Six; the per-token outer scale multiplies the GEMM's rows,
            // so it enters the SwiGLU for gate and up and the scatter weight
            // for down.
            let (ac, asc, alpha) = quantize_nvfp4_46_bf16(&client, &a, m, h);
            let g = self.project_w4a4(&client, &self.gate, &ac, &asc, &plan);
            let u = self.project_w4a4(&client, &self.up, &ac, &asc, &plan);
            self.swiglu(&client, g, u, Some(&alpha), &plan, &s);
            let (sc, ssc, alpha2) = quantize_nvfp4_46_bf16(&client, &s, m, self.inter);
            let y = self.project_w4a4(&client, &self.down, &sc, &ssc, &plan);
            (y, scale_rows(&client, &plan.row_w, &alpha2, m))
        } else {
            // MXFP8: the activations rounded to MXFP8 and held as BF16, against
            // the decoded MXFP8 weights, in the BF16 GEMM (whose accumulator
            // and order are its own, not shown to be an FP8 MMA's).
            let aq = qdq_mxfp8_bf16(&client, &a, m, h);
            let g = self.project_mxfp8(&client, &self.gate, &aq, &plan);
            let u = self.project_mxfp8(&client, &self.up, &aq, &plan);
            self.swiglu(&client, g, u, None, &plan, &s);
            let sq = qdq_mxfp8_bf16(&client, &s, m, self.inter);
            (self.project_mxfp8(&client, &self.down, &sq, &plan), plan.row_w.clone())
        };
        let out = scatter_weighted(&client, &y, &row_w, &plan.tok_rows, &plan.tok_cnt, m, rows, h, self.top_k);
        tensor_of(client, dev, out, rows, h)
    }

    /// [`softcapped_swiglu`] over every stacked row of `plan` into `s` (`[m,
    /// inter]` BF16); `alpha` (`[m]` f32) scales `g` and `u` per row first.
    fn swiglu(&self, client: &Client, g: Handle, u: Handle, alpha: Option<&Handle>, plan: &Plan, s: &Handle) {
        let m = plan.max_tiles * MTILE;
        let total = m * self.inter;
        unsafe {
            softcapped_swiglu::launch_unchecked::<cubecl::cuda::CudaRuntime>(
                client,
                CubeCount::new_1d(total.div_ceil(CUBE as usize) as u32),
                CubeDim::new_1d(CUBE),
                ArrayArg::from_raw_parts(g, total),
                ArrayArg::from_raw_parts(u, total),
                ArrayArg::from_raw_parts(plan.tile_slot.clone(), plan.max_tiles),
                ArrayArg::from_raw_parts(alpha.unwrap_or(&plan.row_w).clone(), if alpha.is_some() { m } else { 1 }),
                ArrayArg::from_raw_parts(s.clone(), total),
                total as u32,
                self.gate_softcap,
                self.up_softcap,
                self.inter as u32,
                alpha.is_some(),
            );
        }
    }

    /// One NVFP4 projection with NVFP4 activations (`codes`, `scales` as
    /// [`quantize_nvfp4_46_bf16`] writes them): `[m, n]` f32, each row still
    /// to be multiplied by its activation's outer scale.
    fn project_w4a4(&self, client: &Client, p: &Proj, codes: &Handle, scales: &Handle, plan: &Plan) -> Handle {
        let Stack::Nvfp4 { codes: wc, scales: ws, scale2, off, .. } = &p.stack else {
            panic!("project_w4a4: an NVFP4 projection only");
        };
        let e = self.experts;
        fp4_linear_grouped_planes_launch_as::<f32, cubecl::cuda::CudaRuntime>(
            client,
            false,
            codes,
            scales,
            (wc, e * p.n * p.k / 2),
            (ws, e * p.n * p.k / GROUP),
            &plan.blk,
            off,
            scale2,
            e,
            plan.max_tiles * MTILE,
            p.k,
            p.n,
        )
    }
}

/// `out[r] = w[r] * alpha[r]` over `m` f32 rows.
#[cube(launch_unchecked)]
fn scale_rows_kernel(w: &Array<f32>, alpha: &Array<f32>, out: &mut Array<f32>, m: u32) {
    let p = ABSOLUTE_POS as u32;
    if p < m {
        out[p as usize] = w[p as usize] * alpha[p as usize];
    }
}

fn scale_rows(client: &Client, w: &Handle, alpha: &Handle, m: usize) -> Handle {
    let out = client.empty(m * 4);
    unsafe {
        scale_rows_kernel::launch_unchecked::<cubecl::cuda::CudaRuntime>(
            client,
            CubeCount::new_1d(m.div_ceil(CUBE as usize) as u32),
            CubeDim::new_1d(CUBE),
            ArrayArg::from_raw_parts(w.clone(), m),
            ArrayArg::from_raw_parts(alpha.clone(), m),
            ArrayArg::from_raw_parts(out.clone(), m),
            m as u32,
        );
    }
    out
}

/// Bind helpers: one projection's stack from per-expert host bytes.
pub(super) fn nvfp4_proj(client: &Client, codes: &[u8], scales: &[u8], scale2: Vec<f32>, n: usize, k: usize) -> Proj {
    let experts = scale2.len();
    assert_eq!(codes.len(), experts * n * k / 2);
    assert_eq!(scales.len(), experts * n * k / GROUP);
    let off: Vec<u64> = (0..experts).flat_map(|e| [(e * n * k / 2) as u64, (e * n * k / GROUP) as u64]).collect();
    Proj {
        stack: Stack::Nvfp4 {
            codes: client.create_from_slice(codes),
            scales: client.create_from_slice(scales),
            scale2: client.create_from_slice(f32::as_bytes(&scale2)),
            off: client.create_from_slice(u64::as_bytes(&off)),
        },
        n,
        k,
    }
}

pub(super) fn bf16_proj(client: &Client, bytes: &[u8], experts: usize, n: usize, k: usize) -> Proj {
    assert_eq!(bytes.len(), experts * n * k * 2);
    let off: Vec<u64> = (0..experts).map(|e| (e * n * k) as u64).collect();
    Proj {
        stack: Stack::Bf16 { w: client.create_from_slice(bytes), off: client.create_from_slice(u64::as_bytes(&off)) },
        n,
        k,
    }
}

/// A router fault cell for [`Bank::forward`]: one u32 on one device, `1 +
/// logit` once a row's router logits were not finite. Only [`fault_cell`]
/// builds one, so its buffer is always that one u32; the handle stays in this
/// module, and [`Bank::forward`] checks the device. A clone is the same cell.
#[derive(Clone)]
pub struct FaultCell {
    handle: Handle,
    client: Client,
    dev: Dev,
}

impl FaultCell {
    /// The device the cell lives on.
    pub fn device(&self) -> &Dev {
        &self.dev
    }
}

/// A fresh fault cell on `dev`: one u32, zero.
pub fn fault_cell(dev: &Dev) -> FaultCell {
    let client = client_of(&Tensor::<Bk, 1>::zeros([1], dev));
    FaultCell { handle: u32s(&client, &[0]), client, dev: dev.clone() }
}

/// Read a fault cell: `Some(logit + 1)` if a row's router logits were not
/// finite. The read is a sync.
pub fn read_fault(fault: &FaultCell) -> Option<u32> {
    let b = fault.client.read_one(fault.handle.clone()).expect("read the router fault cell");
    let v = u32::from_bytes(&b)[0];
    (v != 0).then_some(v)
}

#[cfg(test)]
mod tests {
    //! The grouped experts against the same bank's experts one at a time, on the
    //! same device routing, and the device routing against
    //! `route_from_logits`. Bug detectors for a change that claims the same
    //! math: a wrong tile, row, expert offset or scale is an order-one error.
    use super::*;
    use crate::models::inkling::block::route_from_logits;
    use crate::models::inkling::seam::tensor_of_dt;
    use crate::models::inkling2_loop::sliding::Dims;
    use burn::tensor::{Int, TensorData};

    fn dev() -> burn::backend::cuda::CudaDevice {
        burn::backend::cuda::CudaDevice::default()
    }

    struct Rng(u64);
    impl Rng {
        fn u(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn f(&mut self, scale: f32) -> f32 {
            ((self.u() >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0 * scale
        }
        fn byte(&mut self) -> u8 {
            (self.u() >> 32) as u8
        }
    }

    fn dims(h: usize, i: usize, e: usize, k: usize) -> Dims {
        Dims {
            hidden: h, heads: 1, kv_heads: 1, head_dim: 1, d_rel: 1, rel_extent: 1, window: 1, kernel: 1,
            experts: e, top_k: k, inter: i, eps: 1e-6, qk_clip: 1.0, rel_clip: 1.0,
            gate_softcap: 4.0, up_softcap: 25.0, route_scale: 8.0, block: 6,
        }
    }

    fn bf16_bytes(r: &mut Rng, n: usize, scale: f32) -> Vec<u8> {
        (0..n).flat_map(|_| half::bf16::from_f32(r.f(scale)).to_le_bytes()).collect()
    }

    /// A bank of random experts: NVFP4 (codes any byte, E4M3 scales 0.25..1)
    /// or BF16, and a random BF16 router.
    fn bank(d: &Dims, nvfp4: bool, seed: u64) -> Bank {
        let client = client_of(&Tensor::<Bk, 1>::zeros([1], &dev()));
        let mut r = Rng(seed);
        let (e, h, i) = (d.experts, d.hidden, d.inter);
        let mut proj = |n: usize, k: usize| -> Proj {
            if nvfp4 {
                let codes: Vec<u8> = (0..e * n * k / 2).map(|_| r.byte()).collect();
                let scales: Vec<u8> = (0..e * n * k / GROUP).map(|_| 0x28 + r.byte() % 16).collect();
                let scale2: Vec<f32> = (0..e).map(|_| 0.004 + r.f(0.002).abs()).collect();
                nvfp4_proj(&client, &codes, &scales, scale2, n, k)
            } else {
                bf16_proj(&client, &bf16_bytes(&mut r, e * n * k, 0.03), e, n, k)
            }
        };
        let projs = [proj(i, h), proj(i, h), proj(h, i)];
        let router = Bf16W { h: client.create_from_slice(&bf16_bytes(&mut r, e * h, 0.05)), n: e, k: h, align: 16 };
        let bias: Vec<f32> = (0..e).map(|_| r.f(0.02)).collect();
        Bank::new(&client, router, bias, projs, d.top_k, d.route_scale, d.gate_softcap, d.up_softcap)
    }

    fn x(rows: usize, h: usize, seed: u64) -> Tensor<Bk, 2> {
        let mut r = Rng(seed);
        Tensor::from_data(TensorData::new((0..rows * h).map(|_| r.f(1.0)).collect::<Vec<_>>(), [rows, h]), &dev())
    }

    fn host(t: Tensor<Bk, 2>) -> Vec<f32> {
        t.into_data().convert::<f32>().to_vec().unwrap()
    }

    fn rel_err(got: &[f32], want: &[f32]) -> f32 {
        let peak = want.iter().fold(0f32, |m, v| m.max(v.abs()));
        got.iter().zip(want).fold(0f32, |m, (g, w)| m.max((g - w).abs())) / peak
    }

    /// The device top-k against `route_from_logits` on the same logits:
    /// identical sets, except a row whose 8th and 9th selection scores are
    /// within 1e-6, which is reported, not failed. Weights compared on rows
    /// whose sets agree.
    #[test]
    fn device_routing_matches_route_from_logits() {
        let d = dims(2304, 1152, 128, 8);
        let b = bank(&d, true, 0x1234);
        let rows = 300;
        let xs = x(rows, d.hidden, 0x4321);
        let route = b.route(&xs);
        let logits = route.logits();
        let host_routes = route_from_logits(&logits, &b.bias_host, 1.0, d.route_scale, rows, d.experts, 0, d.top_k);
        let picks = route.picks();
        let sig = |v: f32| 1.0 / (1.0 + (-v).exp());
        let (mut same, mut near_ties, mut worst_w) = (0, 0, 0f32);
        for (row, (hr, (de, dw))) in host_routes.iter().zip(&picks).enumerate() {
            let a: std::collections::BTreeSet<_> = hr.experts.iter().collect();
            let bset: std::collections::BTreeSet<_> = de.iter().collect();
            if a == bset {
                same += 1;
                for (j, e) in de.iter().enumerate() {
                    let hj = hr.experts.iter().position(|x| x == e).unwrap();
                    worst_w = worst_w.max((dw[j] - hr.weights[hj]).abs() / hr.weights[hj].abs().max(1e-6));
                }
                continue;
            }
            let mut s: Vec<f32> = (0..d.experts).map(|e| sig(logits[row * d.experts + e]) + b.bias_host[e]).collect();
            s.sort_by(|x, y| y.total_cmp(x));
            let gap = s[d.top_k - 1] - s[d.top_k];
            assert!(gap < 1e-6, "row {row}: device picks {de:?}, host picks {:?}, 8th-9th gap {gap}", hr.experts);
            near_ties += 1;
        }
        eprintln!("MOE routing: {same} of {rows} rows identical sets, {near_ties} near ties, worst weight rel diff {worst_w:.3e}");
        assert!(worst_w < 1e-4, "weights differ by {worst_w}");
    }

    fn u32s_of(client: &Client, h: &Handle) -> Vec<u32> {
        u32::from_bytes(&client.read_one(h.clone()).expect("read")).to_vec()
    }

    fn f32s_of(client: &Client, h: &Handle) -> Vec<f32> {
        f32::from_bytes(&client.read_one(h.clone()).expect("read")).to_vec()
    }

    /// A [`Route`] whose top-k buffer is `topk` (host values), over `route`'s logits.
    fn with_topk(client: &Client, route: &Route, topk: &[f32]) -> Route {
        Route {
            logits: route.logits.clone(),
            topk: client.create_from_slice(f32::as_bytes(topk)),
            rows: route.rows,
            top_k: route.top_k,
        }
    }

    /// The fault lands in the cell the call was given and in no other: a
    /// second call on the same bank with its own cell, on finite input, stays
    /// clear and gives the output it gave before the faulted call, bit for bit
    /// (what a second session over the same weights sees). The faulted route
    /// is a real device route with one row's flag column set, so the plan's
    /// picks stay well formed and only the flag path is under test.
    #[test]
    fn a_fault_lands_only_in_the_cell_it_was_given() {
        let d = dims(2304, 1152, 128, 8);
        let b = bank(&d, true, 0xfa17);
        let client = b.client();
        let rows = 37;
        let xs = x(rows, d.hidden, 0x0bad);
        let route = b.route(&xs);
        let before_cell = fault_cell(&dev());
        let before = host(b.experts_grouped(&xs, &route, &before_cell));
        assert_eq!(read_fault(&before_cell), None);

        let mut topk = f32s_of(&client, &route.topk);
        let w = 2 * d.top_k + 1;
        assert!(topk.chunks(w).all(|r| r[w - 1] == 0.0), "the input route is already flagged");
        topk[5 * w + w - 1] = 42.0;
        let faulted = with_topk(&client, &route, &topk);
        let a = fault_cell(&dev());
        let _ = host(b.experts_grouped(&xs, &faulted, &a));
        assert_eq!(read_fault(&a), Some(42), "the faulted call's cell");
        assert_eq!(read_fault(&before_cell), None, "an earlier call's cell");

        let c = fault_cell(&dev());
        let after = host(b.experts_grouped(&xs, &route, &c));
        assert_eq!(read_fault(&c), None, "a later call's own cell");
        assert_eq!(read_fault(&a), Some(42), "a set cell stays set");
        let same = before.iter().zip(&after).filter(|(p, q)| p.to_bits() == q.to_bits()).count();
        assert_eq!(same, before.len(), "the bank's output changed after another cell's fault");
        eprintln!("MOE fault: cell A {:?}, earlier and later cells clear, later output {same} of {} equal", read_fault(&a), before.len());
    }

    /// Non-finite router logits, through the real top-k and the plan: rows
    /// that are finite, partly non-finite (one NaN, one +inf, one -inf,
    /// 121 NaN of 128) and fully NaN. Every index the gather, the GEMMs and
    /// the scatter consume is read back and checked in range; the experts
    /// complete; the cell holds a flag; and a finite call on the same bank
    /// afterwards matches the one before, bit for bit.
    ///
    /// Before each faulted plan, buffers the size of `tok_rows` are filled
    /// with an out-of-range value and dropped, so the pool is likely, not
    /// certain, to hand one back: without the init in `route_select` the
    /// fully-NaN row's unwritten slots then read as that value.
    #[test]
    fn non_finite_router_rows_keep_every_index_in_range() {
        let d = dims(2304, 1152, 128, 8);
        let (e, k) = (d.experts, d.top_k);
        let b = bank(&d, true, 0x7a11);
        let client = b.client();
        let rows = 37;
        let xs = x(rows, d.hidden, 0x1157);
        let good = b.route(&xs);
        let good_cell = fault_cell(&dev());
        let before = host(b.experts_grouped(&xs, &good, &good_cell));
        assert_eq!(read_fault(&good_cell), None);

        let mut logits = good.logits();
        logits[2 * e + 17] = f32::NAN;
        logits[3 * e + 40] = f32::INFINITY;
        logits[4 * e + 90] = f32::NEG_INFINITY;
        for v in &mut logits[6 * e..6 * e + 121] {
            *v = f32::NAN;
        }
        for v in &mut logits[9 * e..10 * e] {
            *v = f32::NAN;
        }
        for v in &mut logits[(rows - 1) * e..rows * e] {
            *v = f32::NAN;
        }
        let lh = client.create_from_slice(f32::as_bytes(&logits));
        let topk = router_topk_launch(&client, &lh, &b.bias, rows, e, e, 0, k, b.route_scale);
        let bad = Route { logits: good.logits.clone(), topk, rows, top_k: k };
        let tv = f32s_of(&client, &bad.topk);
        let w = 2 * k + 1;
        let picks_of = |r: usize| -> Vec<u32> { tv[r * w..r * w + k].iter().map(|v| *v as u32).collect() };
        assert!(picks_of(9).iter().all(|p| *p == 0), "a fully NaN row is meant to repeat expert 0: {:?}", picks_of(9));
        for r in [0, 2, 3, 4] {
            let mut p = picks_of(r);
            p.sort();
            p.dedup();
            assert_eq!(p.len(), k, "row {r}: picks repeat: {:?}", picks_of(r));
        }

        for round in 0..2 {
            for _ in 0..64 {
                drop(client.create_from_slice(u32::as_bytes(&vec![0x7fff_fff0u32; rows * k])));
            }
            let cell = fault_cell(&dev());
            let plan = b.plan(&client, &bad, &cell);
            let m_max = plan.max_tiles * MTILE;
            let tok_rows = u32s_of(&client, &plan.tok_rows);
            let tok_cnt = u32s_of(&client, &plan.tok_cnt);
            let row_src = i32::from_bytes(&client.read_one(plan.row_src.clone()).expect("read")).to_vec();
            let tile_slot = u32s_of(&client, &plan.tile_slot);
            let bad_slots: Vec<(usize, u32)> = tok_rows.iter().copied().enumerate().filter(|(_, v)| *v as usize >= m_max).collect();
            assert!(bad_slots.is_empty(), "round {round}: tok_rows out of range (m_max {m_max}): {:?}", &bad_slots[..bad_slots.len().min(8)]);
            assert!(tok_cnt.iter().all(|c| *c as usize == k), "tok_cnt");
            assert!(row_src.iter().all(|v| *v >= -1 && (*v as i64) < rows as i64), "row_src");
            assert!(tile_slot.iter().all(|v| (*v as usize) < e || *v == NO_EXPERT), "tile_slot");
            let out = host(b.experts_grouped(&xs, &bad, &cell));
            let flag = read_fault(&cell);
            assert!(flag.is_some(), "round {round}: no fault recorded");
            let finite_rows = (0..rows).filter(|r| out[r * d.hidden..(r + 1) * d.hidden].iter().all(|v| v.is_finite())).count();
            eprintln!(
                "MOE non-finite round {round}: m_max {m_max}, tok_rows max {}, row 6 slots {:?}, row 9 slots {:?}, flag {flag:?}, {finite_rows} of {rows} output rows finite",
                tok_rows.iter().max().unwrap(),
                &tok_rows[6 * k..7 * k],
                &tok_rows[9 * k..10 * k],
            );
        }

        let after_cell = fault_cell(&dev());
        let after = host(b.experts_grouped(&xs, &good, &after_cell));
        assert_eq!(read_fault(&after_cell), None);
        let same = before.iter().zip(&after).filter(|(p, q)| p.to_bits() == q.to_bits()).count();
        assert_eq!(same, before.len(), "a finite call after the faulted ones changed");
    }

    // ---- the trained expert precision -----------------------------------

    /// E4M3 round to nearest, ties to the even byte, as `cvt.rn` does.
    fn e4m3_rne(v: f32) -> u8 {
        use crate::models::inkling::nvfp4::e4m3_to_f32;
        let (mut best, mut bestd) = (0u8, f32::INFINITY);
        for b in 0u8..=0x7E {
            let d = (e4m3_to_f32(b) - v.abs()).abs();
            if d < bestd || (d == bestd && b % 2 == 0) {
                (best, bestd) = (b, d);
            }
        }
        if v < 0.0 { best | 0x80 } else { best }
    }

    /// E2M1 code of `q`: nearest magnitude, ties to the even code, saturating at 6.
    fn e2m1_rne(q: f32) -> u8 {
        const MAG: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
        let a = q.abs().min(6.0);
        let (mut best, mut bestd) = (0u8, f32::INFINITY);
        for (c, m) in MAG.iter().enumerate() {
            let d = (m - a).abs();
            if d < bestd || (d == bestd && c % 2 == 0) {
                (best, bestd) = (c as u8, d);
            }
        }
        if q.is_sign_negative() { best | 8 } else { best }
    }

    fn e2m1_mag(c: u8) -> f32 {
        let m = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0][(c & 7) as usize];
        if c & 8 != 0 { -m } else { m }
    }

    /// Host twin of `quantize_nvfp4_46_bf16`: (codes bytes, scale bytes, alpha).
    /// `fma`: the candidate errors as `fma(E2M1, a, -x)`, the contraction
    /// the device compiler may apply; the twin is run both ways.
    fn quant46_host(x: &[f32], rows: usize, k: usize, fma: bool) -> (Vec<u8>, Vec<u8>, Vec<f32>) {
        use crate::models::inkling::nvfp4::e4m3_to_f32;
        let (mut codes, mut scales, mut alphas) = (vec![0u8; rows * k / 2], vec![0u8; rows * k / 16], vec![0f32; rows]);
        for r in 0..rows {
            let row = &x[r * k..(r + 1) * k];
            let alpha = row.iter().fold(0f32, |m, v| m.max(v.abs())) / (6.0 * 256.0);
            alphas[r] = alpha;
            for b in 0..k / 16 {
                let blk = &row[b * 16..(b + 1) * 16];
                let amax = blk.iter().fold(0f32, |m, v| m.max(v.abs()));
                if !(alpha > 0.0 && amax > 0.0) {
                    continue;
                }
                let cand = |target: f32| {
                    let se = e4m3_rne(amax / (alpha * target));
                    let a = e4m3_to_f32(se) * alpha;
                    let inv = if a > 0.0 { 1.0 / a } else { 0.0 };
                    let cs: Vec<u8> = blk.iter().map(|v| e2m1_rne(v * inv)).collect();
                    let mut err = 0f32;
                    let d = |c: u8, v: f32| if fma { e2m1_mag(c).mul_add(a, -v).abs() } else { (e2m1_mag(c) * a - v).abs() };
                    for i in 0..8 {
                        err += d(cs[i], blk[i]);
                        err += d(cs[8 + i], blk[8 + i]);
                    }
                    (se, cs, err)
                };
                let (s6, c6, e6) = cand(6.0);
                let (s4, c4, e4) = cand(4.0);
                let (se, cs) = if e4 < e6 { (s4, c4) } else { (s6, c6) };
                let gb = r * k / 16 + b;
                scales[gb] = se;
                for i in 0..8 {
                    codes[gb * 8 + i] = cs[2 * i] | (cs[2 * i + 1] << 4);
                }
            }
        }
        (codes, scales, alphas)
    }

    /// BF16 activations whose blocks span six orders of magnitude, so both 4/6
    /// candidates win somewhere, plus a zero row.
    fn spread_rows(rows: usize, k: usize, seed: u64) -> Vec<f32> {
        let mut r = Rng(seed);
        let mut v = Vec::with_capacity(rows * k);
        for row in 0..rows {
            for b in 0..k / 16 {
                let scale = 10f32.powf(r.f(3.0));
                for _ in 0..16 {
                    let x = if row == 3 { 0.0 } else { r.f(scale) };
                    v.push(half::bf16::from_f32(x).to_f32());
                }
                let _ = b;
            }
        }
        v
    }

    fn bf16_handle(client: &Client, v: &[f32]) -> Handle {
        let b: Vec<u8> = v.iter().flat_map(|x| half::bf16::from_f32(*x).to_le_bytes()).collect();
        client.create_from_slice(&b)
    }

    #[test]
    fn four_over_six_quantizer_matches_its_host_twin() {
        let client = client_of(&Tensor::<Bk, 1>::zeros([1], &dev()));
        for (rows, k) in [(37, 2304), (5, 1152)] {
            let x = spread_rows(rows, k, 0x46 + k as u64);
            let (c, s, a) = quantize_nvfp4_46_bf16(&client, &bf16_handle(&client, &x), rows, k);
            let (hc, hs, ha) = quant46_host(&x, rows, k, false);
            let (fc, fs, _) = quant46_host(&x, rows, k, true);
            let dc = client.read_one(c).unwrap();
            let ds = client.read_one(s).unwrap();
            let da = f32s_of(&client, &a);
            // a block agrees if it is either twin's block, codes and scale
            let blocks = rows * k / 16;
            let same = |cs: &[u8], ss: &[u8], b: usize| ds[b] == ss[b] && dc[b * 8..b * 8 + 8] == cs[b * 8..b * 8 + 8];
            let bad_b = (0..blocks).filter(|&b| !same(&hc, &hs, b) && !same(&fc, &fs, b)).count();
            let fused = (0..blocks).filter(|&b| !same(&hc, &hs, b) && same(&fc, &fs, b)).count();
            let bad_a = da.iter().zip(&ha).filter(|(p, q)| p.to_bits() != q.to_bits()).count();
            let fours = (0..rows * k / 16)
                .filter(|&b| hs[b] != 0 && {
                    let amax = x[b * 16..b * 16 + 16].iter().fold(0f32, |m, v| m.max(v.abs()));
                    let alpha = ha[b / (k / 16)];
                    hs[b] == e4m3_rne(amax / (alpha * 4.0)) && hs[b] != e4m3_rne(amax / (alpha * 6.0))
                })
                .count();
            eprintln!(
                "QUANT46 {rows}x{k}: {bad_b} of {blocks} blocks match neither twin ({fused} match only the fused-error twin), {bad_a} of {rows} alphas differ; {fours} blocks took the 4 candidate"
            );
            assert!(fours > 0 && fours < blocks, "both candidates should win somewhere");
            assert_eq!((bad_b, bad_a), (0, 0));
            assert!(ds[3 * k / 16..4 * k / 16].iter().all(|b| *b == 0) && da[3] == 0.0, "the zero row");
        }
    }

    #[test]
    fn mxfp8_qdq_matches_its_host_twin() {
        use crate::models::inkling::nvfp4::e4m3_to_f32;
        let client = client_of(&Tensor::<Bk, 1>::zeros([1], &dev()));
        let x = spread_rows(9, 1152, 0x88);
        let got = client.read_one(qdq_mxfp8_bf16(&client, &bf16_handle(&client, &x), 9, 1152)).unwrap();
        let got: Vec<f32> = got.chunks_exact(2).map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32()).collect();
        let mut bad = 0;
        for (bi, blk) in x.chunks_exact(32).enumerate() {
            let amax = blk.iter().fold(0f32, |m, v| m.max(v.abs()));
            let ex = ((amax.to_bits() >> 23) & 255) as i32;
            let scale = f32::from_bits(((ex - 8).clamp(1, 254) as u32) << 23);
            for (i, v) in blk.iter().enumerate() {
                let want = e4m3_to_f32(e4m3_rne((v * (1.0 / scale)).clamp(-448.0, 448.0))) * scale;
                let g = got[bi * 32 + i];
                if g.to_bits() != want.to_bits() && !(g == 0.0 && want == 0.0) {
                    bad += 1;
                }
            }
        }
        eprintln!("MXFP8 qdq: {bad} of {} values differ from the host twin", x.len());
        assert_eq!(bad, 0);
    }

    /// `x` through the trained activation quantiser and back: BF16, then 4/6
    /// NVFP4 per row, dequantized with its outer scale (f32).
    fn qdq46(x: Tensor<Bk, 2>) -> Tensor<Bk, 2> {
        use crate::models::inkling::fp4quant::dequantize_nvfp4;
        let [n, k] = x.dims();
        let client = client_of(&x);
        let dev = x.device();
        let (h, dt) = handle_of_any(x.cast(burn::tensor::FloatDType::BF16));
        assert_eq!(dt, burn::tensor::DType::BF16);
        let (c, s, a) = quantize_nvfp4_46_bf16(&client, &h, n, k);
        let deq = tensor_of(client.clone(), dev.clone(), dequantize_nvfp4(&client, &c, &s, n, k), n, k);
        deq * tensor_of(client, dev, a, n, 1)
    }

    /// `x` through the MXFP8 activation rounding, as f32.
    fn qdq_mx(x: Tensor<Bk, 2>) -> Tensor<Bk, 2> {
        let [n, k] = x.dims();
        let client = client_of(&x);
        let dev = x.device();
        let (h, _) = handle_of_any(x.cast(burn::tensor::FloatDType::BF16));
        let h = qdq_mxfp8_bf16(&client, &h, n, k);
        tensor_of_dt(client, dev, h, n, k, burn::tensor::DType::BF16).cast(burn::tensor::FloatDType::F32)
    }

    /// Expert `e` of `p` against `x`, alone: the dense GEMM on the expert's
    /// slice of the stacked weights.
    fn linear(b: &Bank, p: &Proj, e: usize, x: Tensor<Bk, 2>) -> Tensor<Bk, 2> {
        use crate::models::inkling::burn::{PackedW, linear_bf16, linear_w4a16};
        let slice = |h: &Handle, per: usize| {
            h.clone().offset_start((e * per) as u64).offset_end(((b.experts - 1 - e) * per) as u64)
        };
        match &p.stack {
            Stack::Nvfp4 { codes, scales, scale2, .. } => {
                let w = PackedW {
                    codes: slice(codes, p.n * p.k / 2),
                    scales: slice(scales, p.n * p.k / GROUP),
                    n: p.n,
                    k: p.k,
                    scale2: f32s_of(&b.client(), scale2)[e],
                    swizzled: false,
                };
                linear_w4a16(x, &w)
            }
            Stack::Bf16 { w, .. } => {
                linear_bf16(x, &Bf16W { h: slice(w, p.n * p.k * 2), n: p.n, k: p.k, align: 16 })
            }
        }
    }

    /// The grouped experts against one expert at a time, each fed the same
    /// quantized activations (dequantized): the same operands up to the BF16
    /// rounding of an NVFP4 value, in another GEMM. Detects a wrong offset,
    /// outer scale, tile or row. Returns the worst row error.
    fn compare(d: &Dims, nvfp4: bool, rows: usize, seed: u64, bound: f32) -> f32 {
        let b = bank(d, nvfp4, seed);
        let xs = x(rows, d.hidden, seed ^ 0x2222);
        let route = b.route(&xs);
        let picks = route.picks();
        let cell = fault_cell(&dev());
        let grouped = host(b.experts_grouped(&xs, &route, &cell));
        let qdq = |t: Tensor<Bk, 2>| if nvfp4 { qdq46(t) } else { qdq_mx(t) };
        let mut by_expert: std::collections::BTreeMap<usize, Vec<(usize, f32)>> = Default::default();
        for (row, (es, ws)) in picks.iter().enumerate() {
            for (&ex, &g) in es.iter().zip(ws) {
                by_expert.entry(ex).or_default().push((row, g));
            }
        }
        let mut looped = vec![0f32; rows * d.hidden];
        for (&ex, rws) in &by_expert {
            let idx: Vec<i64> = rws.iter().map(|&(r, _)| r as i64).collect();
            let n = idx.len();
            let xsel = xs.clone().select(0, Tensor::<Bk, 1, Int>::from_data(TensorData::new(idx, [n]), &dev()));
            let xq = qdq(xsel);
            let g = linear(&b, &b.gate, ex, xq.clone());
            let u = linear(&b, &b.up, ex, xq);
            let capped = |v: Tensor<Bk, 2>, c: f32| v.div_scalar(c).tanh().mul_scalar(c);
            let a = capped(g.clone(), d.gate_softcap) * burn::tensor::activation::sigmoid(g) * capped(u, d.up_softcap);
            let y = host(linear(&b, &b.down, ex, qdq(a)));
            for (j, &(r, w)) in rws.iter().enumerate() {
                for c in 0..d.hidden {
                    looped[r * d.hidden + c] += w * y[j * d.hidden + c];
                }
            }
        }
        assert_eq!(read_fault(&cell), None);
        let (mut worst, h) = (0f32, d.hidden);
        for row in 0..rows {
            let rr = row * h..(row + 1) * h;
            assert!(looped[rr.clone()].iter().any(|v| *v != 0.0), "row {row}: the loop produced zeros");
            assert!(grouped[rr.clone()].iter().all(|v| v.is_finite()), "row {row}: non-finite");
            let err = rel_err(&grouped[rr.clone()], &looped[rr]);
            assert!(err < bound, "row {row} of {rows}: grouped vs one expert at a time rel err {err}");
            worst = worst.max(err);
        }
        worst
    }

    #[test]
    fn grouped_nvfp4_experts_match_one_at_a_time() {
        // 128 experts at 37 rows: mostly one- or two-row experts, a tile each;
        // 16 experts: about 18 rows each, two tiles; one row: a decode step
        for (e, rows) in [(128, 37), (16, 37), (128, 1)] {
            let worst = compare(&dims(2304, 1152, e, 8), true, rows, 0x4646 + e as u64, 5e-2);
            eprintln!("MOE trained nvfp4 E {e} rows {rows}: grouped vs one at a time worst row rel err {worst:.3e}");
        }
    }

    #[test]
    fn grouped_mxfp8_experts_match_one_at_a_time() {
        for (e, rows) in [(128, 37), (16, 37), (128, 1)] {
            let worst = compare(&dims(2304, 1152, e, 8), false, rows, 0x7f4a + e as u64, 1e-2);
            eprintln!("MOE trained mxfp8 E {e} rows {rows}: grouped vs one at a time worst row rel err {worst:.3e}");
        }
    }

    /// Tiles no expert owns: two equal rows share their eight experts, so
    /// eight of the sixteen planned tiles are owned and eight are not. The
    /// down projection's input is poisoned before the SwiGLU writes it, and
    /// `g` and `u` are poisoned on the unowned rows, which no GEMM writes.
    /// Both down quantizers read every row, so every row must be defined: zero
    /// on the unowned tiles, hence a zero outer scale, zero codes and scales
    /// and zero MXFP8 values. The owned tiles must not be.
    #[test]
    fn tiles_no_expert_owns_reach_the_down_quantizers_as_zeros() {
        let d = dims(2304, 1152, 128, 8);
        let (i, rows) = (d.inter, 2);
        let b = bank(&d, true, 0x0f0f);
        let client = b.client();
        let one = x(1, d.hidden, 0x77);
        let xs = Tensor::cat(vec![one.clone(), one], 0);
        let route = b.route(&xs);
        let plan = b.plan(&client, &route, &fault_cell(&dev()));
        let m = plan.max_tiles * MTILE;
        let slots = u32s_of(&client, &plan.tile_slot);
        let unowned = |row: usize| slots[row / MTILE] == NO_EXPERT;
        let owned_tiles = slots.iter().filter(|s| **s != NO_EXPERT).count();
        assert_eq!((plan.max_tiles, owned_tiles), (16, 8), "the plan this test is about");

        let mut r = Rng(0x5a5a);
        let poison = |row: usize, v: f32| if unowned(row) { 1e30 } else { v };
        let gu: Vec<f32> = (0..m * i).map(|p| poison(p / i, r.f(2.0))).collect();
        let ones = client.create_from_slice(f32::as_bytes(&vec![1f32; m]));
        for scaled in [true, false] {
            let g = client.create_from_slice(f32::as_bytes(&gu));
            let s = bf16_handle(&client, &vec![3.0e5f32; m * i]);
            b.swiglu(&client, g.clone(), g, scaled.then_some(&ones), &plan, &s);
            let bytes = client.read_one(s.clone()).unwrap();
            let sv: Vec<f32> = bytes.chunks_exact(2).map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32()).collect();
            let (codes, scales, alpha) = quantize_nvfp4_46_bf16(&client, &s, m, i);
            let (codes, scales, alpha) = (client.read_one(codes).unwrap(), client.read_one(scales).unwrap(), f32s_of(&client, &alpha));
            let mx = client.read_one(qdq_mxfp8_bf16(&client, &s, m, i)).unwrap();
            let row = |v: &[u8], at: usize, per: usize| v[at * per..(at + 1) * per].iter().any(|b| *b != 0);
            let mut undefined = 0;
            for at in 0..m {
                let live = sv[at * i..(at + 1) * i].iter().any(|v| *v != 0.0);
                let quantized = alpha[at] != 0.0 || row(&codes, at, i / 2) || row(&scales, at, i / GROUP) || row(&mx, at, i * 2);
                if unowned(at) {
                    undefined += (live || quantized) as usize;
                } else {
                    assert!(live && quantized && alpha[at] > 0.0, "owned row {at} is zero");
                }
            }
            eprintln!("MOE unowned tiles, scaled {scaled}: {undefined} of {} unowned rows undefined", m - owned_tiles * MTILE);
            assert_eq!(undefined, 0, "unowned rows reached the down quantizers undefined");
        }
    }

    #[test]
    fn the_tile_bound_holds() {
        assert_eq!(max_tiles(1, 8, 128), 8);
        assert_eq!(max_tiles(4096, 8, 128), (32768 + 15 * 128) / 16);
        assert_eq!(max_tiles(2, 8, 128), 16);
    }
}
