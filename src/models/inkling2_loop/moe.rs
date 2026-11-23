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
//! needs no offset table. The per-expert [`super::sliding::Expert`] values the
//! loop multiplies are views into the same buffers: one copy on the device.

use burn::tensor::Tensor;
use cubecl::e4m3;
use cubecl::ir::MatrixIdent;
use cubecl::prelude::*;
use cubecl::server::Handle;
use half::bf16;

use crate::models::inkling::bf16gemm::{Bf16W, linear_bf16 as linear_bf16_handle, linear_bf16_narrow};
use crate::models::inkling::burn::PackedW;
use crate::models::inkling::moegroup::{
    BlockPlanDev, bf16_linear_grouped_launch, gather_grouped_bf16, gather_grouped_bf16_from_bf16,
    scatter_weighted,
};
use crate::models::inkling::routetopk::router_topk_launch;
use crate::models::inkling::seam::{Bk, client_of, handle_of_any, tensor_of};
use crate::models::inkling::w4a16gemm::{CODES_PER_WORD, GROUP, KTILE, MTILE, NTILE, e2m1_value};
use crate::models::inkling2_loop::sliding::{Expert, ExpertW};

type Client = ComputeClient<cubecl::cuda::CudaRuntime>;
type Dev = burn::backend::cuda::CudaDevice;

/// The tile-table value of a tile no expert owns (`0xFFFF_FFFF` in the kernels).
pub const NO_EXPERT: u32 = u32::MAX;
const CUBE: u32 = 256;

/// One projection of every expert of a layer, stacked `[E, n, k]`.
pub(super) enum Stack {
    /// NVFP4: codes `[E * n, k / 2]` bytes, E4M3 scales `[E * n, k / 16]`,
    /// and each expert's second-level scale `[E]` f32.
    Nvfp4 { codes: Handle, scales: Handle, scale2: Handle, scale2_host: Vec<f32> },
    /// BF16 (MXFP8 decoded exactly): `[E * n, k]`, and the per-expert element
    /// offsets `[E]` u64 that [`bf16_linear_grouped_launch`] indexes.
    Bf16 { w: Handle, off: Handle },
}

pub(super) struct Proj {
    pub stack: Stack,
    pub n: usize,
    pub k: usize,
}

impl Proj {
    /// Expert `e`'s `[n, k]` slice, as the per-expert loop multiplies it.
    fn view(&self, e: usize, experts: usize) -> ExpertW {
        let slice = |h: &Handle, per: usize| {
            h.clone().offset_start((e * per) as u64).offset_end(((experts - 1 - e) * per) as u64)
        };
        match &self.stack {
            Stack::Nvfp4 { codes, scales, scale2_host, .. } => ExpertW::Nvfp4(PackedW {
                codes: slice(codes, self.n * self.k / 2),
                scales: slice(scales, self.n * self.k / GROUP),
                n: self.n,
                k: self.k,
                scale2: scale2_host[e],
                swizzled: false,
            }),
            Stack::Bf16 { w, .. } => {
                ExpertW::Bf16(Bf16W { h: slice(w, self.n * self.k * 2), n: self.n, k: self.k, align: 16 })
            }
        }
    }
}

/// A layer's router and routed experts.
pub struct Bank {
    router: Bf16W,
    /// `[E]` f32 on the device.
    bias: Handle,
    pub(super) bias_host: Vec<f32>,
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
    /// The same experts as views, for the per-expert loop.
    pub(super) loop_experts: Vec<Expert>,
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

/// Which expert lane a visit runs. The grouped lane is the default; the loop
/// is kept, on the same weights, as what the grouped lane is checked against
/// and for NVFP4 activations ([`super::sliding::ExpertAct::Nvfp4`]), which the
/// grouped lane does not implement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lane {
    Grouped,
    Loop,
}

/// The lane for `act`: NVFP4 activations take the loop; otherwise the grouped
/// lane, unless `MINI_MOE=loop` asks for the loop (an A/B in one binary).
pub fn lane(act: super::sliding::ExpertAct) -> Lane {
    use std::sync::OnceLock;
    static FORCED: OnceLock<bool> = OnceLock::new();
    let forced = *FORCED.get_or_init(|| match std::env::var("MINI_MOE").as_deref() {
        Ok("loop") => true,
        Ok("grouped") | Err(_) => false,
        Ok(other) => panic!("MINI_MOE={other}: the lanes are grouped and loop"),
    });
    if forced || act == super::sliding::ExpertAct::Nvfp4 { Lane::Loop } else { Lane::Grouped }
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
/// for the down projection. Rows of tiles no expert owns are skipped.
#[cube(launch_unchecked)]
fn softcapped_swiglu(
    g: &Array<f32>,
    u: &Array<f32>,
    tile_slot: &Array<u32>,
    out: &mut Array<bf16>,
    total: u32,
    gate_cap: f32,
    up_cap: f32,
    #[comptime] inter: u32,
) {
    let p = ABSOLUTE_POS as u32;
    if p < total {
        let tile = p / comptime!(inter * 16);
        if tile_slot[tile as usize] != 0xFFFF_FFFFu32 {
            let gv = g[p as usize];
            let uv = u[p as usize];
            let cg = Tanh::tanh(gv / gate_cap) * gate_cap;
            let cu = Tanh::tanh(uv / up_cap) * up_cap;
            // sigmoid without overflow: e = exp(-|g|), 1 / (1 + e) or e / (1 + e)
            let e = Exp::exp(-Abs::abs(gv));
            let mut s = e / (1.0f32 + e);
            if gv >= 0.0f32 {
                s = 1.0f32 / (1.0f32 + e);
            }
            out[p as usize] = bf16::cast_from(cg * s * cu);
        }
    }
}

/// [`crate::models::inkling::w4a16gemm::w4a16_linear`] over stacked experts:
/// M tile `CUBE_POS_Y` belongs to expert `tile_slot[m_tile]` (or to none, and
/// ends), whose weight rows start at `slot * size_n` of the `[E * n, k]`
/// planes and whose second-level scale is `scale2[slot]`. Everything inside
/// the k loop is the parent's.
///
/// The grid is the parent's transposed, N in x: the dense kernel puts M in x
/// so the few m tiles sharing a weight row run together, but here M is every
/// expert's rows stacked (about 2,200 tiles, 160 MB of A at a 4,096-token
/// chunk), and M-fastest would stream all of A from DRAM once per N tile.
/// N-fastest keeps one m tile's A and its expert's B (about 1.5 MB for a
/// gate or up projection) in L2 while its N tiles run, which is the reuse the
/// per-expert launches had by construction.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn w4a16_grouped<AB: Scalar + Cast, S: Scalar, NA: Size, NC: Size>(
    a: &Array<Vector<AB, NA>>,
    b: &Array<u32>,
    b_sc: &Array<S>,
    scale2: &Array<f32>,
    tile_slot: &Array<u32>,
    out: &mut Array<Vector<f32, NC>>,
    #[comptime] size_k: usize,
    #[comptime] size_n: usize,
) {
    let def = cmma::MmaDefinition::<AB, AB, f32>::new(MTILE, NTILE, KTILE);
    let lane = UNIT_POS_PLANE;
    let pack = AB::packing_factor();

    let n_tile = CUBE_POS_X as usize;
    let m_tile = CUBE_POS_Y as usize;
    let slot = tile_slot[m_tile];
    if slot == 0xFFFF_FFFFu32 {
        terminate!();
    }
    let n_base = n_tile * NTILE;
    let m_base = m_tile * MTILE;
    let row0 = slot as usize * size_n;
    let scale = scale2[slot as usize];

    let ec_a = def.elems_per_lane(MatrixIdent::A);
    let vs_a = def.vector_size(MatrixIdent::A);
    let vc_a = comptime!(ec_a / vs_a);
    let ec_b = def.elems_per_lane(MatrixIdent::B);
    let vs_b = def.vector_size(MatrixIdent::B);
    let vc_b = comptime!(ec_b / vs_b);
    let ec_c = def.elems_per_lane(MatrixIdent::Accumulator);
    let vs_c = def.vector_size(MatrixIdent::Accumulator);
    let vc_c = comptime!(ec_c / vs_c);

    let mut reg_a = Array::<Vector<AB, NA>>::new(vc_a);
    let mut reg_b = Array::<Vector<AB, NA>>::new(vc_b);
    let mut acc = Array::<Vector<f32, NC>>::new(vc_c);
    #[unroll]
    for i in 0..vc_c {
        acc[i] = Vector::<f32, NC>::cast_from(0.0f32);
    }

    let wpr = comptime!(size_k / CODES_PER_WORD);
    let spr = comptime!(size_k / GROUP);
    let k_tiles = comptime!(size_k / KTILE);

    for t in 0..k_tiles {
        let kbase = t * KTILE;
        #[unroll]
        for i in 0..vc_a {
            let (row, col) = def.position_of_nth(lane, (i * vs_a * pack) as u32, MatrixIdent::A);
            let gr = row as usize + m_base;
            let gc = col as usize + kbase;
            reg_a[i] = a[(gr * size_k + gc) / a.vector_size()];
        }
        #[unroll]
        for i in 0..vc_b {
            let (row, col) = def.position_of_nth(lane, (i * vs_b) as u32, MatrixIdent::B);
            let gr = row0 + col as usize + n_base;
            let gc = row as usize + kbase;
            let word = b[gr * wpr + gc / CODES_PER_WORD];
            let s = f32::cast_from(b_sc[gr * spr + gc / GROUP]);
            let mut v = Vector::<AB, NA>::empty();
            #[unroll]
            for j in 0..vs_b {
                let kk = gc + j;
                let code = (word >> (4 * (kk % CODES_PER_WORD)) as u32) & 15u32;
                v[j] = AB::cast_from(e2m1_value(code) * s);
            }
            reg_b[i] = v;
        }
        let d = def.execute(&reg_a, &reg_b, &acc);
        #[unroll]
        for i in 0..vc_c {
            acc[i] = d[i];
        }
    }

    #[unroll]
    for i in 0..vc_c {
        let (row, col) = def.position_of_nth(lane, (i * vs_c) as u32, MatrixIdent::Accumulator);
        let gr = row as usize + m_base;
        let gc = col as usize + n_base;
        out[(gr * size_n + gc) / out.vector_size()] = acc[i] * Vector::<f32, NC>::cast_from(scale);
    }
}

#[allow(clippy::too_many_arguments)]
fn w4a16_grouped_launch(
    client: &Client,
    a: &Handle,
    codes: &Handle,
    scales: &Handle,
    scale2: &Handle,
    tile_slot: &Handle,
    experts: usize,
    max_tiles: usize,
    k: usize,
    n: usize,
) -> Handle {
    assert!(n % NTILE == 0 && k % KTILE == 0, "w4a16_grouped: {n}x{k} does not tile");
    assert!(max_tiles <= 65535, "w4a16_grouped: {max_tiles} m-tiles exceed grid y");
    let m = max_tiles * MTILE;
    let out = client.empty(m * n * core::mem::size_of::<f32>());
    let vs = 32 / bf16::cube_type().size_bits();
    unsafe {
        w4a16_grouped::launch::<bf16, e4m3, cubecl::cuda::CudaRuntime>(
            client,
            CubeCount::Static((n / NTILE) as u32, max_tiles as u32, 1),
            CubeDim::new_1d(32),
            vs,
            2,
            ArrayArg::from_raw_parts(a.clone(), m * k),
            ArrayArg::from_raw_parts(codes.clone(), experts * n * (k / CODES_PER_WORD)),
            ArrayArg::from_raw_parts(scales.clone(), experts * n * (k / GROUP)),
            ArrayArg::from_raw_parts(scale2.clone(), experts),
            ArrayArg::from_raw_parts(tile_slot.clone(), max_tiles),
            ArrayArg::from_raw_parts(out.clone(), m * n),
            k,
            n,
        )
    };
    out
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
        let loop_experts = (0..experts)
            .map(|e| Expert { gate: gate.view(e, experts), up: up.view(e, experts), down: down.view(e, experts) })
            .collect();
        Bank {
            router,
            bias: client.create_from_slice(f32::as_bytes(&bias)),
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
            loop_experts,
        }
    }

    /// The router, then the experts in the lane `act` and `MINI_MOE` select:
    /// `[rows, H]` f32, and the routing on the device.
    ///
    /// `fault` is the caller's cell ([`fault_cell`]): the grouped lane writes
    /// `1 + logit` into it when a row's router logits are not finite, its form
    /// of `route_from_logits`' panic, and nothing else reads or clears it. The
    /// read is a sync, so it is the caller's to place ([`read_fault`]); the
    /// loop lane panics in [`Route::picks`] instead. The cell belongs to
    /// whoever owns the call's state, never to the bank, so one context's
    /// fault cannot reach another context sharing these weights.
    pub fn forward(
        &self,
        d: &super::sliding::Dims,
        x: &Tensor<Bk, 2>,
        act: super::sliding::ExpertAct,
        fault: &FaultCell,
    ) -> (Tensor<Bk, 2>, Route) {
        let dev = x.device();
        assert!(fault.dev == dev, "moe: a fault cell on {:?} for x on {dev:?}", fault.dev);
        let route = super::prof::span("router", &dev, || self.route(x));
        let out = super::prof::span("experts", &dev, || match lane(act) {
            Lane::Grouped => self.experts_grouped(x, &route, fault),
            Lane::Loop => experts_loop(self, d, x, &route.picks(), act),
        });
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

    fn project(&self, client: &Client, p: &Proj, a: &Handle, plan: &Plan) -> Handle {
        let m = plan.max_tiles * MTILE;
        match &p.stack {
            Stack::Nvfp4 { codes, scales, scale2, .. } => w4a16_grouped_launch(
                client, a, codes, scales, scale2, &plan.tile_slot, self.experts, plan.max_tiles, p.k, p.n,
            ),
            Stack::Bf16 { w, off } => bf16_linear_grouped_launch(
                client,
                a,
                w,
                self.experts * p.n * p.k * 2,
                &plan.blk,
                off,
                self.experts,
                m,
                p.k,
                p.n,
            ),
        }
    }

    /// The grouped experts over `x` (`[rows, H]`) routed by `route`:
    /// `[rows, H]` f32, each row the weighted sum of its picks' outputs.
    /// `fault` as in [`Bank::forward`]. The kernels below launch unchecked,
    /// so the shapes are checked here.
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
        let g = self.project(&client, &self.gate, &a, &plan);
        let u = self.project(&client, &self.up, &a, &plan);
        let total = m * self.inter;
        let s = client.empty(total * 2);
        unsafe {
            softcapped_swiglu::launch_unchecked::<cubecl::cuda::CudaRuntime>(
                &client,
                CubeCount::new_1d(total.div_ceil(CUBE as usize) as u32),
                CubeDim::new_1d(CUBE),
                ArrayArg::from_raw_parts(g, total),
                ArrayArg::from_raw_parts(u, total),
                ArrayArg::from_raw_parts(plan.tile_slot.clone(), plan.max_tiles),
                ArrayArg::from_raw_parts(s.clone(), total),
                total as u32,
                self.gate_softcap,
                self.up_softcap,
                self.inter as u32,
            );
        }
        let y = self.project(&client, &self.down, &s, &plan);
        let out = scatter_weighted(&client, &y, &plan.row_w, &plan.tok_rows, &plan.tok_cnt, m, rows, h, self.top_k);
        tensor_of(client, dev, out, rows, h)
    }
}

/// Bind helpers: one projection's stack from per-expert host bytes.
pub(super) fn nvfp4_proj(client: &Client, codes: &[u8], scales: &[u8], scale2: Vec<f32>, n: usize, k: usize) -> Proj {
    let experts = scale2.len();
    assert_eq!(codes.len(), experts * n * k / 2);
    assert_eq!(scales.len(), experts * n * k / GROUP);
    Proj {
        stack: Stack::Nvfp4 {
            codes: client.create_from_slice(codes),
            scales: client.create_from_slice(scales),
            scale2: client.create_from_slice(f32::as_bytes(&scale2)),
            scale2_host: scale2,
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

/// The per-expert loop, on the bank's views (`route` read back to the host).
pub(super) fn experts_loop(
    bank: &Bank,
    d: &super::sliding::Dims,
    x: &Tensor<Bk, 2>,
    picks: &[(Vec<usize>, Vec<f32>)],
    act: super::sliding::ExpertAct,
) -> Tensor<Bk, 2> {
    use std::collections::BTreeMap;
    use burn::tensor::{Int, TensorData};
    let [t, _] = x.dims();
    let dev = x.device();
    let mut by_expert: BTreeMap<usize, Vec<(usize, f32)>> = BTreeMap::new();
    for (row, (es, ws)) in picks.iter().enumerate() {
        for (&e, &g) in es.iter().zip(ws) {
            by_expert.entry(e).or_default().push((row, g));
        }
    }
    let mut acc: Tensor<Bk, 2> = Tensor::zeros([t, d.hidden], &dev);
    for (&e, rows) in &by_expert {
        let idx: Vec<i64> = rows.iter().map(|&(r, _)| r as i64).collect();
        let gw: Vec<f32> = rows.iter().map(|&(_, g)| g).collect();
        let n = idx.len();
        let idx = Tensor::<Bk, 1, Int>::from_data(TensorData::new(idx, [n]), &dev);
        let gw = Tensor::<Bk, 2>::from_data(TensorData::new(gw, [n, 1]), &dev);
        let xs = x.clone().select(0, idx.clone());
        let y = bank.loop_experts[e].forward(d, xs, act) * gw;
        acc = acc.select_assign(0, idx, y, burn::tensor::IndexingUpdateOp::Add);
    }
    acc
}

#[cfg(test)]
mod tests {
    //! The grouped lane against the per-expert loop on the same bank and the
    //! same device routing, and the device routing against
    //! `route_from_logits`. Bug detectors for a change that claims the same
    //! math: a wrong tile, row, expert offset or scale is an order-one error.
    use super::*;
    use crate::models::inkling::block::route_from_logits;
    use crate::models::inkling2_loop::sliding::{Dims, ExpertAct};
    use burn::tensor::TensorData;

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

    /// Grouped against the loop, every row (row 0 included), at the release's
    /// H and I. Returns the worst row error and how many rows differ at all.
    fn compare(d: &Dims, nvfp4: bool, rows: usize, seed: u64) -> (f32, usize) {
        let b = bank(d, nvfp4, seed);
        let xs = x(rows, d.hidden, seed ^ 0x5555);
        let route = b.route(&xs);
        let picks = route.picks();
        let cell = fault_cell(&dev());
        let grouped = host(b.experts_grouped(&xs, &route, &cell));
        let looped = host(experts_loop(&b, d, &xs, &picks, ExpertAct::Bf16));
        assert_eq!(read_fault(&cell), None);
        let h = d.hidden;
        let (mut worst, mut differ) = (0f32, 0);
        for row in 0..rows {
            let rr = row * h..(row + 1) * h;
            let e = rel_err(&grouped[rr.clone()], &looped[rr.clone()]);
            assert!(looped[rr.clone()].iter().any(|v| *v != 0.0), "row {row}: the loop produced zeros");
            assert!(grouped[rr].iter().all(|v| v.is_finite()), "row {row}: non-finite grouped output");
            assert!(e < 1e-2, "{} row {row} of {rows}: grouped vs loop rel err {e}", if nvfp4 { "nvfp4" } else { "bf16" });
            worst = worst.max(e);
            differ += (e > 0.0) as usize;
        }
        (worst, differ)
    }

    #[test]
    fn grouped_matches_the_loop_on_nvfp4_experts() {
        // 128 experts at 37 rows: mostly one- or two-row experts, a tile each;
        // 16 experts: about 18 rows each, two tiles; one row: a decode step
        for (e, rows) in [(128, 37), (16, 37), (128, 1)] {
            let (worst, differ) = compare(&dims(2304, 1152, e, 8), true, rows, 0x9e37 + e as u64);
            eprintln!("MOE nvfp4 E {e} rows {rows}: grouped vs loop worst row rel err {worst:.3e}, {differ} of {rows} rows differ");
        }
    }

    #[test]
    fn grouped_matches_the_loop_on_bf16_experts() {
        for (e, rows) in [(128, 37), (16, 37), (128, 1)] {
            let (worst, differ) = compare(&dims(2304, 1152, e, 8), false, rows, 0x7f4a + e as u64);
            eprintln!("MOE bf16 E {e} rows {rows}: grouped vs loop worst row rel err {worst:.3e}, {differ} of {rows} rows differ");
        }
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

    #[test]
    fn the_tile_bound_holds() {
        assert_eq!(max_tiles(1, 8, 128), 8);
        assert_eq!(max_tiles(4096, 8, 128), (32768 + 15 * 128) / 16);
        assert_eq!(max_tiles(2, 8, 128), 16);
    }
}
