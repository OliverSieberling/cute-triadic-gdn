"""Blackwell (sm100/sm103) primitives the CuTe DSL does not expose directly, emitted as inline PTX.

tcgen05.mma (M=128, cta_group::1, kind::f16: bf16 operands from shared memory, f32 accumulator in tensor memory),
tcgen05.commit to an mbarrier, tcgen05.ld 32x32b (lane L = accumulator row L, one f32 column per register),
the tcgen05 fences, shared-memory matrix descriptors, and a few vector shared/global memory helpers.

Shared-memory operand tiles are 128B-swizzled.  With `off64(r, x)` (tiles of 64 rows) and `off128(r, x)` (tiles
of 128 rows) the element offset of row r, column x is
    off64(r, x)  = 4096*(x//64) + 64*r + 8*(((x//8)%8) ^ (r%8)) + x%8
    off128(r, x) = 8192*(x//64) + 64*r + 8*(((x//8)%8) ^ (r%8)) + x%8
which is the K-major layout when x runs along K and the MN-major layout when x runs along M or N.
"""
import cutlass
from cutlass._mlir.dialects import llvm, arith
from cutlass._mlir import ir
from cutlass.cutlass_dsl import dsl_user_op

ASM = dict(has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)


def idesc(m, n, a_major, b_major):
    """Instruction descriptor, kind::f16 with bf16 A/B and f32 D.  major: 0 = K-major, 1 = MN-major."""
    return (1 << 4) | (1 << 7) | (1 << 10) | (a_major << 15) | (b_major << 16) | ((n >> 3) << 17) | ((m >> 4) << 24)


def _ir(v, loc=None, ip=None):
    return v.ir_value(loc=loc, ip=ip) if hasattr(v, "ir_value") else v


@dsl_user_op
def mma128(c, da, db, idesc_, acc, *, loc=None, ip=None):
    """D[tmem c] (+)= A[desc da] B[desc db], M=128; acc != 0 accumulates.  Issued by one elected lane."""
    z = cutlass.Int32(0)
    llvm.inline_asm(None, [_ir(c, loc, ip), _ir(da, loc, ip), _ir(db, loc, ip), _ir(idesc_, loc, ip), _ir(acc, loc, ip),
                           _ir(z, loc, ip), _ir(z, loc, ip), _ir(z, loc, ip), _ir(z, loc, ip)],
                    "{\n.reg .pred p;\n.reg .pred q;\nelect.sync _|q, 0xFFFFFFFF;\nsetp.ne.b32 p, $4, 0;\n"
                    "@q tcgen05.mma.cta_group::1.kind::f16 [$0], $1, $2, $3, {$5, $6, $7, $8}, p;\n}",
                    "r,l,l,r,r,r,r,r,r", loc=loc, ip=ip, **ASM)


@dsl_user_op
def commit(mbar, *, loc=None, ip=None):
    """Arrive once on mbarrier `mbar` (shared address) when every earlier tcgen05 op of this thread completes."""
    llvm.inline_asm(None, [_ir(mbar, loc, ip)],
                    "{\n.reg .pred q;\nelect.sync _|q, 0xFFFFFFFF;\n"
                    "@q tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [$0];\n}",
                    "r", loc=loc, ip=ip, **ASM)


@dsl_user_op
def fence_before(*, loc=None, ip=None):
    llvm.inline_asm(None, [], "tcgen05.fence::before_thread_sync;", "", loc=loc, ip=ip, **ASM)


@dsl_user_op
def fence_after(*, loc=None, ip=None):
    llvm.inline_asm(None, [], "tcgen05.fence::after_thread_sync;", "", loc=loc, ip=ip, **ASM)


@dsl_user_op
def fence_async(*, loc=None, ip=None):
    """Generic-proxy shared-memory writes -> async-proxy readers (tcgen05.mma operands, TMA stores)."""
    llvm.inline_asm(None, [], "fence.proxy.async.shared::cta;", "", loc=loc, ip=ip, **ASM)


@dsl_user_op
def wait_ld(*, loc=None, ip=None):
    llvm.inline_asm(None, [], "tcgen05.wait::ld.sync.aligned;", "", loc=loc, ip=ip, **ASM)


def ld32(addr_i32, n):
    """tcgen05.ld 32x32b.x{n}: n consecutive f32 columns of this lane's row (as i32 bit patterns)."""
    i32 = ir.IntegerType.get_signless(32)
    st = llvm.StructType.get_literal([i32] * n)
    outs = ", ".join(f"${i}" for i in range(n))
    r = llvm.inline_asm(st, [addr_i32.ir_value()], f"tcgen05.ld.sync.aligned.32x32b.x{n}.b32 {{{outs}}}, [${n}];",
                        ",".join(["=r"] * n) + ",r", **ASM)
    return [cutlass.Int32(llvm.extractvalue(i32, r, [i])) for i in range(n)]


def f32_of_i32(v):
    return cutlass.Float32(arith.bitcast(ir.F32Type.get(), _ir(v)))


def i32_of_f32(x):
    return cutlass.Int32(arith.bitcast(ir.IntegerType.get_signless(32), _ir(x)))


SW128, SW64, SW32 = 2, 4, 6          # descriptor layout types


def smem_desc(addr_i32, lbo_bytes, sbo_bytes, layout=SW128):
    """sm100 shared-memory matrix descriptor (layout: SW128 / SW64 / SW32).  Add (byte offset >> 4) to advance."""
    lo = cutlass.Int64((addr_i32 >> 4) & 0x3FFF) | (cutlass.Int64((lbo_bytes >> 4) & 0x3FFF) << 16)
    hi = cutlass.Int64((sbo_bytes >> 4) & 0x3FFF) | cutlass.Int64(1 << 14) | (cutlass.Int64(layout) << 29)
    return lo | (hi << 32)


def swz_bytes(boff, bits):
    """Byte-offset swizzle S<bits,4,3> (bits 3/2/1 = 128B/64B/32B): XOR bits [7, 7+bits) into [4, 4+bits)."""
    return boff ^ ((boff >> 3) & (((1 << bits) - 1) << 4))


def off64(r, x):
    return 4096 * (x // 64) + 64 * r + 8 * (((x // 8) % 8) ^ (r % 8)) + (x % 8)


def off128(r, x):
    return 8192 * (x // 64) + 64 * r + 8 * (((x // 8) % 8) ^ (r % 8)) + (x % 8)


def mbar_arrive(addr):
    llvm.inline_asm(None, [addr.ir_value()], "mbarrier.arrive.shared::cta.b64 _, [$0];", "r", **ASM)


def mbar_wait(addr, ph):
    llvm.inline_asm(None, [addr.ir_value(), cutlass.Int32(ph).ir_value()],
                    "{\n.reg .pred p;\nLAB_WAIT:\nmbarrier.try_wait.parity.shared::cta.b64 p, [$0], $1, 10000000;\n"
                    "@!p bra LAB_WAIT;\n}", "r,r", **ASM)


def pack2(hi, lo):
    """cvt.rn.bf16x2.f32: bf16(hi) in bits [31:16], bf16(lo) in [15:0] (lo = lower address)."""
    i32 = ir.IntegerType.get_signless(32)
    r = llvm.inline_asm(i32, [_ir(hi), _ir(lo)], "cvt.rn.bf16x2.f32 $0, $1, $2;", "=r,f,f", **ASM)
    return cutlass.Int32(r)


def unpack_lo(w):
    return f32_of_i32(w << 16)


def unpack_hi(w):
    return f32_of_i32(w & cutlass.Int32(-65536))


@dsl_user_op
def sts128(addr, w0, w1, w2, w3, *, loc=None, ip=None):
    llvm.inline_asm(None, [_ir(x, loc, ip) for x in (addr, w0, w1, w2, w3)],
                    "st.shared.v4.b32 [$0], {$1, $2, $3, $4};", "r,r,r,r,r", loc=loc, ip=ip, **ASM)


@dsl_user_op
def stg128(addr, w0, w1, w2, w3, *, loc=None, ip=None):
    llvm.inline_asm(None, [_ir(x, loc, ip) for x in (addr, w0, w1, w2, w3)],
                    "st.global.v4.b32 [$0], {$1, $2, $3, $4};", "l,r,r,r,r", loc=loc, ip=ip, **ASM)


def lds128(addr_i32):
    i32 = ir.IntegerType.get_signless(32)
    st = llvm.StructType.get_literal([i32] * 4)
    r = llvm.inline_asm(st, [addr_i32.ir_value()], "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];", "=r,=r,=r,=r,r", **ASM)
    return [cutlass.Int32(llvm.extractvalue(i32, r, [i])) for i in range(4)]


def ldg128(addr_i64):
    i32 = ir.IntegerType.get_signless(32)
    st = llvm.StructType.get_literal([i32] * 4)
    r = llvm.inline_asm(st, [addr_i64.ir_value()], "ld.global.nc.v4.b32 {$0, $1, $2, $3}, [$4];",
                        "=r,=r,=r,=r,l", **ASM)
    return [cutlass.Int32(llvm.extractvalue(i32, r, [i])) for i in range(4)]


def saddr(it, off):
    """Shared-window byte address of element `off` of shared-memory pointer `it`."""
    return cutlass.Int32((it + off).toint())
