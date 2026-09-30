"""Inline-PTX helpers shared by the kernels."""
import cutlass
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm

_PURE = dict(has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)
_ASMG = dict(has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)


def _f32():
    return ir.F32Type.get()


def _shfl_idx(v, src):
    """broadcast lane `src`'s value of v to every lane of the warp (f32)."""
    return cutlass.Float32(llvm.inline_asm(
        _f32(), [v.ir_value(), cutlass.Int32(src).ir_value()],
        "shfl.sync.idx.b32 $0, $1, $2, 31, 0xffffffff;", "=f,f,r", **_ASMG))


def _selp_f32(a, b, pred):
    """pred != 0 ? a : b  (f32)"""
    return cutlass.Float32(llvm.inline_asm(
        _f32(), [a.ir_value(), b.ir_value(), pred.ir_value()],
        "{\n.reg .pred p;\nsetp.ne.s32 p, $3, 0;\nselp.f32 $0, $1, $2, p;\n}",
        "=f,f,f,r", **_PURE))
