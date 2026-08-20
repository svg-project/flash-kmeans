"""CuTe DSL device helpers (atomic reductions, pointer math).

atomic_add_fp32x4 is the vectorized red.global.add.v4.f32 from FA4
(flash_attn/cute/copy_utils.py); atomic_add_i32/fp32 follow the
nvvm.atomicrmw pattern used by quack/FA4.
"""
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm, nvvm, vector


@dsl_user_op
def elem_pointer(x: cute.Tensor, coord: cute.Coord, *, loc=None, ip=None) -> cute.Pointer:
    return x.iterator + cute.crd2idx(coord, x.layout, loc=loc, ip=ip)


@dsl_user_op
def atomic_add_i32(a: int | Int32, gmem_ptr: cute.Pointer, *, loc=None, ip=None) -> None:
    nvvm.atomicrmw(op=nvvm.AtomicOpKind.ADD, ptr=gmem_ptr.llvm_ptr, a=Int32(a).ir_value())


@dsl_user_op
def atomic_add_fp32(a: float | Float32, gmem_ptr: cute.Pointer, *, loc=None, ip=None) -> None:
    nvvm.atomicrmw(op=nvvm.AtomicOpKind.FADD, ptr=gmem_ptr.llvm_ptr, a=Float32(a).ir_value())


@dsl_user_op
def atomic_add_fp32x4(
    a: Float32, b: Float32, c: Float32, d: Float32, gmem_ptr: cute.Pointer, *, loc=None, ip=None
) -> None:
    gmem_ptr_i64 = gmem_ptr.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [
            gmem_ptr_i64,
            Float32(a).ir_value(loc=loc, ip=ip),
            Float32(b).ir_value(loc=loc, ip=ip),
            Float32(c).ir_value(loc=loc, ip=ip),
            Float32(d).ir_value(loc=loc, ip=ip),
        ],
        "{\n\t"
        ".reg .v4 .f32 abcd;\n\t"
        "mov.f32 abcd.x, $1;\n\t"
        "mov.f32 abcd.y, $2;\n\t"
        "mov.f32 abcd.z, $3;\n\t"
        "mov.f32 abcd.w, $4;\n\t"
        "red.global.add.v4.f32 [$0], abcd;\n\t"
        "}\n",
        "l,f,f,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def f32_bits(a: Float32, *, loc=None, ip=None) -> Int32:
    """Bit-pattern of an fp32 value as int32 (no conversion)."""
    v = vector.from_elements(
        T.vector(1, T.f32()), (Float32(a).ir_value(loc=loc, ip=ip),), loc=loc, ip=ip
    )
    vi = vector.bitcast(T.vector(1, T.i32()), v)
    return Int32(vector.extract(vi, dynamic_position=[], static_position=[0], loc=loc, ip=ip))


@dsl_user_op
def i32_as_f32(a: Int32, *, loc=None, ip=None) -> Float32:
    """Reinterpret int32 bits as fp32 (no conversion)."""
    v = vector.from_elements(
        T.vector(1, T.i32()), (Int32(a).ir_value(loc=loc, ip=ip),), loc=loc, ip=ip
    )
    vf = vector.bitcast(T.vector(1, T.f32()), v)
    return Float32(vector.extract(vf, dynamic_position=[], static_position=[0], loc=loc, ip=ip))
