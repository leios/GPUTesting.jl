module GPUTesting

using KernelAbstractions
using Dagger
using LinearAlgebra
#using Metal

include("v_add.jl")
include("v_mult.jl")
include("matrix_mult.jl")
include("parallel_vadd.jl")
include("performant_matrix_mult.jl")
include("parallel_sum_reduction.jl")
include("parallel_sum_reduction2.jl")
include("naive_TRMM.jl")
include("warpless_TRMM.jl")
include("native_TRMM.jl")
include("performant_TRMM.jl")
include("performant_TRMM_dagger.jl")
include("shark_TRMM.jl")
include("basline_TRMM.jl")
include("trmm_julia.jl")
include("trmm_julia_gemm.jl")
include("trmm_original.jl")
include("perf_trmm2.jl")
include("gemm_add.jl")
include("trmm_base.jl")


end # module GPUTesting
