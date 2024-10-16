module GPUTesting

using KernelAbstractions

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

end # module GPUTesting
