module GPUTesting

using KernelAbstractions

include("v_add.jl")
include("v_mult.jl")
include("matrix_mult.jl")
include("parallel_vadd.jl")
include("performant_matrix_mult.jl")
include("parallel_sum_reduction.jl")

end # module GPUTesting
