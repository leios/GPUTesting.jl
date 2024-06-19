module GPUTesting

using KernelAbstractions

include("v_add.jl")
include("v_mult.jl")
include("matrix_mult.jl")
include("parallel_vadd.jl")


end # module GPUTesting
