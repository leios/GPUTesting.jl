module GPUTesting

using KernelAbstractions

include("v_add.jl")
include("v_mult.jl")
include("matrix_mult.jl")

end # module GPUTesting
