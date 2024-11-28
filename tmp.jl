using KernelAbstractions
using Test
using GPUTesting
using BenchmarkTools

function find_timings(n, ArrayType, type, f)
    timings = zeros(n)

    # to deal with precompile times
    dummy_array = ArrayType(rand(type, 100, 100))

    f(dummy_array, dummy_array, dummy_array)
    for i = 1:n
        array_size = 2^i
        a = ArrayType(rand(type, array_size, array_size))
        b = ArrayType(rand(type, array_size, array_size))
        c = ArrayType(zeros(type, array_size, array_size))
        timings[i] = AMDGPU.@elapsed begin
            f(a, b)
            KernelAbstractions.synchronize(get_backend(a))
        end
        #@test isapprox(c, a*b)
    end
    return timings
end
