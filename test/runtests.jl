using Test
using GPUTesting

#=
using AMDGPU
using CUDA

array_types = [Array]

if has_rocm_gpu()
    push!(array_types, ROCArray)
end

if has_cuda_gpu()
    push!(array_types, CuArray)
end
=#

for ArrayType in array_types
    @testset "Generic tests on $(string(ArrayType))" begin
        a = ArrayType(rand(100))
        b = ArrayType(rand(100))
        c = ArrayType(similar(a))

        # vector addition tests
        v_add!(a, b, c)
        @test a .+ b == c

        # vector multiplication tests
        v_mult!(a, b, c)
        @test a .* b == c

        X = ArrayType(rand(10,20))
        Y = ArrayType(rand(20,30))
        Z = ArrayType(rand(10,30))

        # matrix multiplication tests
        matrix_mult!(X,Y,Z)
        @test isapprox(Z, X*Y)

        perf_mat_mul!(X, Y, Z)
        @test isapprox(Z, X*Y)
    end
end
