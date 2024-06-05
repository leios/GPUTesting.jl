using Test
using GPUTesting

@testset "Generic tests..." begin
    a = rand(100)
    b = rand(100)
    c = similar(a)

    # vector addition tests
    v_add!(a, b, c)
    @test a .+ b == c

    # vector multiplication tests
    v_mult!(a, b, c)
    @test a .* b == c

    X = rand(10,20)
    Y = rand(20,30)
    Z = rand(10,30)

    # matrix multiplication tests
    matrix_mult!(X,Y,Z)
    @test isapprox(Z, X*Y)
end
