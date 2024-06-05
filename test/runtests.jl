using Test
using GPUTesting

@testset "Generic tests..." begin
    a = rand(100)
    b = rand(100)
    c = similar(a)

    # vector addition tests
    v_add!(a, b, c)
    @test a .+ b == c

    #vector multiplication tests
    v_mult!(a, b, c)
    @test a .* b == c
end
