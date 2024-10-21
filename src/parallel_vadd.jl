export par_vadd!

import KernelAbstractions.Extras: @unroll

@kernel function par_vadd_kernel!(a, b, c)

    TILE_DIM = @uniform groupsize()[1]
    BLOCK_ROWS = @uniform groupsize()[2]

    tile = @localmem eltype(a) (TILE_DIM+1, TILE_DIM)

    i = @index(Local, Linear)
    gi = @index(Group, Linear)

    I = (gi-1) * TILE_DIM + i

    @unroll for k in 0:BLOCK_ROWS:(TILE_DIM-1)
        @inbounds tile[i] = A[I] + B[I]
    end

    @synchronize

    @unroll for k in 0:BLOCK_ROWS:(TILE_DIM-1)
        @inbounds c[I] = tile[i]
    end
    
end

function par_vadd!(a, b, c; TILE_DIM = 32, BLOCK_ROWS = 8 )
    if typeof(a) != typeof(b) != typeof(c)
        error("Types of a, b, and c are different!")
    end

    if length(a) != length(b) != length(c)
        error("Lengths of a, b, and c are different!")
    end

    backend = get_backend(a)

    block_factor = div(TILE_DIM,BLOCK_ROWS)
    ndrange = div(length(a), block_factor)
    kernel = par_vadd_kernel!(backend, (TILE_DIM,BLOCK_ROWS))
    kernel(a, b, c; ndrange )
end
