
# implementation of the TRMM low RHS
# 
# The base case of the recursion operates on small triangular blocks 
# located in the diagonal of matrix A, along with a thin and wide portion
# of matrix B.
#
# Time breakdown for RecTRMM and for square and tall matrix profiles
# shows that TRMM kernel calls for thin and wide matrices consumes a large 
# percentage of the overall execution time: mostly due to low hardware 
# occupancy caused by a limited concurrency.
#
# This provides motivation for optimizing the native TRMM kernels for small 
# number of rows, involving diagonal blocks with thin and wide output matrices.
# 
# May improve the overall RecTRMM performances for larger matrix sizes.
#
# The low level optimizations : (1) increase concurrency for better hardware occupancy
# (2) tile algorithm for better caching and register reuse (3)

@kernel function native_TRMM_kernel!(M, N, A, B,
                                    ::Val{BANK} = Val(1)) where BANK
    # A is an MxM matrix
    # B is an MXN matrix

    # The group and local indices needed to compute the global indices
    # cannot use @index(Global) as we use a smaller ndrange in kernel launch
    gi, gj = @index(Group, NTuple)
    i, j = @index(Local, NTuple)

    TILE_DIM = @uniform @groupsize()[1]
    BLOCK_ROWS = @uniform @groupsize()[2]

    # Allocating shared memory as in tile algorithm in the low level optimizations
    # BANK added to avoid bank conflicts as a result of irregular thread access
    TILE_1 = @localmem eltype(B) (TILE_DIM+BANK, TILE_DIM)

    # declaring a private element(in place of a register) to accumulate the dot product
    C_sub = @private eltype(C) 1
    @inbounds C_sub[1] = zero(eltype(C))

    @uniform NUM_TILES = div(M + TILE_DIM - 1, TILE_DIM)

    for c in 0:(NUM_TILES-1)

        sum_out = zero(eltype(B))

        I = (gi-1)*TILE_DIM + i
        J = (gj-1)*TILE_DIM + j

        #load tile A into from global to shared memory

        #load tile B(c) into registers



    end
end
