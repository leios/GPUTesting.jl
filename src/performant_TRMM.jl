export performant_TRMM!

@kernel function TRMM_base_kernel!(A, B )
    i,j = @index(Global, NTuple)

    # a x b = c => c[row_i, col_j] = sum(a[row_i] * b[col_j])


    #loop for addition, iterations = size of col of a or row of b

    temp_sum = 0

    for a_col in 1:size(A)[2]
        temp_sum = temp_sum + A[i,a_col] * B[a_col, j]
    end


    B[i,j] = temp_sum
end
 




@kernel function GEMM_TRMM_kernel!(A, B, 
                                    ::Val{BANK} = Val(1)) where BANK
    
    gi,gj = @index(Group, NTuple)
    i,j = @index(Local, NTuple)

    TILE_DIM = @uniform @groupsize()[1]
    BLOCK_ROWS = @uniform @groupsize()[2]

    #allocating shared memory for the sub matrix product calculation
    #BANK = 1, added to avoid bank coonflicts as a result of irregular thread access
    tile1 = @localmem eltype(B) (TILE_DIM+BANK, TILE_DIM)
    tile2 = @localmem eltype(B) (TILE_DIM+BANK, TILE_DIM)

    #declaring a private variable to accumulate the result of submatrix multiplication
    C_sub = @private eltype(B) 1
    @inbounds C_sub[1] = -zero(eltype(B))

    @uniform N = size(A, 1)
    @uniform R = size(A, 2)
    @uniform M = size(B, 2)


    #the number of tiles required will be dependent on the inner dimensions
    @uniform NUM_TILES = div(R + TILE_DIM - 1, TILE_DIM)

    #loop over all tiles needed for the calculation
    for t in 0:(NUM_TILES-1)
        # Cannot use @index(Global), because we use a smaller ndrange(gridsize would reduce)
        I = (gi-1) * TILE_DIM + i
        J = (gj-1) * TILE_DIM + j

        # load inputs into tiles, with bounds checking for non-square matrices
        if I <= N && t*TILE_DIM + j <= R
            @inbounds tile1[i, j] = A[I, t*TILE_DIM + j]
        else
            @inbounds tile1[i, j] = 0.0
        end
        if t*TILE_DIM + i <= R && J <= M
            @inbounds tile2[i, j] = B[t*TILE_DIM + i, J]
        else
            @inbounds tile2[i, j] = 0.0
        end

        # wait for all tiles to be loaded
        @synchronize

        # get global values again (because of synchronize?)
        I = (gi-1) * TILE_DIM + i
        J = (gj-1) * TILE_DIM + j

        # calculate value of spot in output, use temporary value to allow for vectorization
        out = zero(eltype(B))
        @simd for k in 1:TILE_DIM
            @inbounds out += tile1[i, k] * tile2[k, j]
        end
        C_sub[1] += out

        @synchronize
    end

    # get global indices again
    I = (gi-1) * TILE_DIM + i
    J = (gj-1) * TILE_DIM + j

    # save if inbounds
    if I <= N && J <= M
        @inbounds B[I, J] = C_sub[1] + B[I,J]
    end
end



function GEMM_TRMM!(A, B; n_threads = (16,16))

    backend = get_backend(A)
    kernel = GEMM_TRMM_kernel!(backend, n_threads)
    padded_b = (size(B,1)+16, size(B,2)+16)
    kernel(A, B; ndrange = padded_b)
end

function TRMM_base!(A, B; n_threads = 256)

    backend = get_backend(A)
    kernel = TRMM_base_kernel!(backend, n_threads)
    kernel(A,B; ndrange = size(B))
end





function performant_TRMM!(A, B, LIMIT = 16)

    #resize
     
    k  = 2^(Int(ceil(log(2, size(A)[2]))))
    ArrayType = typeof(A).name.wrapper
    A_2 = ArrayType(zeros(eltype(A) , k, k))
    B_2 = ArrayType(zeros(eltype(A), k ,k))
    A_2[1 : size(A)[2] , 1:size(A)[2]] .= A
    B_2[1 : size(A)[2] , 1:size(A)[2]] .= @view(B[1:end, 1:end])
    size_a = div(k, 2)

    #recursive_TRMM!(@view(A_pad[1:end, 1:end]), @view(B_pad[1:end, 1:end]), div(k,2))
    #timing = Metal.@elapsed recursive_TRMM!(A_2, @view(B_2[1:end, 1:end]), size_a, LIMIT)
    recursive_TRMM!(A_2, @view(B_2[1:end, 1:end]), size_a, LIMIT)

    B .= @view(B_2[1:size(A)[2], 1:size(A)[2]])
end

#recursive function
function recursive_TRMM!(A_2, B_2, size_a, LIMIT = 128)

    if (size_a < LIMIT)

        # b00 = copy(B_2[1:size_a,1:size_a])
        # b01 = copy(B_2[1:size_a, size_a+1:end])

        # #step 1 TRMM
        # B_2[size_a+1:end , 1:size_a]        = A_2[size_a+1:end ,size_a+1: end] * B_2[size_a+1:end, 1:size_a]
        # B_2[size_a+1:end , size_a+1:end] = A_2[size_a+1:end ,size_a+1: end] * B_2[size_a+1:end, size_a+1:end]

        # #step 2
        # b1 = (A_2[size_a+1:end , 1:size_a] * B_2[1:size_a,1:size_a])
        # b2 = (A_2[size_a+1:end , 1:size_a] * B_2[1:size_a, size_a+1:end])

        # B_2[size_a+1:end, 1:size_a] = B_2[size_a+1:end ,1:size_a] + b1
        # B_2[size_a+1:end, size_a+1:end] =   B_2[size_a+1:end ,size_a+1:end] + b2

        # #B_2[size_a+1:end, 1:size_a]        =  B_2[size_a+1:end ,1:size_a] + (A_2[size_a+1:end , 1:size_a] * B_2[1:size_a,1:size_a])
        # #B_2[size_a+1:end, size_a+1:end] =   B_2[size_a+1:end ,size_a+1:end] + (A_2[size_a+1:end , 1:size_a] * B_2[1:size_a, size_a+1:end])

        # #step 3 TRMM 
        # B_2[1:size_a, 1:size_a] = A_2[ 1:size_a , 1:size_a] * b00
        # B_2[1:size_a , size_a+1:end] = A_2[1:size_a , 1:size_a] * b01

        ##################
        # B_2 = A_2 * B_2
        TRMM_base!(A_2, B_2)
        ####################

    else
        # B00 = copy(B_2[1: h_size, 1: h_size])
        # B01 = copy(B_2[1: h_size, h_size + 1: end])

        # recursive case
        h_size = div(size_a, 2)
        #step 1
        recursive_TRMM!(A_2[h_size+1:end, h_size+1:end], @view(B_2[h_size+1:end ,1:h_size]), h_size, LIMIT)
        recursive_TRMM!(A_2[h_size+1:end, h_size+1:end], @view(B_2[h_size+1:end ,h_size+1:end]), h_size, LIMIT)

        #step 2: GEMM: use parallelism
        # B00 =  (A_2[h_size + 1: end , 1: h_size] * B00)
        # B01 =    (A_2[h_size + 1: end , 1: h_size] * B01)

        # B_2[h_size+1:end ,1:h_size]        =  B_2[h_size + 1: end ,1:h_size] + B00
        # B_2[h_size+1:end ,h_size+1:end] = B_2[h_size+1:end ,h_size+1:end] + B01

        ###########################################

        GEMM_TRMM!(A_2[h_size + 1: end , 1: h_size], (B_2[h_size+1:end ,1:h_size]))
        GEMM_TRMM!(A_2[h_size + 1: end , 1: h_size], (B_2[h_size+1:end ,h_size+1:end]))

        ###########################################

        #step 3
        recursive_TRMM!(A_2[1: h_size, 1: h_size] , @view(B_2[1: h_size, 1: h_size]), h_size, LIMIT)
        recursive_TRMM!(A_2[ 1: h_size, 1: h_size], @view(B_2[1: h_size, h_size + 1: end]), h_size, LIMIT)

    end

end
