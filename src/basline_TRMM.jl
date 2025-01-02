export baseline_trmm!

# start: start of block A
# end_index: end of block A
# unit_index: whether A is unit triangular
# upper_index: whether A is an upper triaingular matrix


function baseline_trmm_recursive!(Afull, Bfull, start_index, end_index, tileSizeA, tileSizeB, nthreads)
    size_tile = end_index - start_index + 1

    # if the matrix is small enough, call the computation kernel directly for the block
    if size_tile <= tileSizeA
        # set the kernel arguments
        A = @view(Afull[start_index:end_index, start_index: end_index])
        B = @view(Bfull[start_index:end_index, 1:end])
        

    
        BLAS.trmm!('L', 'L', 'N', 'N', Float32(1), A, B) #BLAS.trmm!
        
    
    else
        # split at the next multiple of the TileSize
        split = div(size_tile, 2)
        
         

        # considering the lower triangular case first
        baseline_trmm_recursive!(Afull, Bfull, start_index+split, end_index, tileSizeA, tileSizeB, nthreads)
        base_gemm!(Afull, Bfull, start_index+split, end_index, start_index, start_index+split - 1, start_index, start_index + split - 1, end_index)        
        baseline_trmm_recursive!(Afull, Bfull, start_index, start_index+split-1, tileSizeA, tileSizeB, nthreads)

    end
end




# holder wrapper for the kernel

function baseline_trmm!(A, B)
    if size(A)[1] != size(A)[2]
        error("Dimension mismatch: Matrix A must be triangular!")
    end

    if size(A)[2] != size(B)[1]
        error("Matrix A and B not compatible for matrix product!")
    end

    TILE_SIZE_A = 32
    TILE_SIZE_B = 32
    nthreads = 16

    baseline_trmm_recursive!(A, B, 1, size(A)[1], TILE_SIZE_A, TILE_SIZE_B, nthreads)

end


function base_gemm!(Afull, Bfull, ll_startR, ll_endR, ll_startC, ll_endC, b_upper_start, b_upper_end, end_index)

    A = @view(Afull[ll_startR:ll_endR, ll_startC:ll_endC])
    B = @view(Bfull[b_upper_start:b_upper_end, 1:end])
    C = @view(Bfull[b_upper_end+1:end_index, 1:end])

    
    
    BLAS.gemm!('N', 'N', Float32(1), A, B, Float32(1), C)   
end
