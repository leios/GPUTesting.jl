export h_decomp

using LinearAlgebra

# h_decomp: function that subdivides the matrix to 4 blocks
#           finds the maximum norm of the off diagonal blocks 
#           change all the elements in that level to the value of that norm
#           repeat the process for the diagonal blocks



# proxy to determine if the subdivision is consistent with 
# the paper subdivision by comparing the distribution of norms at different levels

function h_decomp(A, n_min, leaf=false)
    N = size(A, 1)
    @assert N == size(A, 2) "Matrix must be square in shape"

    result = zeros(Float64, size(A)) # display of the norm distribution in the submatrices
    vertices = [[(1, 1), (N, N)]]  # [(top-left, bottom-right)] coordinates
    sub_length = size(A,1)
    
    

    half_length = sub_length
    while ((half_length/2) >= n_min)
        #partition the submatrices
        new_vertices_diag = []
        new_vertices_offdiag = []
        frob_norms_level = []
        half_length = Int(ceil(half_length/2))

        for vertex in vertices

            # diagonal submatrices
            top_left_vertices = [[vertex[1][1], vertex[1][2]], [vertex[1][1] + half_length-1,vertex[1][2] + half_length-1]]
            bottom_right_vertices = [[vertex[1][1] + half_length, vertex[1][2] + half_length], [vertex[2][1] ,vertex[2][2]]]

            # off diagonal submatrices
            off_diag_left = [[vertex[1][1] + half_length, vertex[1][2]], [vertex[2][1], vertex[1][2] + half_length -1]]
            off_diag_right = [[vertex[1][1], vertex[1][2]+ half_length], [vertex[1][1]+half_length-1, vertex[2][2]]]
            

            off_diag_norm_left = norm(@view(A[off_diag_left[1][1] : off_diag_left[2][1], off_diag_left[1][2] : off_diag_left[2][2]]))
            off_diag_norm_right = norm(@view(A[off_diag_right[1][1] : off_diag_right[2][1], off_diag_right[1][2] : off_diag_right[2][2]]))


            #result[off_diag_left[1][1] : off_diag_left[2][1], off_diag_left[1][2] : off_diag_left[2][2]] .= off_diag_norm_left
            #result[off_diag_right[1][1] : off_diag_right[2][1], off_diag_right[1][2] : off_diag_right[2][2]] .= off_diag_norm_right

            push!(frob_norms_level, off_diag_norm_left, off_diag_norm_right)
            push!(new_vertices_offdiag, off_diag_left, off_diag_right)
            push!(new_vertices_diag, top_left_vertices, bottom_right_vertices)
        end

        # find the maximum norm at the level
        # then change the elements on this level to the maximum norm in the result matrix 
        level_norm = maximum(frob_norms_level)

        for vertex_offdiag in new_vertices_offdiag
            result[vertex_offdiag[1][1] : vertex_offdiag[2][1], vertex_offdiag[1][2] : vertex_offdiag[2][2]] .= level_norm
        end

        vertices = new_vertices_diag
        
    end
    

    if (leaf)
        # computing the norm for the final level
        # changing the diagonal blocks elements to the computed norm
        leaf_norms = []
        for vertex in vertices
            l_norm = norm(@view(A[vertex[1][1] : vertex[2][1], vertex[1][2] : vertex[2][2]]))
            push!(leaf_norms, l_norm)
        end
        
        norm_diag = maximum(leaf_norms)

        for vertex_diag in vertices
            result[vertex_diag[1][1] : vertex_diag[2][1], vertex_diag[1][2] : vertex_diag[2][2]] .= norm_diag
        end
    end

    return result
end