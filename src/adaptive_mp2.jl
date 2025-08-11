export adaptive_precision_k


# computes and outputs the adaptive working precision level u' for 
# different levels of a triangular matrix, relative to some
# approximation parameter epsilon = 1e-8 by default

using LinearAlgebra # for the inbuilt Frobenius norm function


# Function to select the adaptive precision at each level
# A : Lower Triangular matrix
# n_min : minimal diiagonal block size
# U : vector of the available precision levels expressed as the unit roundoff errors 
    # computed from the machine precision levels
function adaptive_precision_k(A::AbstractMatrix,n_min = 4, epsilon = 1e-8)
    
    # creating the array of roundoff errors (working precision)
    u16 = eps(Float16)/2
    u32 = eps(Float32)/2
    u64 = eps(Float64)/2

    # Hardcoding the rest for convenience (will change later, or create another module)
    q52 = 1.25e-1 # quarter precision
    bf16 = 3.91e-3 # bfloat16


    U = Float64[q52, bf16, u16, u32, u64] # precision set
    U_string = ["q52", "bf16", "f16", "f32", "f64"]

    u_approx = [] # will hold the working precisions

    norm_A = norm(A)

    N = size(A, 1)
    @assert N == size(A, 2) "Matrix must be square in shape"

    vertices = [[(1, 1), (N, N)]]  # [(top-left, bottom-right)] coordinates

    half_length = N
    k = 0 # the current level

    while ((half_length/2) >= n_min)
        #partition the submatrices
        new_vertices_diag = []
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
            
            push!(new_vertices_diag, top_left_vertices, bottom_right_vertices)
        end

        # find the maximum norm at the level
        # then change the elements on this level to the maximum norm in the result matrix 
        level_norm = maximum(frob_norms_level)
        
        E_k = level_norm/norm_A
        u_work = epsilon/((2^((k+1)/2))*E_k)

        i = 1
        while (u_work < U[i])
            i += 1
        end

        push!(u_approx, U_string[i])


        vertices = new_vertices_diag
        k = k+1
        
    end
    

    # if (leaf)
    #     # computing the norm for the final level
    #     # changing the diagonal blocks elements to the computed norm
    #     leaf_norms = []
    #     for vertex in vertices
    #         l_norm = norm(@view(A[vertex[1][1] : vertex[2][1], vertex[1][2] : vertex[2][2]]))
    #         push!(leaf_norms, l_norm)
    #     end
        
    #     norm_diag = maximum(leaf_norms)

    #     for vertex_diag in vertices
    #         result[vertex_diag[1][1] : vertex_diag[2][1], vertex_diag[1][2] : vertex_diag[2][2]] .= norm_diag
    #     end
    # end

    return u_approx

end


