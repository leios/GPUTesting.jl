export performant_TRMM_dag!

@kernel function performant_TRMM_dag_kernel(A, B )
    




end



function performant_TRMM_dag!(A, B, LIMIT = 16 ; n_threads = (16,16))

    #resize
     
    k  = 2^(Int(ceil(log(2, size(A)[2]))))
    A_2 = zeros(k, k)
    B_2 = zeros(k ,k)

    A_2[1 : size(A)[2] , 1:size(A)[2]] = A
    B_2[1 : size(A)[2] , 1:size(A)[2]] = @view(B[1:end, 1:end])
    size_a = div(k, 2)

    # recursive_TRMM!(@view(A_pad[1:end, 1:end]), @view(B_pad[1:end, 1:end]), div(k,2))
    recursive_TRMM_dag!(A_2, @view(B_2[1:end, 1:end]), size_a, LIMIT)

    B .= @view(B_2[1:size(A)[2], 1:size(A)[2]])
end

#recursive function
function recursive_TRMM_dag!(A_2, B_2, size_a, LIMIT = 16)

    if (size_a < LIMIT)

        # Forgot to add datadeps for the base case
        # Dagger.spawn(() -> begin
        #     #step 1 TRMM
        #     B_2[size_a+1:end , 1:size_a]        = A_2[size_a+1:end ,size_a+1: end] * B_2[size_a+1:end, 1:size_a]
        #     B_2[size_a+1:end , size_a+1:end] = A_2[size_a+1:end ,size_a+1: end] * B_2[size_a+1:end, size_a+1:end]

        #     #step 2
        #     B_2[size_a+1:end, 1:size_a]        =  B_2[size_a+1:end ,1:size_a] + (A_2[size_a+1:end , 1:size_a] * B_2[1:size_a,1:size_a])
        #     B_2[size_a+1:end, size_a+1:end] =   B_2[size_a+1:end ,size_a+1:end] + (A_2[size_a+1:end , 1:size_a] * B_2[1:size_a, size_a+1:end])

        #     #step 3
        #     B_2[1:size_a, 1:size_a] = A_2[ 1:size_a , 1:size_a] * B_2[1:size_a , 1:size_a]
        #     B_2[1:size_a , size_a+1:end] = A_2[1:size_a , 1:size_a] * B_2[1:size_a , size_a+1:end]
        # end)

        Dagger.spawn_datadeps() do
            #step 1 TRMM
            Dagger.@spawn  inplace_mult(A_2, InOut(@view(B_2[1:end, 1:end])))
            # Dagger.@spawn  inplace_mult(A_2[size_a+1:end ,size_a+1: end] , InOut(@view(B_2[size_a+1:end, size_a+1:end])))

            # #step 2
            # Dagger.@spawn mult(A_2[size_a + 1: end , 1: size_a], B_2[1:size_a, 1:size_a],  InOut(@view(B_2[size_a+1:end ,1:size_a])))
            # Dagger.@spawn mult(A_2[size_a + 1: end , 1: size_a], B_2[1:size_a ,size_a + 1: end], InOut(@view(B_2[size_a+1:end ,size_a+1:end])))

            # #step 3
            # Dagger.@spawn  inplace_mult(A_2[ 1:size_a , 1:size_a] , InOut(@view(B_2[1:size_a , 1:size_a])))
            # Dagger.@spawn  inplace_mult(A_2[1:size_a , 1:size_a] , InOut(@view(B_2[1:size_a , size_a+1:end])))
        end
        
        

    else
        # recursive case
        h_size = div(size_a, 2)
        
        Dagger.spawn_datadeps() do
        #step 1
            Dagger.@spawn recursive_TRMM!(A_2[h_size+1:end, h_size+1:end], InOut(@view(B_2[h_size+1:end ,1:h_size])), h_size, LIMIT)
            Dagger.@spawn recursive_TRMM!(A_2[h_size+1:end, h_size+1:end], InOut(@view(B_2[h_size+1:end ,h_size+1:end])), h_size, LIMIT)

        #step 2: GEMM
            Dagger.@spawn mult(A_2[h_size + 1: end , 1: h_size], B_2[1:h_size, 1:h_size],  InOut(@view(B_2[h_size+1:end ,1:h_size])))
            Dagger.@spawn mult(A_2[h_size + 1: end , 1: h_size], B_2[1:h_size ,h_size + 1: end], InOut(@view(B_2[h_size+1:end ,h_size+1:end])))
        
        #step 3
            Dagger.@spawn recursive_TRMM!(A_2[1: h_size, 1: h_size] , InOut(@view(B_2[1: h_size, 1: h_size])), h_size, LIMIT)
            Dagger.@spawn recursive_TRMM!(A_2[ 1: h_size, 1: h_size], InOut(@view(B_2[1: h_size, h_size + 1: end])), h_size, LIMIT)
        end
    end

end

function mult(A, B, C)
    D = A*B + C
    copyto!(C, D)
end

# in-place multiplication
function inplace_mult(A,B)
    D = A*B
    copyto!(B, D)
end

