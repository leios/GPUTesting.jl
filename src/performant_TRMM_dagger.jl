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
        Dagger.spawn(() -> begin
            #step 1 TRMM
            B_2[size_a+1:end , 1:size_a]        = A_2[size_a+1:end ,size_a+1: end] * B_2[size_a+1:end, 1:size_a]
            B_2[size_a+1:end , size_a+1:end] = A_2[size_a+1:end ,size_a+1: end] * B_2[size_a+1:end, size_a+1:end]

            #step 2
            B_2[size_a+1:end, 1:size_a]        =  B_2[size_a+1:end ,1:size_a] + (A_2[size_a+1:end , 1:size_a] * B_2[1:size_a,1:size_a])
            B_2[size_a+1:end, size_a+1:end] =   B_2[size_a+1:end ,size_a+1:end] + (A_2[size_a+1:end , 1:size_a] * B_2[1:size_a, size_a+1:end])

            #step 3
            B_2[1:size_a, 1:size_a] = A_2[ 1:size_a , 1:size_a] * B_2[1:size_a , 1:size_a]
            B_2[1:size_a , size_a+1:end] = A_2[1:size_a , 1:size_a] * B_2[1:size_a , size_a+1:end]
        end)
        
        

    else
        # recursive case
        h_size = div(size_a, 2)
        #step 1
        Dagger.@spawn recursive_TRMM!(A_2[h_size+1:end, h_size+1:end], @view(B_2[h_size+1:end ,1:h_size]), h_size, LIMIT)
        Dagger.@spawn recursive_TRMM!(A_2[h_size+1:end, h_size+1:end], @view(B_2[h_size+1:end ,h_size+1:end]), h_size, LIMIT)

        #step 2: GEMM
        Dagger.spawn(() -> begin
            B_2[h_size+1:end ,1:h_size]        =  B_2[h_size + 1: end ,1:h_size] + (A_2[h_size + 1: end , 1: h_size] * B_2[1:h_size, 1:h_size])
            B_2[h_size+1:end ,h_size+1:end] =   B_2[h_size + 1: end ,h_size + 1: end] + (A_2[h_size + 1: end , 1: h_size] * B_2[1:h_size ,h_size + 1: end])
        end)
        #B_2[h_size+1:end ,1:h_size]        =  B_2[h_size + 1: end ,1:h_size] + (A_2[h_size + 1: end , 1: h_size] * B_2[1:h_size, 1:h_size])
        #B_2[h_size+1:end ,h_size+1:end] =   B_2[h_size + 1: end ,h_size + 1: end] + (A_2[h_size + 1: end , 1: h_size] * B_2[1:h_size ,h_size + 1: end])


        #step 3
        Dagger.@spawn recursive_TRMM!(A_2[1: h_size, 1: h_size] , @view(B_2[1: h_size, 1: h_size]), h_size, LIMIT)
        Dagger.@spawn recursive_TRMM!(A_2[ 1: h_size, 1: h_size], @view(B_2[1: h_size, h_size + 1: end]), h_size, LIMIT)

    end

end
