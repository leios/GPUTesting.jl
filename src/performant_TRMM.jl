export performant_TRMM!

@kernel function performant_TRMM_kernel(A, B )
    




end



function performant_TRMM!(A, B, LIMIT = 16 ; n_threads = (16,16))

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
function recursive_TRMM!(A_2, B_2, size_a, LIMIT = 16)

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

        B_2 = A_2 * B_2

    else
        B00 = copy(B_2[1: h_size, 1: h_size])
        B01 = copy(B_2[1: h_size, h_size + 1: end])

        # recursive case
        h_size = div(size_a, 2)
        #step 1
        recursive_TRMM!(A_2[h_size+1:end, h_size+1:end], @view(B_2[h_size+1:end ,1:h_size]), h_size, LIMIT)
        recursive_TRMM!(A_2[h_size+1:end, h_size+1:end], @view(B_2[h_size+1:end ,h_size+1:end]), h_size, LIMIT)

        #step 2: GEMM
        B00 =  (A_2[h_size + 1: end , 1: h_size] * B00)
        B01 =    (A_2[h_size + 1: end , 1: h_size] * B01)

        B_2[h_size+1:end ,1:h_size]        =  B_2[h_size + 1: end ,1:h_size] + B00
        B_2[h_size+1:end ,h_size+1:end] = B_2[h_size+1:end ,h_size+1:end] + B01

        #step 3
        recursive_TRMM!(A_2[1: h_size, 1: h_size] , @view(B_2[1: h_size, 1: h_size]), h_size, LIMIT)
        recursive_TRMM!(A_2[ 1: h_size, 1: h_size], @view(B_2[1: h_size, h_size + 1: end]), h_size, LIMIT)

    end

end
