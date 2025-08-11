export run_benchmarks

using LinearAlgebra
using BenchmarkTools
# --- Main Benchmarking Function ---
"""
Runs a comprehensive set of benchmarks for TriangularBlockMatrix creation,
access, and memory consumption, comparing against standard Julia matrices.

Arguments:
- `matrix_sizes::Vector{Int}`: An array of matrix dimensions (e.g., [128, 256, 512]).
- `num_levels_array::Vector{Int}`: An array of hierarchical depths (l) for TriangularBlockMatrix.
- `num_queries::Int`: Number of random accesses to perform for access time benchmarking.

Returns a dictionary with benchmarking results for plotting.
Keys: :access_upper, :access_lower, :access_normal, :generation_upper, :generation_lower,
      :memory_upper, :memory_lower, :memory_normal.
Values: Each value is a Vector of two Vectors: `[sizes_vector, times_or_memory_vector]`.
"""
function run_benchmarks(matrix_sizes::Vector{Int}, num_levels_array::Vector{Int}, num_queries::Int)
    println("--- Comprehensive TriangularBlockMatrix Performance Benchmarks ---")
    println("------------------------------------------------------------------------------------------------------------------------------------")
    println(rpad("Matrix Size", 15), rpad("Levels", 10), rpad("Triangular Type", 20), rpad("Creation Time (min)", 25), rpad("Access Time (min)", 25), rpad("Allocations", 15), "Memory (KiB)")
    println("------------------------------------------------------------------------------------------------------------------------------------")

    triangular_types = [:lower, :upper]
    
    # Initialize results dictionary
    results_dict = Dict{Symbol, Vector{Vector{Any}}}()
    results_dict[:access_upper] = [[], []]
    results_dict[:access_lower] = [[], []]
    results_dict[:access_normal] = [[], []]
    results_dict[:generation_upper] = [[], []]
    results_dict[:generation_lower] = [[], []]
    results_dict[:memory_upper] = [[], []]
    results_dict[:memory_lower] = [[], []]
    results_dict[:memory_normal] = [[], []]


    for matrix_size in matrix_sizes
        for num_levels in num_levels_array
            # --- Prepare matrices for this size and level ---
            dense_matrix_lower = Matrix(LowerTriangular(rand(Float64, matrix_size, matrix_size)))
            dense_matrix_upper = Matrix(UpperTriangular(rand(Float64, matrix_size, matrix_size)))
            
            # For memory comparison, we'll need a full dense matrix
            normal_matrix_full = rand(Float64, matrix_size, matrix_size)

            # --- Generation Time Benchmarks ---
            # Lower TriangularBlockMatrix generation
            trial_gen_lower = @benchmark TriangularBlockMatrix($dense_matrix_lower, $num_levels; triangular_type=:lower) samples=10 evals=5
            gen_time_lower = isempty(trial_gen_lower.times) ? "N/A" : string(minimum(trial_gen_lower.times))
            gen_allocs_lower = isempty(trial_gen_lower.times) ? "N/A" : string(BenchmarkTools.allocs(trial_gen_lower))
            gen_memory_lower = isempty(trial_gen_lower.times) ? "N/A" : string(round(BenchmarkTools.memory(trial_gen_lower) / 1024, digits=2))
            
            # Upper TriangularBlockMatrix generation
            trial_gen_upper = @benchmark TriangularBlockMatrix($dense_matrix_upper, $num_levels; triangular_type=:upper) samples=10 evals=5
            gen_time_upper = isempty(trial_gen_upper.times) ? "N/A" : string(minimum(trial_gen_upper.times))
            gen_allocs_upper = isempty(trial_gen_upper.times) ? "N/A" : string(BenchmarkTools.allocs(trial_gen_upper))
            gen_memory_upper = isempty(trial_gen_upper.times) ? "N/A" : string(round(BenchmarkTools.memory(trial_gen_upper) / 1024, digits=2))

            # Store generation results
            push!(results_dict[:generation_lower][1], matrix_size)
            push!(results_dict[:generation_lower][2], gen_time_lower)
            push!(results_dict[:generation_upper][1], matrix_size)
            push!(results_dict[:generation_upper][2], gen_time_upper)
            # For memory, we store the actual memory taken by the created object, not just allocations during creation
            # We'll create the objects outside the benchmark for memory comparison
            tbm_lower_instance = TriangularBlockMatrix(dense_matrix_lower, num_levels; triangular_type=:lower)
            tbm_upper_instance = TriangularBlockMatrix(dense_matrix_upper, num_levels; triangular_type=:upper)
            
            mem_upper_kib = round(Base.summarysize(tbm_upper_instance) / 1024, digits=2)
            mem_lower_kib = round(Base.summarysize(tbm_lower_instance) / 1024, digits=2)
            mem_normal_kib = round(Base.summarysize(normal_matrix_full) / 1024, digits=2)

            push!(results_dict[:memory_upper][1], matrix_size)
            push!(results_dict[:memory_upper][2], mem_upper_kib)
            push!(results_dict[:memory_lower][1], matrix_size)
            push!(results_dict[:memory_lower][2], mem_lower_kib)
            push!(results_dict[:memory_normal][1], matrix_size)
            push!(results_dict[:memory_normal][2], mem_normal_kib)


            # --- Access Time Benchmarks ---
            # Generate random query indices for access benchmarks
            query_indices = Vector{Tuple{Int, Int}}(undef, num_queries)
            for k in 1:num_queries
                i = rand(1:matrix_size)
                j = rand(1:matrix_size)
                query_indices[k] = (i, j)
            end

            # Access time for Lower TriangularBlockMatrix
            trial_access_lower = @benchmark begin
                for (i, j) in $query_indices
                    val = fast_get($tbm_lower_instance, i, j)
                end
            end samples=10 evals=5
            access_time_lower = isempty(trial_access_lower.times) ? "N/A" : string(minimum(trial_access_lower.times))
            
            # Access time for Upper TriangularBlockMatrix
            trial_access_upper = @benchmark begin
                for (i, j) in $query_indices
                    val = fast_get($tbm_upper_instance, i, j)
                end
            end samples=10 evals=5
            access_time_upper = isempty(trial_access_upper.times) ? "N/A" : string(minimum(trial_access_upper.times))

            # Access time for normal dense matrix
            trial_access_normal = @benchmark begin
                for (i, j) in $query_indices
                    val = fast_get_normal_matrix($normal_matrix_full, i, j)
                end
            end samples=10 evals=5
            access_time_normal = isempty(trial_access_normal.times) ? "N/A" : string(minimum(trial_access_normal.times))
            
            # Store access results
            push!(results_dict[:access_lower][1], matrix_size)
            push!(results_dict[:access_lower][2], access_time_lower)
            push!(results_dict[:access_upper][1], matrix_size)
            push!(results_dict[:access_upper][2], access_time_upper)
            push!(results_dict[:access_normal][1], matrix_size)
            push!(results_dict[:access_normal][2], access_time_normal)


            # --- Print Results for Current Iteration ---
            println(rpad(matrix_size, 15), rpad(num_levels, 10), rpad("Lower", 20), rpad(gen_time_lower, 25), rpad(access_time_lower, 25), rpad(gen_allocs_lower, 15), gen_memory_lower)
            println(rpad("", 15), rpad("", 10), rpad("Upper", 20), rpad(gen_time_upper, 25), rpad(access_time_upper, 25), rpad(gen_allocs_upper, 15), gen_memory_upper)
            println(rpad("", 15), rpad("", 10), rpad("Normal Matrix (Access)", 20), rpad("N/A", 25), rpad(access_time_normal, 25), rpad("N/A", 15), mem_normal_kib)
            println("------------------------------------------------------------------------------------------------------------------------------------")
        end
    end
    
    println("\n--- Benchmarks Complete ---")
    println("Note: For precise benchmarks, increase `samples` and `evals` in `@benchmark`.")
    println("Returned dictionary contains data for plotting.")

    return results_dict
end

# Example Usage:
# To run, copy all the code above into a Julia REPL or a .jl file and execute it.
# Then call the benchmark function:
# matrix_sizes_to_test = [128, 256, 512, 1024]
# levels_to_test = [2, 3, 4] # Number of levels for the block matrix
# num_access_queries = 10000 # Number of random accesses to benchmark
#
# benchmark_results = run_benchmarks(matrix_sizes_to_test, levels_to_test, num_access_queries)
#
# # You can now access the data for plotting, e.g.:
# # println("Access Times for Upper Block Matrix (Sizes): ", benchmark_results[:access_upper][1])
# # println("Access Times for Upper Block Matrix (Times): ", benchmark_results[:access_upper][2])



function run_trmm_benchmarks(matrix_sizes::Vector{Int})
    all_timings = Vector{Vector{Float64}}()
    
    println("Starting TRMM benchmark for lower triangular matrices...")
    
    # We'll use a fixed number of levels (l) for simplicity, which will be
    # automatically determined based on the matrix size to ensure the block
    # sizes are reasonable powers of 2.
    for n in matrix_sizes
        # We need to choose an appropriate number of levels for each matrix size n.
        # This can be found by a simple integer logarithm.
        l = Int(floor(log2(n)))
        
        println("\n--- Benchmarking n = $n (l = $l) ---")
        
        # Run the benchmark function which returns a dictionary of results
        results = benchmark_trmm_trms_speed(n, l, :lower)
        
        # Extract the relevant timing data and push to the results vector
        base_time = results[:trmm_base_time_ms]
        recursive_time = results[:trmm_recursive_time_ms]
        
        push!(all_timings, [base_time, recursive_time])
    end
    
    println("\nTRMM benchmark complete.")
    
    return all_timings
end



function benchmark_trmm_comparison(matrix_sizes::Vector{Int})
    all_timings = Vector{Vector{Float64}}()
    
    println("Starting TRMM benchmark for lower triangular matrices...")
    
    # --- Benchmark Loop ---
    for n in matrix_sizes
        # Determine the number of decomposition levels (l) based on the matrix size.
        l = Int(floor(log2(n)))
        
        println("\n--- Benchmarking n = $n (l = $l) ---")
        
        # 1. Create a reference lower triangular matrix (A_dense) and a test matrix (B_mat).
        A_dense = tril(rand(n, n))
        B_mat = rand(n, n)
        
        # 2. Create the TriangularBlockMatrix (A_tbm) from A_dense.
        A_tbm = TriangularBlockMatrix(A_dense, l; triangular_type=:lower)

        # 3. Benchmark the base multiplication (A_dense * B_mat).
        println("   Benchmarking Base TRMM...")
        base_trmm_time = @benchmark $A_dense * $B_mat samples=5 evals=1
        base_time_ms = median(base_trmm_time).time / 1_000_000
        @printf "   Base TRMM time: %.3f ms (median)\n" base_time_ms

        # 4. Benchmark the recursive multiplication (trmm(A_tbm, B_mat)).
        println("   Benchmarking Recursive TRMM...")
        recursive_trmm_time = @benchmark trmm($A_tbm, $B_mat) samples=5 evals=1
        recursive_time_ms = median(recursive_trmm_time).time / 1_000_000
        @printf "   Recursive TRMM time: %.3f ms (median)\n" recursive_time_ms
        
        # Store the results.
        push!(all_timings, [base_time_ms, recursive_time_ms])
    end
    
    println("\nTRMM benchmark complete.")

    return all_timings
end
