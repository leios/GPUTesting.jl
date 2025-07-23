using BenchmarkTools
"""
Benchmarks the indexing speed of TriangularBlockMatrix vs. standard Matrix.

Arguments:
- `matrix_size::Int`: The dimension of the square matrix (e.g., 1024).
- `num_levels::Int`: The number of hierarchical levels (l).
- `triangular_type::Symbol`: :lower or :upper.
- `num_queries::Int`: Number of random indices to query for benchmarking.
"""
function benchmark_indexing_speed(matrix_size::Int, num_levels::Int, triangular_type::Symbol, num_queries::Int)
    println("--- Benchmarking Indexing Speed (N=$matrix_size, L=$num_levels, Type=$triangular_type) ---")

    # 1. Create a random dense matrix and then make it triangular
    println("Creating test matrices...")
    if triangular_type == :lower
        dense_matrix = tril(rand(Float64, matrix_size, matrix_size))
    else # :upper
        dense_matrix = triu(rand(Float64, matrix_size, matrix_size))
    end

    # 2. Construct the TriangularBlockMatrix
    tbm = TriangularBlockMatrix(dense_matrix, num_levels; triangular_type=triangular_type)
    println("TriangularBlockMatrix created.")

    # 3. Generate random query indices
    println("Generating $num_queries random query indices...")
    query_indices = Vector{Tuple{Int, Int}}(undef, num_queries)
    for k in 1:num_queries
        i = rand(1:matrix_size)
        j = rand(1:matrix_size)
        query_indices[k] = (i, j)
    end
    println("Query indices generated.")

    # 4. Benchmark standard Matrix indexing
    println("\nBenchmarking standard Matrix indexing...")
    # Use a loop to iterate through queries to get a more realistic average time per access
    # @btime measures the total time for the expression, including loop overhead.
    # To get per-access time, we can divide or use a larger loop.
    # For simplicity, let's benchmark the entire query set.
    
    # Benchmark standard matrix indexing
    std_matrix_times = @btime begin
        for (i, j) in $query_indices
            val = $dense_matrix[i, j]
        end
    end samples=5 evals=1 # Small samples/evals for quick test, increase for precision

    println("Standard Matrix Indexing Time: $std_matrix_times")

    # 5. Benchmark TriangularBlockMatrix indexing using fast_get
    println("\nBenchmarking TriangularBlockMatrix indexing (fast_get)...")
    tbm_fast_get_times = @btime begin
        for (i, j) in $query_indices
            val = fast_get($tbm, i, j)
        end
    end samples=5 evals=1

    println("TriangularBlockMatrix (fast_get) Indexing Time: $tbm_fast_get_times")

    # 6. Benchmark TriangularBlockMatrix indexing using Base.getindex (if overloaded)
    println("\nBenchmarking TriangularBlockMatrix indexing (Base.getindex)...")
    tbm_getindex_times = @btime begin
        for (i, j) in $query_indices
            val = $tbm[i, j]
        end
    end samples=5 evals=1

    println("TriangularBlockMatrix (Base.getindex) Indexing Time: $tbm_getindex_times")

    println("\n--- Benchmark Complete ---")
    println("Note: For precise benchmarks, increase `samples` and `evals` in `@btime`.")
    println("Also, consider running in a fresh Julia session to avoid JIT compilation effects on first run.")
end

# Example Usage:
# Then call the benchmark_indexing_speed function.

# Example 1: Lower Triangular, 1024x1024 matrix, 3 levels, 10000 queries
# benchmark_indexing_speed(1024, 3, :lower, 10000)

# Example 2: Upper Triangular, 512x512 matrix, 2 levels, 5000 queries
# benchmark_indexing_speed(512, 2, :upper, 5000)

# Example 3: Smaller matrix, more levels (deeper tree)
# benchmark_indexing_speed(64, 5, :lower, 1000)















function benchmark_structure_creation_time(matrix_sizes::Vector{Int}, num_levels_array::Vector{Int})
    println("--- Benchmarking TriangularBlockMatrix Creation Time ---")
    println("--------------------------------------------------------------------")
    println(rpad("Matrix Size", 15), rpad("Levels", 10), rpad("Triangular Type", 20), "Creation Time (min)")
    println("--------------------------------------------------------------------")

    triangular_types = [:lower, :upper]

    for matrix_size in matrix_sizes
        for num_levels in num_levels_array
            for triangular_type in triangular_types
                # Create a random dense matrix and then make it triangular
                if triangular_type == :lower
                    dense_matrix_source = tril(rand(Float64, matrix_size, matrix_size))
                else # :upper
                    dense_matrix_source = triu(rand(Float64, matrix_size, matrix_size))
                end

                # U_approx is only used to pass the 'l' parameter to the constructor
                # Create a dummy U_approx of the correct size
                u_approx_dummy = zeros(num_levels, num_levels) 

                # Benchmark TriangularBlockMatrix construction
                # Use `minimum` to get the fastest observed time
                trial = @btime TriangularBlockMatrix($dense_matrix_source, $u_approx_dummy; triangular_type=$triangular_type) samples=5 evals=1
                creation_time_min = minimum(trial)

                # Print results in columnar format
                println(rpad(matrix_size, 15), rpad(num_levels, 10), rpad(String(triangular_type), 20), creation_time_min)
            end
        end
    end
    println("--------------------------------------------------------------------")
    println("\n--- Creation Benchmark Complete ---")
    println("Note: For precise benchmarks, increase `samples` and `evals` in `@btime`.")
    println("Also, consider running in a fresh Julia session to avoid JIT compilation effects on first run.")
end











function benchmark_triangular_block_matrix_creation_time(matrix_sizes::Vector{Int}, num_levels_array::Vector{Int})
    println("--- Benchmarking TriangularBlockMatrix Constructor Creation Time ---")
    println("--------------------------------------------------------------------")
    println(rpad("Matrix Size", 15), rpad("Levels", 10), rpad("Triangular Type", 20), rpad("Creation Time (min)", 25), rpad("Allocations", 15), "Memory (KiB)")
    println("--------------------------------------------------------------------")

    triangular_types = [:lower, :upper]

    for matrix_size in matrix_sizes
        for num_levels in num_levels_array
            for triangular_type in triangular_types
                # Create a random dense matrix and then make it triangular
                if triangular_type == :lower
                    dense_matrix_source = Matrix(LowerTriangular(rand(Float64, matrix_size, matrix_size)))
                else # :upper
                    dense_matrix_source = Matrix(UpperTriangular(rand(Float64, matrix_size, matrix_size)))
                end

                # U_approx is only used to pass the 'l' parameter to the constructor
                # Create a dummy U_approx as a vector of the correct length
                u_approx_dummy = zeros(num_levels) 

                # Corrected: trial now directly captures the BenchmarkTools.Trial object.
                # Increased samples and evals for more robust measurement.
                trial = @benchmark TriangularBlockMatrix($dense_matrix_source, $u_approx_dummy; triangular_type=$triangular_type) samples=10 evals=5
                
                # Check if trial.times is empty before calling minimum
                if isempty(trial.times)
                    creation_time_str = "No valid samples collected (too fast or compilation issues)"
                    allocs_str = "N/A"
                    memory_str = "N/A"
                else
                    creation_time_min = minimum(trial.times)
                    creation_time_str = string(creation_time_min)
                    
                    allocs_count = BenchmarkTools.allocs(trial)
                    allocs_str = string(allocs_count)

                    memory_bytes = BenchmarkTools.memory(trial)
                    memory_kib = round(memory_bytes / 1024, digits=2)
                    memory_str = string(memory_kib)
                end

                # Print results in columnar format
                println(rpad(matrix_size, 15), rpad(num_levels, 10), rpad(String(triangular_type), 20), rpad(creation_time_str, 25), rpad(allocs_str, 15), memory_str)
            end
        end
    end
    println("--------------------------------------------------------------------")
    println("\n--- TriangularBlockMatrix Constructor Benchmark Complete ---")
    println("Note: For precise benchmarks, increase `samples` and `evals` in `@btime`.")
    println("Also, consider running in a fresh Julia session to avoid JIT compilation effects on first run.")
end




function benchmark_new_triangular_block_matrix_creation_time(matrix_sizes::Vector{Int}, num_levels_array::Vector{Int})
    println("--- Benchmarking NEW TriangularBlockMatrix Constructor Creation Time ---")
    println("--------------------------------------------------------------------")
    println(rpad("Matrix Size", 15), rpad("Levels", 10), rpad("Triangular Type", 20), rpad("Creation Time (min)", 25), rpad("Allocations", 15), "Memory (KiB)")
    println("--------------------------------------------------------------------")

    triangular_types = [:lower, :upper]

    for matrix_size in matrix_sizes
        for num_levels in num_levels_array
            for triangular_type in triangular_types
                # Create a random dense matrix and then make it triangular
                if triangular_type == :lower
                    dense_matrix_source = Matrix(LowerTriangular(rand(Float64, matrix_size, matrix_size)))
                else # :upper
                    dense_matrix_source = Matrix(UpperTriangular(rand(Float64, matrix_size, matrix_size)))
                end

                # Benchmark the NEW TriangularBlockMatrix constructor directly
                trial = @benchmark TriangularBlockMatrix($dense_matrix_source, $num_levels; triangular_type=$triangular_type) samples=10 evals=5
                
                # Check if trial.times is empty before calling minimum
                if isempty(trial.times)
                    creation_time_str = "No valid samples collected (too fast or compilation issues)"
                    allocs_str = "N/A"
                    memory_str = "N/A"
                else
                    creation_time_min = minimum(trial.times)
                    creation_time_str = string(creation_time_min)
                    
                    allocs_count = BenchmarkTools.allocs(trial)
                    allocs_str = string(allocs_count)

                    memory_bytes = BenchmarkTools.memory(trial)
                    memory_kib = round(memory_bytes / 1024, digits=2)
                    memory_str = string(memory_kib)
                end

                # Print results in columnar format
                println(rpad(matrix_size, 15), rpad(num_levels, 10), rpad(String(triangular_type), 20), rpad(creation_time_str, 25), rpad(allocs_str, 15), memory_str)
            end
        end
    end
    println("--------------------------------------------------------------------")
    println("\n--- NEW TriangularBlockMatrix Constructor Benchmark Complete ---")
    println("Note: For precise benchmarks, increase `samples` and `evals` in `@btime`.")
    println("Also, consider running in a fresh Julia session to avoid JIT compilation effects on first run.")
end
