# Generate All Dataset Configurations
# This script creates a comprehensive set of datasets with various configurations

using MadNLP4NN
using Printf
using Dates

println("="^80)
println("MadNLP4NN.jl - Complete Dataset Generation")
println("="^80)
println("Start time: $(now())")
println()

# =============================================================================
# Configuration: Synthetic Datasets
# =============================================================================

# Sample sizes to explore
sample_sizes = [10000, 20000]

# Input dimensions (ambient dimensions for manifold datasets)
input_dims = [500]

# Output dimensions (number of classes)
output_dims = [10]

# For Gaussian Mixture: number of components per class
n_components_list = [10, 20, 30]

# For Nonlinear Manifold: manifold dimensions
manifold_dims = [20, 50, 100]

# For Nonlinear Manifold: nonlinearity types
nonlinearity_types = ["polynomial", "trigonometric", "mixed"]

# Seeds for reproducibility (multiple seeds for robustness testing)
seeds = [42]

# Output directory
output_dir = "./output"

# =============================================================================
# Generate Gaussian Mixture Datasets
# =============================================================================

println("\n" * "="^80)
println("PART 1: Gaussian Mixture Datasets")
println("="^80)

gm_count = 0
gm_datasets = []

for n_samples in sample_sizes
    for input_dim in input_dims
        for output_dim in output_dims
            for n_components in n_components_list
                for seed in seeds
                    global gm_count
                    gm_count += 1
                    
                    dataset_name = "GM_n$(n_samples)_d$(input_dim)_c$(output_dim)_comp$(n_components)_s$(seed)"
                    
                    println("\n[$gm_count] Creating: $dataset_name")
                    
                    try
                        # Datasets will be saved to output/datasets/gaussian_mixture/{dataset_name}/
                        metadata = create_dataset(
                            dataset_type="gaussian_mixture",
                            output_dir=output_dir,
                            n_samples=n_samples,
                            input_dim=input_dim,
                            output_dim=output_dim,
                            n_components=n_components,
                            seed=seed
                        )
                        
                        push!(gm_datasets, (
                            name=dataset_name,
                            type="gaussian_mixture",
                            n_samples=n_samples,
                            input_dim=input_dim,
                            output_dim=output_dim,
                            n_components=n_components,
                            seed=seed,
                            status="✓"
                        ))
                        
                        println("  ✓ Success")
                    catch e
                        println("  ✗ Failed: $e")
                        push!(gm_datasets, (
                            name=dataset_name,
                            type="gaussian_mixture",
                            n_samples=n_samples,
                            input_dim=input_dim,
                            output_dim=output_dim,
                            n_components=n_components,
                            seed=seed,
                            status="✗"
                        ))
                    end
                end
            end
        end
    end
end

println("\nGaussian Mixture Datasets: $gm_count generated")

# =============================================================================
# Generate Nonlinear Manifold Datasets
# =============================================================================

println("\n" * "="^80)
println("PART 2: Nonlinear Manifold Datasets")
println("="^80)

nm_count = 0
nm_datasets = []

for n_samples in sample_sizes
    for ambient_dim in input_dims
        for output_dim in output_dims
            for manifold_dim in manifold_dims
                # Skip if manifold_dim >= ambient_dim
                if manifold_dim >= ambient_dim
                    continue
                end
                
                for nonlinearity in nonlinearity_types
                    for seed in seeds
                        global nm_count
                        nm_count += 1
                        
                        dataset_name = "NM_n$(n_samples)_a$(ambient_dim)_m$(manifold_dim)_c$(output_dim)_$(nonlinearity)_s$(seed)"
                        
                        println("\n[$nm_count] Creating: $dataset_name")
                        
                        try
                            # Datasets will be saved to output/datasets/nonlinear_manifold/{dataset_name}/
                            metadata = create_dataset(
                                dataset_type="nonlinear_manifold",
                                output_dir=output_dir,
                                n_samples=n_samples,
                                input_dim=ambient_dim,
                                manifold_dim=manifold_dim,
                                output_dim=output_dim,
                                nonlinearity=nonlinearity,
                                seed=seed
                            )
                            
                            push!(nm_datasets, (
                                name=dataset_name,
                                type="nonlinear_manifold",
                                n_samples=n_samples,
                                ambient_dim=ambient_dim,
                                manifold_dim=manifold_dim,
                                output_dim=output_dim,
                                nonlinearity=nonlinearity,
                                seed=seed,
                                status="✓"
                            ))
                            
                            println("  ✓ Success")
                        catch e
                            println("  ✗ Failed: $e")
                            push!(nm_datasets, (
                                name=dataset_name,
                                type="nonlinear_manifold",
                                n_samples=n_samples,
                                ambient_dim=ambient_dim,
                                manifold_dim=manifold_dim,
                                output_dim=output_dim,
                                nonlinearity=nonlinearity,
                                seed=seed,
                                status="✗"
                            ))
                        end
                    end
                end
            end
        end
    end
end

println("\nNonlinear Manifold Datasets: $nm_count generated")

# =============================================================================
# Generate Real Datasets
# =============================================================================

println("\n" * "="^80)
println("PART 3: Real-World Datasets")
println("="^80)

real_datasets = ["mnist", "fashionmnist", "cifar10"]
real_count = 0
real_dataset_info = []

for dataset_name in real_datasets
    global real_count
    real_count += 1
    
    println("\n[$real_count] Creating: $(uppercase(dataset_name))")
    
    try
        metadata = create_dataset(
            dataset_type=dataset_name,
            output_dir=output_dir
        )
        
        push!(real_dataset_info, (
            name=dataset_name,
            type="real",
            input_dim=metadata[:input_dim],
            output_dim=metadata[:output_dim],
            status="✓"
        ))
        
        println("  ✓ Success")
        println("  Input dim: $(metadata[:input_dim])")
        println("  Output dim: $(metadata[:output_dim])")
    catch e
        println("  ✗ Failed: $e")
        push!(real_dataset_info, (
            name=dataset_name,
            type="real",
            status="✗"
        ))
    end
end

println("\nReal Datasets: $real_count generated")

# =============================================================================
# Summary and Statistics
# =============================================================================

println("\n" * "="^80)
println("SUMMARY")
println("="^80)

total_datasets = gm_count + nm_count + real_count
successful_gm = count(d -> d.status == "✓", gm_datasets)
successful_nm = count(d -> d.status == "✓", nm_datasets)
successful_real = count(d -> d.status == "✓", real_dataset_info)
total_successful = successful_gm + successful_nm + successful_real

println("\nDataset Generation Complete!")
println("  End time: $(now())")
println()
println("Total Datasets Generated: $total_datasets")
println("  - Gaussian Mixture: $gm_count ($successful_gm successful)")
println("  - Nonlinear Manifold: $nm_count ($successful_nm successful)")
println("  - Real Datasets: $real_count ($successful_real successful)")
println()
println("Success Rate: $(@sprintf("%.1f%%", 100 * total_successful / total_datasets))")

# =============================================================================
# Detailed Configuration Breakdown
# =============================================================================

println("\n" * "="^80)
println("CONFIGURATION DETAILS")
println("="^80)

println("\n1. Gaussian Mixture Datasets ($gm_count total)")
println("-"^80)
println("  Sample sizes: $(join(sample_sizes, ", "))")
println("  Input dimensions: $(join(input_dims, ", "))")
println("  Output dimensions (classes): $(join(output_dims, ", "))")
println("  Components per class: $(join(n_components_list, ", "))")
println("  Seeds: $(join(seeds, ", "))")
println("  Formula: $(length(sample_sizes)) × $(length(input_dims)) × $(length(output_dims)) × $(length(n_components_list)) × $(length(seeds)) = $gm_count")

println("\n2. Nonlinear Manifold Datasets ($nm_count total)")
println("-"^80)
println("  Sample sizes: $(join(sample_sizes, ", "))")
println("  Ambient dimensions: $(join(input_dims, ", "))")
println("  Manifold dimensions: $(join(manifold_dims, ", "))")
println("  Output dimensions (classes): $(join(output_dims, ", "))")
println("  Nonlinearity types: $(join(nonlinearity_types, ", "))")
println("  Seeds: $(join(seeds, ", "))")
println("  Note: Only manifold_dim < ambient_dim combinations are generated")

# Count valid manifold combinations
valid_manifold_combos = 0
for ambient_dim in input_dims
    for manifold_dim in manifold_dims
        if manifold_dim < ambient_dim
            global valid_manifold_combos
            valid_manifold_combos += 1
        end
    end
end
println("  Valid (ambient, manifold) combinations: $valid_manifold_combos")
println("  Formula: $(length(sample_sizes)) × $valid_manifold_combos × $(length(output_dims)) × $(length(nonlinearity_types)) × $(length(seeds)) = $nm_count")

println("\n3. Real Datasets ($real_count total)")
println("-"^80)
for dataset in real_dataset_info
    if dataset.status == "✓"
        println("  - $(uppercase(dataset.name)): input_dim=$(dataset.input_dim), output_dim=$(dataset.output_dim)")
    else
        println("  - $(uppercase(dataset.name)): Failed")
    end
end

# =============================================================================
# Save Manifest
# =============================================================================

println("\n" * "="^80)
println("SAVING MANIFEST")
println("="^80)

using JSON3

manifest = Dict(
    "generation_time" => string(now()),
    "total_datasets" => total_datasets,
    "successful_datasets" => total_successful,
    "configuration" => Dict(
        "sample_sizes" => sample_sizes,
        "input_dims" => input_dims,
        "output_dims" => output_dims,
        "n_components_list" => n_components_list,
        "manifold_dims" => manifold_dims,
        "nonlinearity_types" => nonlinearity_types,
        "seeds" => seeds
    ),
    "gaussian_mixture" => [
        Dict(
            "name" => d.name,
            "n_samples" => d.n_samples,
            "input_dim" => d.input_dim,
            "output_dim" => d.output_dim,
            "n_components" => d.n_components,
            "seed" => d.seed,
            "status" => d.status
        )
        for d in gm_datasets
    ],
    "nonlinear_manifold" => [
        Dict(
            "name" => d.name,
            "n_samples" => d.n_samples,
            "ambient_dim" => d.ambient_dim,
            "manifold_dim" => d.manifold_dim,
            "output_dim" => d.output_dim,
            "nonlinearity" => d.nonlinearity,
            "seed" => d.seed,
            "status" => d.status
        )
        for d in nm_datasets
    ],
    "real_datasets" => [
        Dict(
            "name" => d.name,
            "type" => d.type,
            "input_dim" => get(d, :input_dim, nothing),
            "output_dim" => get(d, :output_dim, nothing),
            "status" => d.status
        )
        for d in real_dataset_info
    ]
)

manifest_path = joinpath(output_dir, "dataset_manifest.json")
open(manifest_path, "w") do f
    JSON3.pretty(f, manifest)
end

println("✓ Manifest saved to: $manifest_path")

# =============================================================================
# Storage Information
# =============================================================================

println("\n" * "="^80)
println("STORAGE INFORMATION")
println("="^80)

println("\nAll datasets saved to: $output_dir/datasets/")
println("\nDirectory structure:")
println("  output/")
println("    ├── datasets/")
println("    │   ├── cifar10/")
println("    │   │   ├── metadata.json")
println("    │   │   └── ... (CIFAR10 data files)")
println("    │   ├── fashionmnist/")
println("    │   │   ├── metadata.json")
println("    │   │   └── ... (FashionMNIST data files)")
println("    │   ├── mnist/")
println("    │   │   ├── metadata.json")
println("    │   │   └── ... (MNIST data files)")
println("    │   ├── gaussian_mixture/")
println("    │   │   ├── GM_n10000_d500_c10_comp10_s42/")
println("    │   │   │   ├── data.pt")
println("    │   │   │   ├── labels.pt")
println("    │   │   │   └── metadata.json")
println("    │   │   ├── GM_n10000_d500_c10_comp20_s42/")
println("    │   │   └── ... (more GM datasets)")
println("    │   └── nonlinear_manifold/")
println("    │       ├── NM_n10000_a500_m20_c10_polynomial_s42/")
println("    │       │   ├── data.pt")
println("    │       │   ├── labels.pt")
println("    │       │   └── metadata.json")
println("    │       ├── NM_n10000_a500_m20_c10_trigonometric_s42/")
println("    │       └── ... (more NM datasets)")
println("    └── dataset_manifest.json")

# =============================================================================
# Next Steps
# =============================================================================

println("\n" * "="^80)
println("NEXT STEPS")
println("="^80)

println("""
Now that you have generated all datasets, you can:

1. Train models on these datasets:
   julia examples/train_all_parallel.jl

2. Train specific combinations:
   julia> using MadNLP4NN
   julia> train_model(
       dataset_type="gaussian_mixture",
       model_config="medium_mlp",
       n_samples=10000,
       input_dim=784,
       output_dim=10,
       n_components=20,
       seed=42
   )

3. Inspect the manifest:
   julia> using JSON3
   julia> manifest = JSON3.read(read("output/dataset_manifest.json", String))

4. Use datasets for NLP optimization (once implemented)

For more information, see:
  - docs/TUTORIAL.md
  - README.md
""")

println("="^80)
println("Dataset generation complete!")
println("="^80)
