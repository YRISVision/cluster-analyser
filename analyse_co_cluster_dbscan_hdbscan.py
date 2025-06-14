import numpy as np
from pymatgen.core import Structure
from scipy.spatial import ConvexHull
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import hdbscan
import plotly.graph_objects as go
from itertools import permutations, product
import argparse
import os
import logging

# Set up logging to debug issues
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def parse_arguments():
    """
    Parse command-line arguments to configure the clustering and facet analysis.
    
    Why: Allows users to customize clustering method, distance thresholds, and facet assignment criteria
         without modifying the code, enhancing flexibility for different cluster sizes or substrates.
    
    Parameters:
    - vasp_file: Path to the VASP file (e.g., 'relaxed_structure_bfgs.vasp').
    - clustering-method: 'dbscan', 'hdbscan', or 'both' (default: 'both') to compare methods.
    - eps: DBSCAN radius (default: 2.7 Å), tune 2.5–3.2 Å based on Co-Co distances (~2.5–3.2 Å).
    - min-samples: Minimum points for core points (default: 4), reduce to 3 for looser clustering.
    - min-cluster-size: HDBSCAN minimum cluster size (default: 8), increase to 10 for stricter clustering.
    - cosine-threshold: Facet assignment similarity (default: 0.90, ~26°), lower to 0.85 for distorted facets.
    
    Returns: Parsed arguments as a namespace object.
    """
    parser = argparse.ArgumentParser(description="Analyze Co cluster facets from VASP file using DBSCAN or HDBSCAN.")
    parser.add_argument("vasp_file", help="Path to VASP file")
    parser.add_argument("--clustering-method", choices=["dbscan", "hdbscan", "both"], default="both", help="Clustering method: dbscan, hdbscan, or both")
    parser.add_argument("--eps", type=float, default=2.7, help="DBSCAN eps distance (Å)")
    parser.add_argument("--min-samples", type=int, default=4, help="Minimum samples for DBSCAN or HDBSCAN")
    parser.add_argument("--min-cluster-size", type=int, default=8, help="HDBSCAN minimum cluster size")
    parser.add_argument("--cosine-threshold", type=float, default=0.90, help="Cosine similarity threshold for facet assignment")
    return parser.parse_args()

def compute_pbc_distance_matrix(coords, lattice):
    """
    Compute a distance matrix between all pairs of coordinates, accounting for periodic boundary conditions (PBCs).
    
    Why: Ensures accurate distances for atoms in a periodic supercell (17.754 Å, 25.988 Å, 60.000 Å), critical for
         small clusters that may span cell boundaries. The minimum image convention minimizes distances across periodic images.
    
    Parameters:
    - coords: Array of Cartesian coordinates (n_atoms, 3).
    - lattice: Pymatgen Lattice object defining the supercell.
    
    Implementation: Checks neighboring cells (±2 in x, y, z) to find the shortest distance between each atom pair.
    
    Tuning: If distances remain large (>4 Å), verify VASP file coordinates or increase offset range (e.g., ±3).
    
    Returns: n_atoms x n_atoms distance matrix (Å).
    """
    n_atoms = len(coords)
    dist_matrix = np.zeros((n_atoms, n_atoms))
    cell = lattice.matrix
    inv_cell = np.linalg.inv(cell)
    offsets = np.array([[i, j, k] for i in [-2, -1, 0, 1, 2] for j in [-2, -1, 0, 1, 2] for k in [-2, -1, 0, 1, 2]])
    distances = []
    for i in range(n_atoms):
        for j in range(i, n_atoms):
            diff = coords[i] - coords[j]
            min_dist = float('inf')
            for offset in offsets:
                frac_diff = np.dot(diff + np.dot(offset, cell), inv_cell)
                frac_diff -= np.round(frac_diff)
                cart_diff = np.dot(frac_diff, cell)
                dist = np.linalg.norm(cart_diff)
                min_dist = min(min_dist, dist)
            dist_matrix[i, j] = min_dist
            dist_matrix[j, i] = min_dist
            if min_dist > 0:
                distances.append(min_dist)
    if distances:
        q1 = np.percentile(distances, 25)
        median = np.median(distances)
        q3 = np.percentile(distances, 75)
        logger.info(f"Distance distribution: Q1={q1:.2f} Å, Median={median:.2f} Å, Q3={q3:.2f} Å, Min={np.min(distances):.2f} Å, Max={np.max(distances):.2f} Å")
        if median > 3.5:
            logger.warning("Median distance is high (>3.5 Å). Expected ~2.5–3.2 Å for FCC Co. Check VASP file or increase PBC offset range.")
    return dist_matrix

def filter_outliers(coords, lattice, method, eps, min_samples, min_cluster_size):
    """
    Identify and filter outlier atoms using DBSCAN or HDBSCAN clustering, followed by a nearest neighbor check.
    
    Why: Removes detached or low-coordination atoms (e.g., simulation artifacts) to focus on the main cluster for
         facet analysis. For a 38-atom Co cluster, expect 3–8 outliers, retaining ~30–35 atoms.
    
    Parameters:
    - coords: Array of Co atom coordinates (n_atoms, 3).
    - lattice: Pymatgen Lattice object for PBC calculations.
    - method: 'dbscan' or 'hdbscan' to select clustering algorithm.
    - eps: DBSCAN radius (Å), set ~1.2× expected Co-Co distance (2.5–3.2 Å).
    - min_samples: Minimum points for core points, adjust 3–5 for small clusters.
    - min_cluster_size: HDBSCAN minimum cluster size, set 6–10 to avoid fragmentation.
    
    Implementation:
    - Computes PBC-aware distance matrix.
    - Applies DBSCAN (fixed radius) or HDBSCAN (variable density) to cluster atoms.
    - Selects the largest cluster and applies a neighbor check (4.0 Å cutoff).
    - Uses a fallback (≥1 neighbor) if <70% atoms are retained to prevent over-filtering.
    
    Tuning:
    - DBSCAN: Increase `eps` to 3.0 Å if too many outliers; decrease to 2.3 Å if too few.
    - HDBSCAN: Increase `min_cluster_size` to 10 if cluster splits; decrease `min_samples` to 3 for leniency.
    - If median distance >3.5 Å, check VASP file or PBC logic.
    
    Returns: Boolean mask for kept atoms, cluster labels.
    """
    logger.info(f"[{method.upper()}] Starting outlier filtering with {len(coords)} atoms")
    dist_matrix = compute_pbc_distance_matrix(coords, lattice)
    
    if method == "dbscan":
        clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
    else:  # hdbscan
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric="precomputed")
    
    try:
        labels = clusterer.fit_predict(dist_matrix)
    except Exception as e:
        logger.error(f"[{method.upper()}] Clustering failed: {e}")
        raise
    
    cluster_sizes = np.bincount(labels[labels >= 0])
    largest_label = np.argmax(cluster_sizes) if len(cluster_sizes) > 0 else -1
    cluster_mask = (labels == largest_label)
    
    neighbor_cutoff = 4.0
    neighbors = (dist_matrix < neighbor_cutoff).sum(axis=1) - 1
    nn_mask = neighbors >= min_samples
    
    keep_mask = cluster_mask & nn_mask
    
    if np.sum(keep_mask) < len(coords) * 0.7:
        logger.warning(f"[{method.upper()}] Too few atoms kept ({np.sum(keep_mask)}); relaxing neighbor check")
        nn_mask = neighbors >= 1
        keep_mask = cluster_mask & nn_mask
    
    logger.info(f"[{method.upper()}] Filtered {np.sum(keep_mask)} atoms, {len(coords) - np.sum(keep_mask)} outliers")
    return keep_mask, labels

def get_normal_and_area(points):
    """
    Calculate the normal vector and area of a triangular face from the convex hull.
    
    Why: Facets are defined by triangular faces in the convex hull, and their normals determine
         FCC facet assignments. Area contributes to percentage calculations.
    
    Parameters:
    - points: Array of 3 vertices (3, 3) defining the triangle.
    
    Implementation: Uses cross product to compute the normal and area (half the cross product magnitude).
    
    Tuning: No parameters to adjust, but ensure filtered_coords are accurate to avoid degenerate triangles.
    
    Returns: Normalized normal vector, area (Å²).
    """
    v1 = points[1] - points[0]
    v2 = points[2] - points[0]
    normal = np.cross(v1, v2)
    area = 0.5 * np.linalg.norm(normal)
    normal = normal / np.linalg.norm(normal) if np.linalg.norm(normal) > 0 else normal
    return normal, area

def generate_equivalent_normals(hkl, normalize=True):
    """
    Generate all symmetry-equivalent normal vectors for a given FCC Miller index.
    
    Why: FCC facets (e.g., (111)) have multiple equivalent directions due to cubic symmetry.
         This ensures all orientations are considered during facet assignment.
    
    Parameters:
    - hkl: List of Miller indices [h, k, l] (e.g., [1, 1, 1] for (111)).
    - normalize: Normalize vectors to unit length (default: True).
    
    Implementation: Generates permutations and sign combinations of hkl, excluding zero vectors.
    
    Tuning: No parameters to adjust, but ensure hkl inputs match fcc_facets definitions.
    
    Returns: List of equivalent normal vectors.
    """
    h, k, l = hkl
    perms = set(permutations([h, k, l]))
    normals = []
    for p in perms:
        for signs in product([1, -1], repeat=3):
            vec = np.array([p[i] * signs[i] for i in range(3)])
            if np.all(vec == 0):
                continue
            if normalize:
                vec = vec / np.linalg.norm(vec)
            normals.append(vec)
    return normals

def analyze_facets(filtered_coords, method, cosine_threshold):
    """
    Compute the convex hull and assign triangular faces to FCC facets based on normal vectors.
    
    Why: Identifies exposed facets ((111), (100), etc.) of the Co cluster, accounting for
         substrate-induced distortions (e.g., TiO₂ effects). Critical for catalytic or material studies.
    
    Parameters:
    - filtered_coords: Coordinates of filtered Co atoms (n_filtered, 3).
    - method: 'dbscan' or 'hdbscan' for output naming.
    - cosine_threshold: Similarity threshold for facet assignment (0.90, ~26°), lower to 0.85 for distortions.
    
    Implementation:
    - Computes convex hull to define surface faces.
    - Assigns each face to an FCC facet by comparing its normal to ideal normals.
    - Calculates area percentages for each facet type.
    
    Tuning:
    - Lower `cosine_threshold` to 0.85 (~32°) if many facets are “unassigned” due to substrate distortions.
    - If facets misclassify (e.g., (111) as (211)), inspect normals in facet_data_{method}.txt.
    
    Returns: Facet data, assignments, areas, percentages, total surface area, hull object.
    """
    logger.info(f"[{method.upper()}] Computing convex hull with {len(filtered_coords)} atoms")
    try:
        hull = ConvexHull(filtered_coords)
    except Exception as e:
        logger.error(f"[{method.upper()}] Convex hull computation failed: {e}")
        raise
    
    facets = hull.simplices
    total_surface_area = hull.area

    fcc_facets = {
        "(111)": np.array([1, 1, 1]) / np.sqrt(3),
        "(100)": np.array([1, 0, 0]),
        "(110)": np.array([1, 1, 0]) / np.sqrt(2),
        "(211)": np.array([2, 1, 1]) / np.sqrt(6),
        "(311)": np.array([3, 1, 1]) / np.sqrt(11),
        "(221)": np.array([2, 2, 1]) / np.sqrt(9)
    }
    equivalent_normals = {
        "(111)": generate_equivalent_normals([1, 1, 1]),
        "(100)": generate_equivalent_normals([1, 0, 0]),
        "(110)": generate_equivalent_normals([1, 1, 0]),
        "(211)": generate_equivalent_normals([2, 1, 1]),
        "(311)": generate_equivalent_normals([3, 1, 1]),
        "(221)": generate_equivalent_normals([2, 2, 1]),
        "unassigned": []
    }

    facet_data = []
    for simplex in facets:
        points = filtered_coords[simplex]
        try:
            normal, area = get_normal_and_area(points)
            facet_data.append({"simplex": simplex, "normal": normal, "area": area})
        except Exception as e:
            logger.error(f"[{method.upper()}] Failed to compute normal/area for simplex {simplex}: {e}")
            continue

    facet_assignments = []
    facet_areas = {name: 0.0 for name in fcc_facets}
    facet_areas["unassigned"] = 0.0
    for data in facet_data:
        normal = data["normal"]
        area = data["area"]
        assigned = False
        for facet_name, normals in equivalent_normals.items():
            if not normals:
                continue
            similarities = cosine_similarity([normal], normals)[0]
            if np.max(similarities) > cosine_threshold:
                facet_assignments.append(facet_name)
                facet_areas[facet_name] += area
                assigned = True
                break
        if not assigned:
            facet_assignments.append("unassigned")
            facet_areas["unassigned"] += area

    percentages = {k: (v / total_surface_area * 100) for k, v in facet_areas.items()}
    logger.info(f"[{method.upper()}] Facet analysis completed with {len(facet_data)} facets")
    return facet_data, facet_assignments, facet_areas, percentages, total_surface_area, hull

def generate_visualizations(co_coords, filtered_coords, filtered_indices, outlier_indices, labels, facet_data, facet_assignments, facet_areas, percentages, hull, method, cosine_threshold):
    """
    Generate Plotly visualizations for atom positions, clustering results, and facet assignments.
    
    Why: Visualizes the cluster’s structure, outliers, and facets to validate clustering and facet analysis.
         Critical for inspecting substrate effects (e.g., distorted (111) facets) and comparing DBSCAN/HDBSCAN.
    
    Parameters:
    - co_coords: All Co atom coordinates (n_atoms, 3).
    - filtered_coords: Filtered Co atom coordinates (n_filtered, 3).
    - filtered_indices: Indices of filtered atoms.
    - outlier_indices: Indices of outlier atoms.
    - labels: Cluster labels from DBSCAN/HDBSCAN (n_atoms,).
    - facet_data: List of facet dictionaries (simplex, normal, area).
    - facet_assignments: List of facet labels ((111), unassigned, etc.).
    - facet_areas: Dictionary of total areas per facet type.
    - percentages: Dictionary of percentage contributions per facet type.
    - hull: ConvexHull object for surface geometry.
    - method: 'dbscan' or 'hdbscan' for output naming.
    - cosine_threshold: Used for documentation in output files.
    
    Implementation:
    - Creates atom plot (`co_atoms_{method}.html`) with filtered (blue) and outlier (red) atoms.
    - Creates facet plot (`co_facets_{method}.html`) with colored triangular faces and dropdown filters.
    - Creates clustering plot (`co_clusters_{method}.html`) showing cluster labels.
    - Saves facet data and percentages to `facet_data_{method}.txt`.
    
    Tuning:
    - Adjust `neighbor_cutoff` in filter_outliers if outliers appear incorrect in atom plot.
    - Use facet plot dropdown to inspect (111) facets for substrate distortions.
    
    Returns: Facet areas, percentages, filtered indices, outlier indices (for comparison summary).
    """
    logger.info(f"[{method.upper()}] Starting visualization generation")
    colors = {
        "(111)": "green",
        "(100)": "red",
        "(110)": "purple",
        "(211)": "blue",
        "(311)": "orange",
        "(221)": "cyan",
        "unassigned": "gray"
    }

    # Ensure output directory is current working directory
    output_dir = os.getcwd()
    logger.info(f"[{method.upper()}] Writing output files to {output_dir}")

    # Atom plot
    try:
        fig1 = go.Figure()
        if len(filtered_coords) > 0:
            fig1.add_trace(go.Scatter3d(
                x=filtered_coords[:, 0], y=filtered_coords[:, 1], z=filtered_coords[:, 2],
                mode="markers+text",
                marker=dict(size=5, color="blue"),
                text=[str(i) for i in filtered_indices],
                textposition="top center",
                name="Filtered Co atoms"
            ))
        if len(outlier_indices) > 0:
            outlier_coords = co_coords[outlier_indices]
            fig1.add_trace(go.Scatter3d(
                x=outlier_coords[:, 0], y=outlier_coords[:, 1], z=outlier_coords[:, 2],
                mode="markers+text",
                marker=dict(size=5, color="red", symbol="x"),
                text=[str(i) for i in outlier_indices],
                textposition="top center",
                name="Outlier Co atoms"
            ))
        fig1.update_layout(
            title=f"Co Cluster: Filtered Atoms (Blue) and Outliers (Red) with Indices ({method.upper()})",
            scene=dict(xaxis_title="X (Å)", yaxis_title="Y (Å)", zaxis_title="Z (Å)"),
            showlegend=True,
            updatemenus=[
                dict(
                    buttons=[
                        dict(
                            args=[{"visible": [True, True if len(outlier_indices) > 0 else False]}],
                            label="Show All Atoms",
                            method="update"
                        ),
                        dict(
                            args=[{"visible": [True, False]}],
                            label="Show Filtered Atoms",
                            method="update"
                        ),
                        dict(
                            args=[{"visible": [False, True if len(outlier_indices) > 0 else False]}],
                            label="Show Outlier Atoms",
                            method="update"
                        )
                    ],
                    direction="down",
                    showactive=True,
                    x=0.1,
                    xanchor="left",
                    y=1.1,
                    yanchor="top"
                )
            ]
        )
        atom_plot_path = os.path.join(output_dir, f"co_atoms_{method}.html")
        fig1.write_html(atom_plot_path)
        logger.info(f"[{method.upper()}] Atom plot saved as '{atom_plot_path}'")
    except Exception as e:
        logger.error(f"[{method.upper()}] Failed to generate atom plot: {e}")
        raise

    # Facet plot
    try:
        fig2 = go.Figure()
        facet_groups = {key: {"vertices": [], "indices": [], "centroids": [], "labels": []} for key in facet_areas.keys()}
        vertex_map = {}
        current_vertex = 0
        all_vertices = []

        for i, (data, assignment) in enumerate(zip(facet_data, facet_assignments)):
            simplex = data["simplex"]
            points = filtered_coords[simplex]
            for v_idx in simplex:
                v = tuple(filtered_coords[v_idx])
                if v not in vertex_map:
                    vertex_map[v] = current_vertex
                    all_vertices.append(filtered_coords[v_idx])
                    current_vertex += 1
            facet_groups[assignment]["indices"].extend([vertex_map[tuple(filtered_coords[v_idx])] for v_idx in simplex])
            centroid = np.mean(points, axis=0)
            facet_groups[assignment]["centroids"].append(centroid)
            facet_groups[assignment]["labels"].append(assignment)

        all_vertices = np.array(all_vertices)

        traces = []
        trace_indices = {}
        atom_trace_index = 0

        atom_trace = go.Scatter3d(
            x=filtered_coords[:, 0], y=filtered_coords[:, 1], z=filtered_coords[:, 2],
            mode="markers",
            marker=dict(size=3, color="blue"),
            name="Filtered Co atoms",
            visible=True
        )
        traces.append(atom_trace)

        for facet_name in facet_groups:
            if len(facet_groups[facet_name]["indices"]) == 0:
                continue
            mesh_trace = go.Mesh3d(
                x=all_vertices[:, 0],
                y=all_vertices[:, 1],
                z=all_vertices[:, 2],
                i=facet_groups[facet_name]["indices"][0::3],
                j=facet_groups[facet_name]["indices"][1::3],
                k=facet_groups[facet_name]["indices"][2::3],
                color=colors[facet_name],
                opacity=0.5,
                name=facet_name,
                showlegend=True
            )
            traces.append(mesh_trace)
            trace_indices[facet_name] = [len(traces) - 1]
            centroids = np.array(facet_groups[facet_name]["centroids"])
            if len(centroids) > 0:
                label_trace = go.Scatter3d(
                    x=centroids[:, 0],
                    y=centroids[:, 1],
                    z=centroids[:, 2],
                    mode="text",
                    text=facet_groups[facet_name]["labels"],
                    textposition="middle center",
                    showlegend=False
                )
                traces.append(label_trace)
                trace_indices[facet_name].append(len(traces) - 1)

        for trace in traces:
            fig2.add_trace(trace)

        buttons = []
        facet_names = [name for name in facet_areas.keys() if len(facet_groups[name]["indices"]) > 0]
        for facet_name in facet_names:
            visible = [False] * len(traces)
            visible[atom_trace_index] = True
            for idx in trace_indices[facet_name]:
                visible[idx] = True
            buttons.append(
                dict(
                    args=[{"visible": visible}],
                    label=f"Show {facet_name}",
                    method="update"
                )
            )
        visible = [True] * len(traces)
        buttons.append(
            dict(
                args=[{"visible": visible}],
                label="Show All Facets",
                method="update"
            )
        )

        fig2.update_layout(
            title=f"Co Cluster: Facets Colored by Miller Index ({method.upper()})",
            scene=dict(xaxis_title="X (Å)", yaxis_title="Y (Å)", zaxis_title="Z (Å)"),
            showlegend=True,
            legend=dict(x=0.8, y=0.9),
            updatemenus=[
                dict(
                    buttons=buttons,
                    direction="down",
                    showactive=True,
                    x=0.1,
                    xanchor="left",
                    y=1.1,
                    yanchor="top"
                )
            ]
        )
        facet_plot_path = os.path.join(output_dir, f"co_facets_{method}.html")
        fig2.write_html(facet_plot_path)
        logger.info(f"[{method.upper()}] Facet plot saved as '{facet_plot_path}'")
    except Exception as e:
        logger.error(f"[{method.upper()}] Failed to generate facet plot: {e}")
        raise

    # Clustering plot
    try:
        fig_cluster = go.Figure()
        if len(co_coords) > 0 and len(labels) == len(co_coords):
            fig_cluster.add_trace(go.Scatter3d(
                x=co_coords[:, 0], y=co_coords[:, 1], z=co_coords[:, 2],
                mode="markers+text",
                marker=dict(size=5, color=labels, colorscale="Viridis"),
                text=[str(i) for i in range(len(co_coords))],
                textposition="top center",
                name="Clustered Atoms"
            ))
            fig_cluster.update_layout(
                title=f"{method.upper()} Clustering Results",
                scene=dict(xaxis_title="X (Å)", yaxis_title="Y (Å)", zaxis_title="Z (Å)"),
                showlegend=True
            )
            cluster_plot_path = os.path.join(output_dir, f"co_clusters_{method}.html")
            fig_cluster.write_html(cluster_plot_path)
            logger.info(f"[{method.upper()}] Clustering plot saved as '{cluster_plot_path}'")
        else:
            logger.error(f"[{method.upper()}] Invalid clustering data: co_coords={len(co_coords)}, labels={len(labels)}")
            raise ValueError("Invalid clustering data")
    except Exception as e:
        logger.error(f"[{method.upper()}] Failed to generate clustering plot: {e}")
        raise

    # Save facet data
    try:
        facet_data_path = os.path.join(output_dir, f"facet_data_{method}.txt")
        with open(facet_data_path, "w") as f:
            f.write(f"[{method.upper()}] Outlier Information:\n")
            f.write(f"Filtered atoms: {len(filtered_coords)}\n")
            f.write(f"Outlier atoms: {len(outlier_indices)} (indices: {outlier_indices.tolist()})\n\n")
            f.write(f"[{method.upper()}] Facet Assignments and Areas:\n")
            for i, (data, assignment) in enumerate(zip(facet_data, facet_assignments)):
                f.write(f"Facet {i}: {assignment}, Area: {data['area']:.2f} Å², Normal: {data['normal']}\n")
            f.write(f"\n[{method.upper()}] Percentages:\n")
            for facet, percentage in percentages.items():
                f.write(f"{facet}: {percentage:.2f}%\n")
            f.write(f"\nNote: The facet plot (co_facets_{method}.html) includes a dropdown menu to filter facets by Miller index.")
        logger.info(f"[{method.upper()}] Facet data saved as '{facet_data_path}'")
    except Exception as e:
        logger.error(f"[{method.upper()}] Failed to save facet data: {e}")
        raise

    logger.info(f"[{method.upper()}] Visualization generation completed")
    return facet_areas, percentages, filtered_indices, outlier_indices

def main():
    """
    Main function to orchestrate the Co cluster analysis using DBSCAN and/or HDBSCAN.
    
    Why: Coordinates loading the VASP file, running clustering, facet analysis, visualizations,
         and generating a comparison summary for DBSCAN and HDBSCAN results.
    
    Implementation:
    - Loads Co atom coordinates from the VASP file.
    - Runs DBSCAN and/or HDBSCAN based on the clustering method argument.
    - Processes each method’s results (outliers, facets, visualizations).
    - Compares results in a summary file if both methods are run.
    
    Tuning:
    - Use `--clustering-method dbscan` or `hdbscan` to run one method for quick tests.
    - Adjust parameters via command-line arguments to optimize for your cluster.
    - Check `co_clusters_{method}.html` to validate clustering and `co_facets_{method}.html` for facet distortions.
    
    Returns: None, outputs files and console logs.
    """
    args = parse_arguments()

    logger.info("Starting Co cluster analysis")
    try:
        structure = Structure.from_file(args.vasp_file)
    except Exception as e:
        logger.error(f"Failed to load VASP file: {e}")
        raise ValueError(f"Failed to load VASP file: {e}")
    
    co_atoms = [site for site in structure if site.species_string == "Co"]
    if not co_atoms:
        logger.error("No Co atoms found in the VASP file")
        raise ValueError("No Co atoms found in the VASP file")
    
    co_coords = np.array([site.coords for site in co_atoms])
    logger.info(f"Loaded VASP file with {len(co_coords)} Co atoms")

    methods = ["dbscan", "hdbscan"] if args.clustering_method == "both" else [args.clustering_method]
    results = {}

    for method in methods:
        logger.info(f"[{method.upper()}] Processing...")
        try:
            keep_mask, labels = filter_outliers(co_coords, structure.lattice, method, args.eps, args.min_samples, args.min_cluster_size)
            filtered_coords = co_coords[keep_mask]
            filtered_indices = np.where(keep_mask)[0]
            outlier_indices = np.where(~keep_mask)[0]
            logger.info(f"[{method.upper()}] Filtered Co atoms: {len(filtered_coords)}")
            logger.info(f"[{method.upper()}] Outlier atoms: {len(outlier_indices)} (indices: {outlier_indices.tolist()})")

            if len(filtered_coords) < 4:
                logger.error(f"[{method.upper()}] Too few atoms after filtering to compute convex hull")
                raise ValueError(f"[{method.upper()}] Too few atoms after filtering to compute convex hull")

            facet_data, facet_assignments, facet_areas, percentages, total_surface_area, hull = analyze_facets(filtered_coords, method, args.cosine_threshold)
            facet_areas, percentages, filtered_indices, outlier_indices = generate_visualizations(
                co_coords, filtered_coords, filtered_indices, outlier_indices, labels, facet_data, facet_assignments, facet_areas, percentages, hull, method, args.cosine_threshold
            )
            results[method] = {
                "facet_areas": facet_areas,
                "percentages": percentages,
                "filtered_count": len(filtered_coords),
                "outlier_indices": outlier_indices.tolist()
            }
            logger.info(f"[{method.upper()}] Analysis completed successfully")
        except Exception as e:
            logger.error(f"[{method.upper()}] Analysis failed: {e}")
            continue  # Continue with next method instead of raising

    if len(methods) > 1 and len(results) == 2:
        try:
            output_dir = os.getcwd()
            comparison_path = os.path.join(output_dir, "comparison_summary.txt")
            with open(comparison_path, "w") as f:
                f.write("Comparison of DBSCAN and HDBSCAN Results\n")
                f.write("=" * 40 + "\n\n")
                f.write("Outlier Information:\n")
                f.write(f"DBSCAN: {results['dbscan']['filtered_count']} filtered atoms, {len(results['dbscan']['outlier_indices'])} outliers (indices: {results['dbscan']['outlier_indices']})\n")
                f.write(f"HDBSCAN: {results['hdbscan']['filtered_count']} filtered atoms, {len(results['hdbscan']['outlier_indices'])} outliers (indices: {results['hdbscan']['outlier_indices']})\n\n")
                f.write("Facet Percentages:\n")
                f.write("{:<15} {:<15} {:<15}\n".format("Facet", "DBSCAN (%)", "HDBSCAN (%)"))
                for facet in results['dbscan']['percentages']:
                    dbscan_pct = results['dbscan']['percentages'][facet]
                    hdbscan_pct = results['hdbscan']['percentages'][facet]
                    f.write("{:<15} {:<15.2f} {:<15.2f}\n".format(facet, dbscan_pct, hdbscan_pct))
            logger.info(f"Comparison summary saved as '{comparison_path}'")
        except Exception as e:
            logger.error(f"Failed to generate comparison summary: {e}")
            raise
    elif len(methods) > 1:
        logger.warning("Comparison summary not generated due to incomplete analysis for one or both methods")

    logger.info("Co cluster analysis completed")

if __name__ == "__main__":
    main()