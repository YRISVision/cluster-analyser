import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from pymatgen.core import Structure
import os
import tempfile
import shutil
import importlib.util

# Check if analyse_co_cluster_dbscan_hdbscan module exists
if not os.path.exists("analyse_co_cluster_dbscan_hdbscan.py"):
    st.error("Required module 'analyse_co_cluster_dbscan_hdbscan.py' not found in the current directory. Please ensure it is present.")
    st.stop()

# Import functions from analyse_co_cluster_dbscan_hdbscan
try:
    from analyse_co_cluster_dbscan_hdbscan import compute_pbc_distance_matrix, filter_outliers, analyze_facets, generate_equivalent_normals, generate_visualizations
except ModuleNotFoundError as e:
    st.error(f"Failed to import 'analyse_co_cluster_dbscan_hdbscan' module: {e}. Ensure 'analyse_co_cluster_dbscan_hdbscan.py' is in the same directory.")
    st.stop()

# Streamlit page configuration for a clean, research-oriented layout
st.set_page_config(page_title="Co Cluster Facet Analysis", layout="wide")

def main():
    """
    Main function for the Streamlit GUI to analyze Co cluster facets.
    
    Why: Provides an interactive interface to upload VASP files, configure clustering parameters,
         run DBSCAN/HDBSCAN analysis, and visualize/compare results for small Co clusters (≤38 atoms).
    
    Implementation:
    - Organizes the GUI into sections: file upload, parameter selection, analysis execution, and results.
    - Integrates with analyse_co_cluster_dbscan_hdbscan.py functions for consistency with command-line results.
    - Displays interactive Plotly plots and downloadable outputs.
    
    Tuning:
    - Adjust parameter ranges in sliders (e.g., eps, cosine_threshold) for different cluster sizes.
    - Modify visualization layouts or add features (e.g., coordination analysis) in the results section.
    """
    st.title("Co Cluster Facet Analysis")
    st.markdown("""
        This tool analyzes cobalt (Co) cluster facets from VASP files using DBSCAN or HDBSCAN clustering.
        Upload a VASP file (e.g., relaxed_structure_bfgs.vasp), configure parameters, and compare results.
        Optimized for small clusters (≤38 atoms) on TiO₂ substrates.
    """)

    # File upload section
    st.header("1. Upload VASP File")
    vasp_file = st.file_uploader("Select VASP file", type=["vasp"])
    if vasp_file:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".vasp") as tmp:
            tmp.write(vasp_file.read())
            tmp_vasp_path = tmp.name
        try:
            structure = Structure.from_file(tmp_vasp_path)
            co_atoms = [site for site in structure if site.species_string == "Co"]
            if not co_atoms:
                st.error("No Co atoms found in the VASP file.")
                os.unlink(tmp_vasp_path)
                return
            co_coords = np.array([site.coords for site in co_atoms])
            st.success(f"Loaded VASP file with {len(co_coords)} Co atoms.")

            # Compute and display distance diagnostics
            dist_matrix = compute_pbc_distance_matrix(co_coords, structure.lattice)
            non_zero_distances = dist_matrix[dist_matrix > 0]
            median_distance = np.median(non_zero_distances) if non_zero_distances.size > 0 else 0
            min_distance = np.min(non_zero_distances) if non_zero_distances.size > 0 else 0
            max_distance = np.max(non_zero_distances) if non_zero_distances.size > 0 else 0
            st.write(f"Distance Diagnostics: Median={median_distance:.2f} Å, Min={min_distance:.2f} Å, Max={max_distance:.2f} Å")
            if median_distance > 3.5:
                st.warning("Median Co-Co distance is unusually high (>3.5 Å). Expected ~2.5–3.2 Å for FCC Co. Check VASP file or adjust clustering parameters (e.g., reduce eps).")
        except Exception as e:
            st.error(f"Failed to load VASP file: {e}")
            os.unlink(tmp_vasp_path)
            return
    else:
        st.info("Please upload a VASP file to proceed.")
        return

    # Parameter selection section
    st.header("2. Configure Parameters")
    clustering_method = st.selectbox("Clustering Method", ["Both", "DBSCAN", "HDBSCAN"], help="Choose DBSCAN, HDBSCAN, or both for comparison. DBSCAN uses fixed-radius clustering; HDBSCAN handles variable density.")
    col1, col2 = st.columns(2)
    with col1:
        if clustering_method in ["DBSCAN", "Both"]:
            eps = st.slider("DBSCAN eps (Å)", 2.3, 3.5, 2.5, help="Radius for DBSCAN clustering. Set ~1.2× Co-Co distance (2.5–3.2 Å). Increase for fewer outliers, decrease for more. Default 2.5 Å due to potential distance issues.")
        min_samples = st.slider("Min Samples", 1, 6, 5, help="Minimum points for core points. Lower (e.g., 3) for looser clustering; higher (e.g., 6) for stricter.")
    with col2:
        if clustering_method in ["HDBSCAN", "Both"]:
            min_cluster_size = st.slider("HDBSCAN Min Cluster Size", 4, 12, 10, help="Minimum cluster size for HDBSCAN. Increase (e.g., 12) to avoid splitting; decrease for smaller clusters. Default 10 for stricter clustering.")
        cosine_threshold = st.slider("Cosine Threshold", 0.85, 1.0, 0.90, help="Facet assignment similarity (~26° at 0.90). Lower (e.g., 0.85) for distorted facets due to TiO₂. Default 0.90 to capture distortions.")

    # Run analysis button
    st.header("3. Run Analysis")
    if st.button("Run Analysis"):
        with st.spinner("Running analysis..."):
            methods = ["dbscan", "hdbscan"] if clustering_method == "Both" else [clustering_method.lower()]
            results = {}
            output_dir = tempfile.mkdtemp()
            st.session_state["output_dir"] = output_dir

            for method in methods:
                st.subheader(f"[{method.upper()}] Results")
                try:
                    # Run clustering
                    keep_mask, labels = filter_outliers(co_coords, structure.lattice, method, eps, min_samples, min_cluster_size)
                    filtered_coords = co_coords[keep_mask]
                    filtered_indices = np.where(keep_mask)[0]
                    outlier_indices = np.where(~keep_mask)[0]
                    st.write(f"Filtered Co atoms: {len(filtered_coords)}")
                    st.write(f"Outlier atoms: {len(outlier_indices)} (indices: {outlier_indices.tolist()})")
                    if len(outlier_indices) <= 1:
                        st.warning(f"[{method.upper()}] Only {len(outlier_indices)} outlier detected. Consider reducing eps or min_samples for stricter clustering.")
                    elif len(outlier_indices) > 10:
                        st.warning(f"[{method.upper()}] {len(outlier_indices)} outliers detected. Consider increasing eps or min_cluster_size for fewer outliers.")

                    if len(filtered_coords) < 4:
                        st.error(f"[{method.upper()}] Too few atoms after filtering to compute convex hull.")
                        continue

                    # Run facet analysis and visualizations
                    facet_data, facet_assignments, facet_areas, percentages, total_surface_area, hull = analyze_facets(filtered_coords, method, cosine_threshold)
                    facet_areas, percentages, filtered_indices, outlier_indices = generate_visualizations(
                        co_coords, filtered_coords, filtered_indices, outlier_indices, labels, facet_data, facet_assignments, facet_areas, percentages, hull, method, cosine_threshold
                    )
                    results[method] = {
                        "facet_areas": facet_areas,
                        "percentages": percentages,
                        "filtered_count": len(filtered_coords),
                        "outlier_indices": outlier_indices.tolist()
                    }

                    # Display facet percentages
                    st.write("Facet Exposure Percentages:")
                    df_percentages = pd.DataFrame.from_dict(percentages, orient="index", columns=["Percentage (%)"])
                    st.dataframe(df_percentages.style.format({"Percentage (%)": "{:.2f}"}))
                    if percentages.get("unassigned", 0) > 20:
                        st.warning("High unassigned facet percentage. Consider lowering cosine_threshold (e.g., 0.85) to capture distorted facets.")

                    # Display visualizations
                    st.write("Visualizations:")
                    col3, col4, col5 = st.columns(3)
                    with col3:
                        st.markdown("**Atom Plot**")
                        try:
                            with open(f"co_atoms_{method}.html", "r") as f:
                                st.components.v1.html(f.read(), height=400)
                        except FileNotFoundError:
                            st.error(f"Atom plot (co_atoms_{method}.html) not found. Check analysis logs.")
                    with col4:
                        st.markdown("**Facet Plot**")
                        try:
                            with open(f"co_facets_{method}.html", "r") as f:
                                st.components.v1.html(f.read(), height=400)
                        except FileNotFoundError:
                            st.error(f"Facet plot (co_facets_{method}.html) not found. Check analysis logs.")
                    with col5:
                        st.markdown("**Clustering Plot**")
                        try:
                            with open(f"co_clusters_{method}.html", "r") as f:
                                st.components.v1.html(f.read(), height=400)
                        except FileNotFoundError:
                            st.error(f"Clustering plot (co_clusters_{method}.html) not found. Check analysis logs.")

                    # Provide download links
                    st.write("Download Outputs:")
                    try:
                        with open(f"facet_data_{method}.txt", "rb") as f:
                            st.download_button(f"Download facet_data_{method}.txt", f, file_name=f"facet_data_{method}.txt")
                    except FileNotFoundError:
                        st.error(f"Facet data (facet_data_{method}.txt) not found. Check analysis logs.")
                    for plot in [f"co_atoms_{method}.html", f"co_facets_{method}.html", f"co_clusters_{method}.html"]:
                        try:
                            with open(plot, "rb") as f:
                                st.download_button(f"Download {plot}", f, file_name=plot)
                        except FileNotFoundError:
                            st.error(f"Plot file ({plot}) not found. Check analysis logs.")

                except Exception as e:
                    st.error(f"[{method.upper()}] Analysis failed: {e}")

            # Comparison summary
            if len(methods) > 1:
                st.subheader("Comparison Summary")
                try:
                    with open("comparison_summary.txt", "r") as f:
                        st.text(f.read())
                    with open("comparison_summary.txt", "rb") as f:
                        st.download_button("Download comparison_summary.txt", f, file_name="comparison_summary.txt")

                    # Facet percentage bar chart
                    st.write("Facet Percentage Comparison")
                    df_compare = pd.DataFrame({
                        "Facet": list(results["dbscan"]["percentages"].keys()),
                        "DBSCAN (%)": [results["dbscan"]["percentages"][f] for f in results["dbscan"]["percentages"]],
                        "HDBSCAN (%)": [results["hdbscan"]["percentages"][f] for f in results["hdbscan"]["percentages"]]
                    })
                    fig = px.bar(df_compare, x="Facet", y=["DBSCAN (%)", "HDBSCAN (%)"], barmode="group")
                    st.plotly_chart(fig)
                except FileNotFoundError:
                    st.warning("Comparison summary not generated due to analysis errors. Ensure both DBSCAN and HDBSCAN complete successfully.")

            # Clean up temporary files
            os.unlink(tmp_vasp_path)
            shutil.rmtree(output_dir, ignore_errors=True)

if __name__ == "__main__":
    main()