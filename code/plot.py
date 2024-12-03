import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import os
import pandas as pd

def plot_precipitation(ax, lon, lat, prec, min_prec_threshold=1):
    """
    Plot the precipitation data on the map.

    Parameters:
    - ax: Matplotlib axes object.
    - lon: 2D array of longitudes.
    - lat: 2D array of latitudes.
    - prec: 2D array of precipitation values.
    - min_prec_threshold: Minimum precipitation threshold for masking (mm/hr).
    """

    # Set up the color scheme for precipitation
    cmap = plt.cm.RdYlGn_r  # Radar-like reversed colormap
    levels = [1, 2, 3, 4, 6, 8, 10, 15, 20]
    norm = plt.cm.colors.BoundaryNorm(levels, ncolors=cmap.N, clip=False)
    

    # Mask precipitation below the threshold
    prec_masked = np.ma.masked_less(prec, min_prec_threshold)
    
    # Make sure that data below the threshold is white or blue (ocean)
    cmap.set_under('white')
    cmap.set_bad('white')


    # Plot the precipitation data
    precip_plot = ax.pcolormesh(lon, lat, prec_masked, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())

    # Create a colorbar
    cbar = plt.colorbar(precip_plot, ax=ax, orientation="vertical", label="Precipitation (mm)")

    return precip_plot


def plot_mcs_regions(ax, df_mcs):
    """
    Plot the identified MCS regions on the map.

    Parameters:
    - ax: Matplotlib axes object.
    - df_mcs: DataFrame containing MCS regions.
    """
    import numpy as np

    # Get unique region labels
    region_labels = df_mcs['region_label'].unique()

    # Define a colormap
    cmap = plt.cm.get_cmap('tab20', len(region_labels))

    # Plot each region with a different color
    for idx, region_label in enumerate(region_labels):
        region_data = df_mcs[df_mcs['region_label'] == region_label]
        ax.scatter(region_data['lon'], region_data['lat'], s=1, color=cmap(idx), transform=ccrs.PlateCarree(), label=f'Region {int(region_label)}')

    # Optionally, add a legend
    #ax.legend(loc='upper right', markerscale=5, fontsize='small')

def save_detection_plot(lon, lat, prec, final_labeled_regions, file_time, output_dir, min_prec_threshold=0.1):
    """
    Generate and save a plot of the detected MCS regions for a single time step.

    Parameters:
    - lon: 2D array of longitudes.
    - lat: 2D array of latitudes.
    - prec: 2D array of precipitation values.
    - final_labeled_regions: 2D array of labeled MCS regions.
    - file_time: The time associated with the file (used for the filename and title).
    - output_dir: Directory where the plot will be saved.
    - min_prec_threshold: Minimum precipitation threshold for masking (default is 0.1 mm).
    """
    

    # Create a DataFrame from the final_labeled_regions
    mcs_indices = np.where(final_labeled_regions > 0)
    if len(mcs_indices[0]) == 0:
        print(f"No MCS regions detected at time {file_time}. Skipping plot.")
        return

    mcs_lat = lat[mcs_indices]
    mcs_lon = lon[mcs_indices]
    mcs_prec_values = prec[mcs_indices]
    mcs_region_labels = final_labeled_regions[mcs_indices]

    df_mcs = pd.DataFrame({
        'lat': mcs_lat,
        'lon': mcs_lon,
        'precip': mcs_prec_values,
        'region_label': mcs_region_labels
    })

    # Create a figure and axes with Cartopy projection
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={"projection": ccrs.PlateCarree()})

    # Set the extent of the map (adjust as necessary)
    ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())

    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue")

    # Plot precipitation data
    plot_precipitation(ax, lon, lat, prec, min_prec_threshold)

    # Plot MCS regions
    plot_mcs_regions(ax, df_mcs)

    # Add title
    ax.set_title(f'MCS Detection at Time {file_time}')

    # Save the plot
    output_filename = f"mcs_detection_{file_time}.png"
    output_filepath = os.path.join(output_dir, output_filename)
    plt.savefig(output_filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close the figure to free memory

    print(f"Saved plot for time {file_time} to {output_filepath}")

    
def save_intermediate_plots(detection_result, output_dir):
    lon = detection_result['lon']
    lat = detection_result['lat']
    precipitation = detection_result['precipitation']
    final_labeled_regions = detection_result['final_labeled_regions']
    moderate_prec_clusters = detection_result['moderate_prec_clusters']
    convective_plumes = detection_result['convective_plumes']
    file_time = detection_result['time']
    file_time_str = np.datetime_as_string(file_time, unit='h')

    # Plot precipitation field
    # Create a figure and axes with Cartopy projection
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={"projection": ccrs.PlateCarree()})

    # Set the extent of the map (adjust as necessary)
    ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())

    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue")

    # Plot precipitation data
    plot_precipitation(ax, lon, lat, precipitation, min_prec_threshold=0.1)
    ax.set_title(f'Precipitation in mm at {file_time_str}')
    plt.savefig(f"{output_dir}/precipitation_{file_time_str}.png", dpi=300)
    plt.close(fig)

    # Plot moderate prec clusters
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.coastlines()
    clusters_plot = ax.pcolormesh(lon, lat, moderate_prec_clusters, cmap='tab20', shading='auto')
    plt.colorbar(clusters_plot, ax=ax, label='Cluster Label')
    ax.set_title(f'Moderate prec Clusters at {file_time_str}')
    plt.savefig(f"{output_dir}/moderate_clusters_{file_time_str}.png", dpi=300)
    plt.close(fig)


    # Final labeled precipitation clusters
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.coastlines()
    clusters_plot = ax.pcolormesh(lon, lat, final_labeled_regions, cmap='tab20', shading='auto')
    plt.colorbar(clusters_plot, ax=ax, label='Cluster Label')
    ax.set_title(f'MCS Clusters at {file_time_str}')
    plt.savefig(f"{output_dir}/MCS_clusters_{file_time_str}.png", dpi=300)
    plt.close(fig)

    # Plot convective plumes
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.coastlines()
    plumes_plot = ax.pcolormesh(lon, lat, convective_plumes, cmap='tab20', shading='auto')
    plt.colorbar(plumes_plot, ax=ax, label='Convective Plume Label')
    ax.set_title(f'Convective Plumes at {file_time_str}')
    plt.savefig(f"{output_dir}/convective_plumes_{file_time_str}.png", dpi=300)
    plt.close(fig)

