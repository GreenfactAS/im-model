import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from im_model.cost_model import CostModel
from im_model.utils import to_multidimensional_array
idx = pd.IndexSlice

# Function to update the charts based on dropdown selection
def update_tech_mix_emissions_charts(df, segment, region):
    # Filter the dataframe based on segment and region
    tech_mix = df[(df['variable'] == "technology_mix") & (df['segment'] == segment) & (df['region'] == region)]
    emissions = df[(df['variable'] == "emissions") & (df['segment'] == segment) & (df['region'] == region)]
    total_emissions = df[(df['variable'] == "emissions")].groupby(['year', "segment"])['value'].sum().reset_index() 

    # Create subplots for technology mix, emissions, and total emissions
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Stacked area chart for technology mix
    axes[0].stackplot(
        tech_mix.year.unique(), *(tech_mix['value'][tech_mix.technology == t] for t in tech_mix['technology'].unique()), labels=tech_mix['technology'].unique())
    axes[0].set_xlabel('Year')
    axes[0].set_ylabel('Technology Mix')
    axes[0].legend(title='Technology')
    axes[0].grid(True)

    # Stacked area chart for emissions
    axes[1].stackplot(emissions.year.unique(), *(emissions['value'][emissions.technology == t].div(1e6) for t in emissions['technology'].unique()), labels=emissions['technology'].unique())
    axes[1].set_title('Emissions')
    axes[1].set_xlabel('Year')
    axes[1].set_ylabel('Emissions (MtCO2e)')
    axes[1].legend(title='Emission Type')
    axes[1].grid(True)

    plt.tight_layout()  # Adjust the spacing between subplots
    plt.show()


#The find_x_pos function used for plotting! (out of scope)
def find_x_pos(widths):
    cumulative_widths = [0]
    cumulative_widths.extend(np.cumsum(widths))
    half_widths = [i/2 for i in widths]
    x_pos = []
    for i in range(0, len(half_widths)):
        x_pos.append(half_widths[i] + cumulative_widths[i])
    return x_pos

def plot_mac_curve(df):
    df_filtered = df[df.variable.isin(["abatement_potential", "mac"])]
    pivoted_df = df_filtered.pivot(index=['year', 'region', 'segment', 'technology'], columns='variable', values='value')
    grouped_df = pivoted_df.groupby(['segment', 'technology']).agg({'abatement_potential': 'sum', 'mac': 'mean'}).reset_index().sort_values(by='mac', ascending=True)
    #Prepare the data for plotting
    width_group = grouped_df['abatement_potential']
    height_group = grouped_df['mac']
    new_x_group = find_x_pos(width_group)

    # Get the segment for each data point
    segments = grouped_df['segment']
    # Generate a color map based on the available segments
    segment_colors = {segment: color for segment, color in zip(segments.unique(), sns.color_palette())}
    # Get the colors for each segment
    colors = [segment_colors.get(segment, 'gray') for segment in segments]

    plt.figure(figsize=(9,6))
    plt.bar(new_x_group, height_group, width=width_group, edgecolor='black', color=colors)
    plt.xlabel('Abatement Potential')
    plt.ylabel('Abatement Cost')
    
    # Add legend
    handles = [plt.Rectangle((0,0),1,1, color=color) for color in segment_colors.values()]
    labels = segment_colors.keys()
    plt.legend(handles, labels)
    plt.grid(True)
    plt.show()

# Define the interactive function
def plot_abatement_curves(cost_curve_df, year, segment):
    # Create a dropdown menu for year selection
    cost_curve_grouped = cost_curve_df.groupby(["Year", "Segment"]).sum()

    plt.figure(figsize=(10, 6))
    cost_curve_grouped.loc[year].loc[segment].div(1e6).T.plot()
    plt.legend()
    plt.xlabel('Price (EUR / tCO2e)')
    plt.ylabel('Abatement (MtCO2e)')
    plt.title(f'Cost Curve for Year {year}')
    plt.grid(True)
    plt.show()


def plot_production_costs(data_model, pd_data_dict, segment, year, region):
    # Testing
    cost_model = CostModel(
        to_multidimensional_array(pd_data_dict["other_opex"].loc[idx[segment]]),
        to_multidimensional_array(pd_data_dict["commodity_use"].loc[idx[segment]]), 
        to_multidimensional_array(pd_data_dict["commodity_prices"]),
        to_multidimensional_array(pd_data_dict["capex"].loc[idx[:,segment,:]])
    )

    # Test update_capex
    disagg_opex = cost_model.update_opex(return_disaggregated_opex=True)
    lifetime = pd_data_dict["asset_lifetime"][:,segment][0]
    beta = pd_data_dict["β"][:,segment][0]
    other_opex = pd_data_dict["other_opex"].loc[idx[segment]]

    multiindex = pd.MultiIndex.from_product(
        [data_model["regions"], 
            data_model["technologies"][segment], 
            sorted(data_model["commodities"])]
    )
    # Create a list of years, starting with the years in the data model, and then the years after
    year_index = \
        [int(yy) for yy in data_model["years"]] \
    + [int(data_model["years"][-1]) + x for x in range(1, lifetime + 1)]

    df = pd.DataFrame(
        disagg_opex.reshape(-1, disagg_opex.shape[-1]), 
        index=multiindex, 
        columns=year_index
    )
    capex_copy = pd_data_dict["capex"].loc[idx[:,segment,:]].copy()*((1-beta)/(1-beta**lifetime))
    capex_copy["commodity"] = "Capex"
    capex_copy.set_index(
        "commodity", 
        append=True, 
        inplace=True
    )

    capex_copy.columns = year_index

    multiindex_dummy = pd.MultiIndex.from_product([ 
        data_model["technologies"][segment],
        data_model["regions"]
    ])

    multiindex_dummy.names = ["technology", "region"]
    dummy = pd.Series(index=multiindex_dummy, data=1)
    other_opex_with_regions = other_opex.multiply(dummy, axis="index")
    other_opex_with_regions = other_opex_with_regions.reorder_levels([1,0])
    other_opex_with_regions["commodity"] = "Other opex"
    other_opex_with_regions.set_index("commodity", append=True, inplace=True)
    other_opex_with_regions.columns = year_index
    all_costs = pd.concat([df, capex_copy, other_opex_with_regions])
    all_costs.index.names = ["region", "technology", "commodity"]

    df = all_costs.loc[region,:][year]
    df = df[df > 0].unstack("commodity")
    plt.figure(figsize=(10, 6))
    # Plotting the stacked bar chart
    ax = df.plot(kind='bar', stacked=True)

    # Rotate the x-axis labels
    plt.xticks(rotation=45)

    # Optionally, adjust font size
    plt.xticks(fontsize=9)

    # Setting labels and title
    plt.xlabel('Technology', fontsize=9)
    plt.ylabel('TCO (EUR / tonne of product)', fontsize=9)
    plt.title('Technology cost', fontsize=9)
    plt.grid(True)
    plt.show()