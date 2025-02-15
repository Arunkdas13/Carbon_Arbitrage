import streamlit as st
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import os

###############################################################################
#                         HELPER FUNCTIONS
###############################################################################
def calculate_rho(beta):
    """
    Helper function for discount rate calculation.

    """
    rho_f = 0.0208
    carp = 0.0299
    # Always subtract 1% from carp
    carp -= 0.01
    # Weighted leverage factor (lambda)
    _lambda = 0.5175273490449868
    # Corporate tax rate
    tax_rate = 0.15

    # Weighted discount rate
    rho = (
        _lambda * rho_f * (1 - tax_rate)
        + (1 - _lambda) * (rho_f + beta * carp)
    )
    return rho

def calculate_discount(rho, deltat):
    """
    Discount factor for deltat years ahead at rate rho.
    """
    return (1 + rho) ** (-deltat)

def EJ2MWh(x):
    """
    Convert exajoules (EJ) to megawatt-hours (MWh).
    """
    joule = x * 1e18  # EJ -> J
    wh = joule / 3600 # J -> Wh
    return wh / 1e6   # Wh -> MWh

def EJ2Mcoal(x):
    """
    Convert exajoules (EJ) to million tonnes of coal (approx).
    """
    # 1 tonne of coal ~ 29.3076 GJ (approx).
    # x EJ -> x * 1e9 GJ, then /29.3076 => million tonnes, then /1e6.
    coal = x * 1e9 / 29.3076
    return coal / 1e6

def calculate_emissions_and_production(scenario, df_ngfs, beta):
    """
    For a given scenario, returns a dict with:
      - "emissions" (GtCO2), cumulative 2023–2100
      - "production_2022" (million tonnes of coal)
      - "production_discounted" (EJ), discounted sum 2023–2100
    """
    # Constants from the original snippet
    coal_emissions_2022_iea = 15.5  # from IEA
    years_5_step = list(range(2010, 2101, 5))
    full_years = range(2023, 2101)

    sub = df_ngfs[df_ngfs.Scenario == scenario]

    # Emissions (in million tonnes CO2 -> GtCO2 by dividing by 1e3)
    emissions_row = sub[sub.Variable == "Emissions|CO2"].iloc[0]
    emissions_values = [emissions_row[str(y)] / 1e3 for y in years_5_step]
    f_e = interp1d(years_5_step, emissions_values)

    # Sum yearly emissions from 2023 to 2100
    total_emissions = sum(f_e(y) for y in full_years)
    # Rescale to 2022 IEA coal-based emissions
    # (We assume we're focusing on the portion from coal.)
    total_emissions *= coal_emissions_2022_iea / f_e(2022)

    # Production (EJ/yr)
    production_row = sub[sub.Variable == "Primary Energy|Coal"].iloc[0]
    production_values = [production_row[str(y)] for y in years_5_step]
    f_p = interp1d(years_5_step, production_values)

    # Discount rate
    rho = calculate_rho(beta)

    # Discounted sum of coal production (in EJ)
    production_discounted = sum(
        f_p(y) * calculate_discount(rho, y - 2022)
        for y in full_years
    )

    return {
        "emissions": total_emissions,  # GtCO2
        "production_2022": EJ2Mcoal(f_p(2022)),  # million tonnes of coal
        "production_discounted": production_discounted,  # EJ
    }

def calculate_cost_and_benefit(
    social_cost_of_carbon=80.0,
    global_lcoe_average=59.25,
    beta=0.9132710997126332,
    df_ngfs=None
):
    """
    Computes:
      - avoided_emissions (GtCO2)
      - cost (trillion USD)
      - benefit (trillion USD)
      - coal_production_2022 (million tonnes of coal)
    using two NGFS scenarios:
      - NGFS2_Current Policies
      - NGFS2_Net-Zero 2050
    """
    # The user must pass in the NGFS DataFrame
    if df_ngfs is None:
        raise ValueError("df_ngfs must not be None. Load the data before calling.")

    # Calculate for both scenarios
    ep_cps = calculate_emissions_and_production(
        scenario="NGFS2_Current Policies",
        df_ngfs=df_ngfs,
        beta=beta
    )
    ep_nz2050 = calculate_emissions_and_production(
        scenario="NGFS2_Net-Zero 2050",
        df_ngfs=df_ngfs,
        beta=beta
    )

    # Emissions avoided
    avoided_emissions = ep_cps["emissions"] - ep_nz2050["emissions"]

    # Production difference (in EJ, discounted)
    discounted_production_increase = (
        ep_cps["production_discounted"] - ep_nz2050["production_discounted"]
    )
    discounted_production_increase_mwh = EJ2MWh(discounted_production_increase)

    # Cost = LCOE * discounted production difference
    cost = global_lcoe_average * discounted_production_increase_mwh
    cost /= 1e12  # convert to trillion dollars

    # Benefit = avoided emissions * social cost of carbon
    # avoided_emissions is in GtCO2, so multiply by scc, then / 1e3 for trillion
    benefit = avoided_emissions * social_cost_of_carbon / 1e3  # trillion dollars

    return {
        "avoided_emissions": avoided_emissions,  # GtCO2
        "cost": cost,                            # trillion USD
        "benefit": benefit,                      # trillion USD
        "arbitrage": benefit - cost,             # trillion USD
        "coal_2022": ep_cps["production_2022"]   # million tonnes
    }

###############################################################################
#                         MAIN STREAMLIT APP
###############################################################################
def main():
    st.title("Carbon Arbitrage Opportunity Calculator")

    st.markdown(
        """
This Streamlit app calculates the _carbon arbitrage opportunity_ by estimating
the net benefit (in **trillions of dollars**) of shifting from a "Current Policies"
coal production/consumption scenario to a "Net-Zero 2050" scenario, based on
public data from IEA [International Energy Agency] (https://www.iea.org/) and NGFS [Network for Greening the Financial System] (https://www.ngfs.net/en).

1. Adjust the parameters in the side‐bar.  
2. See the resulting cost, benefit, and carbon arbitrage.  
3. Then pick **which parameter** to sweep, and see how the 4 outcomes change across a range of values.
        """
    )

    # ------------------
    # Sidebar inputs
    # ------------------
    with st.sidebar:
        st.header("Model Parameters")

        social_cost_of_carbon = st.slider(
            "Social Cost of Carbon (USD per ton CO₂)",
            min_value=1,
            max_value=200,
            value=80,
            step=1
        )

        global_lcoe_average = st.slider(
            "Global average LCOE (USD/MWh for renewables)",
            min_value=1.0,
            max_value=200.0,
            value=59.25,
            step=1.0
        )

        beta = st.number_input(
            "Unleveraged beta (advanced)",
            value=0.9132710997,
            min_value=0.0,
            max_value=2.0,
            step=0.01
        )

    # ------------------
    # Load data
    # ------------------
    # Expect the CSV file in ./data subfolder
    data_path = os.path.join("data", "ar6_snapshot_1700882949.csv")
    if not os.path.isfile(data_path):
        st.error(
            f"CSV file not found at: {data_path}\n"
            "Please place the CSV in a 'data' folder or update the path."
        )
        return

    df_ngfs = pd.read_csv(data_path)

    # ------------------
    # Core calculations for user-chosen parameters
    # ------------------
    results = calculate_cost_and_benefit(
        social_cost_of_carbon=social_cost_of_carbon,
        global_lcoe_average=global_lcoe_average,
        beta=beta,
        df_ngfs=df_ngfs
    )

    # Unpack results
    avoided_emissions = results["avoided_emissions"]
    cost = results["cost"]
    benefit = results["benefit"]
    arbitrage = results["arbitrage"]
    coal_2022 = results["coal_2022"]

    st.subheader("Single‐Point Results for Current Sliders")
    st.write(f"**Global coal production in 2022**: {coal_2022:.2f} million tonnes of coal")
    st.write(f"**Total emissions prevented**: {avoided_emissions:.2f} GtCO₂")
    st.write(f"**Cost**: {cost:.2f} trillion dollars")
    st.write(f"**Benefit**: {benefit:.2f} trillion dollars")
    st.write(
        f"**Carbon arbitrage opportunity** = {arbitrage:.2f} trillion dollars"
    )

    # ------------------
    # Plot: how do these results vary with one chosen parameter?
    # ------------------
    st.subheader("Parameter Sweep Plots")

    # Let user select which parameter to vary
    param_choice = st.selectbox(
        "Which parameter would you like to vary?",
        ("Social Cost of Carbon (USD per ton CO₂)", "Global LCOE (USD/MWh for renewables)", "Beta (advanced)")
    )

    # We'll define a range of values for the chosen parameter:
    # - For Social Cost of Carbon: 10..200, step 10
    # - For Global LCOE: 10..200, step 10
    # - For Beta: 0..2, step 0.1
    if param_choice == "Social Cost of Carbon":
        param_values = list(range(1, 201, 10))  # [10, 20, 30, ... 200]
    elif param_choice == "Global LCOE":
        param_values = list(range(1, 201, 10))  # [10, 20, 30, ... 200]
    else:  # Beta
        # We'll use floating steps (0.0 -> 2.0)
        param_values = [x / 10.0 for x in range(0, 21)]  # 0.0..2.0 step=0.1

    # We'll hold the other parameters constant at the values chosen by the user
    # and just sweep param_choice in param_values.
    list_emissions = []
    list_cost = []
    list_benefit = []
    list_arbitrage = []

    for val in param_values:
        if param_choice == "Social Cost of Carbon":
            # Vary SCC
            res = calculate_cost_and_benefit(
                social_cost_of_carbon=val,
                global_lcoe_average=global_lcoe_average,
                beta=beta,
                df_ngfs=df_ngfs
            )
        elif param_choice == "Global LCOE":
            # Vary LCOE
            res = calculate_cost_and_benefit(
                social_cost_of_carbon=social_cost_of_carbon,
                global_lcoe_average=val,
                beta=beta,
                df_ngfs=df_ngfs
            )
        else:
            # Vary Beta
            res = calculate_cost_and_benefit(
                social_cost_of_carbon=social_cost_of_carbon,
                global_lcoe_average=global_lcoe_average,
                beta=val,
                df_ngfs=df_ngfs
            )

        list_emissions.append(res["avoided_emissions"])
        list_cost.append(res["cost"])
        list_benefit.append(res["benefit"])
        list_arbitrage.append(res["arbitrage"])

    # Build a DataFrame for plotting
    df_plot = pd.DataFrame({
        "Parameter": param_values,
        "Emissions (GtCO2)": list_emissions,
        "Cost (trillion $)": list_cost,
        "Benefit (trillion $)": list_benefit,
        "Arbitrage (trillion $)": list_arbitrage
    })

    # Streamlit line chart expects an index for the x-axis
    df_plot = df_plot.set_index("Parameter")

    # Show the data frame if you like
    # st.dataframe(df_plot)

    # Plot all 4 metrics in a single line chart
    st.line_chart(df_plot)

    st.markdown(
        """
**Interpretation**: Each line shows how the result changes when you vary the chosen parameter
across a range of plausible values, while holding the other two parameters fixed at the
sidebar’s settings.
        """
    )


# -----------------------------------------------------------------------------
# Run the app
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()