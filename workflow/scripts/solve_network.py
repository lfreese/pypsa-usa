"""
Solves optimal operation and capacity for a network with the option to
iteratively optimize while updating line reactances.

This script is used for optimizing the electrical network as well as the
sector coupled network.

Description
-----------

Total annual system costs are minimised with PyPSA. The full formulation of the
linear optimal power flow (plus investment planning
is provided in the
`documentation of PyPSA <https://pypsa.readthedocs.io/en/latest/optimal_power_flow.html#linear-optimal-power-flow>`_.

The optimization is based on the :func:`network.optimize` function.
Additionally, some extra constraints specified in :mod:`solve_network` are added.

.. note::

    The rules ``solve_elec_networks`` and ``solve_sector_networks`` run
    the workflow for all scenarios in the configuration file (``scenario:``)
    based on the rule :mod:`solve_network`.
"""

import copy
import logging
import os

import numpy as np
import pandas as pd
import pypsa
import xarray as xr
import yaml
from _helpers import (
    configure_logging,
    update_config_from_wildcards,
    update_config_with_sector_opts,
)
from linopy import LinearExpression, QuadraticExpression

logger = logging.getLogger(__name__)
pypsa.pf.logger.setLevel(logging.WARNING)


def get_region_buses(n, region_list):
    return n.buses[
        (
            n.buses.country.isin(region_list)
            | n.buses.reeds_zone.isin(region_list)
            | n.buses.reeds_state.isin(region_list)
            | n.buses.interconnect.str.lower().isin(region_list)
            | n.buses.nerc_reg.isin(region_list)
            | (1 if "all" in region_list else 0)
        )
    ]


def filter_components(
    n: pypsa.Network,
    component_type: str,
    planning_horizon: str | int,
    carrier_list: list[str],
    region_buses: pd.Index,
    extendable: bool,
):
    """
    Filter components based on common criteria.

    Parameters
    ----------
    - n: pypsa.Network
        The PyPSA network object.
    - component_type: str
        The type of component (e.g., "Generator", "StorageUnit").
    - planning_horizon: str or int
        The planning horizon to filter active assets.
    - carrier_list: list
        List of carriers to filter.
    - region_buses: pd.Index
        Index of region buses to filter.
    - extendable: bool, optional
        If specified, filters by extendable or non-extendable assets.

    Returns
    -------
    - pd.DataFrame
        Filtered assets.
    """
    component = n.df(component_type)
    if planning_horizon != "all":
        ph = int(planning_horizon)
        iv = n.investment_periods
        active_components = n.get_active_assets(component.index.name, iv[iv >= ph][0])
    else:
        active_components = component.index

    # Links will throw the following attribute error, as we must specify bus0
    # AttributeError: 'DataFrame' object has no attribute 'bus'. Did you mean: 'bus0'?
    bus_name = "bus0" if component_type.lower() == "link" else "bus"

    filtered = component.loc[
        active_components
        & component.carrier.isin(carrier_list)
        & component[bus_name].isin(region_buses)
        & (component.p_nom_extendable == extendable)
    ]

    return filtered


def add_land_use_constraints(n):
    """
    Adds constraint for land-use based on information from the generators
    table.

    Constraint is defined by land-use per carrier and land_region. The
    definition of land_region enables sub-bus level land-use
    constraints.
    """
    # breakpoint()
    model = n.model
    generators = n.generators.query(
        "p_nom_extendable & land_region != '' ",
    ).rename_axis(index="Generator-ext")

    if generators.empty:
        return
    p_nom = n.model["Generator-p_nom"].loc[generators.index]

    grouper = pd.concat([generators.carrier, generators.land_region], axis=1)
    lhs = p_nom.groupby(grouper).sum()

    maximum = generators.groupby(["carrier", "land_region"])["p_nom_max"].max()
    maximum = maximum[np.isfinite(maximum)]

    rhs = xr.DataArray(maximum).rename(dim_0="group")
    index = rhs.indexes["group"].intersection(lhs.indexes["group"])

    if not index.empty:
        logger.info("Adding land-use constraints")
        model.add_constraints(
            lhs.sel(group=index) <= rhs.loc[index],
            name="land_use_constraint",
        )


def prepare_network(
    n,
    solve_opts=None,
):
    if "clip_p_max_pu" in solve_opts:
        for df in (
            n.generators_t.p_max_pu,
            n.generators_t.p_min_pu,
            n.storage_units_t.inflow,
        ):
            df = df.where(df > solve_opts["clip_p_max_pu"], other=0.0)

    load_shedding = solve_opts.get("load_shedding")
    if load_shedding:
        # intersect between macroeconomic and surveybased willingness to pay
        # http://journal.frontiersin.org/article/10.3389/fenrg.2015.00055/full
        # TODO: retrieve color and nice name from config
        logger.warning("Adding load shedding generators.")
        n.add("Carrier", "load", color="#dd2e23", nice_name="Load shedding")
        buses_i = n.buses.query("carrier == 'AC'").index
        if not np.isscalar(load_shedding):
            # TODO: do not scale via sign attribute (use Eur/MWh instead of Eur/kWh)
            load_shedding = 1e2  # Eur/kWh

        n.madd(
            "Generator",
            buses_i,
            " load",
            bus=buses_i,
            carrier="load",
            sign=1e-3,  # Adjust sign to measure p and p_nom in kW instead of MW
            marginal_cost=load_shedding,  # Eur/kWh
            p_nom=1e9,  # kW
        )

    if solve_opts.get("noisy_costs"):  ##random noise to costs of generators
        for t in n.iterate_components():
            if "marginal_cost" in t.df:
                t.df["marginal_cost"] += 1e-2 + 2e-3 * (np.random.random(len(t.df)) - 0.5)

        for t in n.iterate_components(["Line", "Link"]):
            t.df["capital_cost"] += (1e-1 + 2e-2 * (np.random.random(len(t.df)) - 0.5)) * t.df["length"]

    if solve_opts.get("nhours"):
        nhours = solve_opts["nhours"]
        n.set_snapshots(n.snapshots[:nhours])
        n.snapshot_weightings[:] = 8760 / nhours

    return n


lookup = pd.read_csv(
    os.path.join(os.path.dirname(__file__), "..", "variables.csv"),
    index_col=["component", "variable"],
)


def prep_brownfield(n, planning_horizon):
    """Prepare the network for the next planning horizon in myopic solving. Original version."""
    # electric transmission grid set optimised capacities of previous as minimum
    n.lines.s_nom_min = n.lines.s_nom_opt  # for lines
    # Set DC links minimum capacity to previous optimal (allows expansion, prevents contraction)
    dc_i = n.links[n.links.carrier == "DC"].index
    n.links.loc[dc_i, "p_nom_min"] = n.links.loc[dc_i, "p_nom_opt"]  # for links

    for c in n.iterate_components(["Link", "Generator", "StorageUnit"]):
        nm = c.name
        # limit our components that we remove/modify to those prior to this time horizon
        c_lim = c.df.loc[n.get_active_assets(nm, planning_horizon)]

        logger.info(f"Preparing brownfield for the component {nm}")
        # attribute selection for naming convention
        attr = "p"
        # copy over asset sizing from previous period
        c_lim[f"{attr}_nom"] = c_lim[f"{attr}_nom_opt"]
        df = copy.deepcopy(c_lim)
        time_df = copy.deepcopy(c.pnl)

        for c_idx in c_lim.index:
            n.remove(nm, c_idx)
        for df_idx in df.index:
            if nm == "Generator":
                n.madd(
                    nm,
                    [df_idx],
                    carrier=df.loc[df_idx].carrier,
                    bus=df.loc[df_idx].bus,
                    p_nom_min=df.loc[df_idx].p_nom_min,
                    p_nom=df.loc[df_idx].p_nom,
                    p_nom_max=df.loc[df_idx].p_nom_max,
                    p_nom_extendable="False",
                    ramp_limit_up=df.loc[df_idx].ramp_limit_up,
                    ramp_limit_down=df.loc[df_idx].ramp_limit_down,
                    efficiency=df.loc[df_idx].efficiency,
                    marginal_cost=df.loc[df_idx].marginal_cost,
                    capital_cost=df.loc[df_idx].capital_cost,
                    build_year=df.loc[df_idx].build_year,
                    lifetime=df.loc[df_idx].lifetime,
                    heat_rate=df.loc[df_idx].heat_rate,
                    fuel_cost=df.loc[df_idx].fuel_cost,
                    vom_cost=df.loc[df_idx].vom_cost,
                    carrier_base=df.loc[df_idx].carrier_base,
                    p_min_pu=df.loc[df_idx].p_min_pu,
                    p_max_pu=df.loc[df_idx].p_max_pu,
                    land_region=df.loc[df_idx].land_region,
                )  ##TODO: fix and see how dispatch variables
            else:
                # For Links, ensure proper brownfield setup
                if nm == "Link":
                    df_attrs = df.loc[df_idx].copy()
                    # df_attrs["p_nom_extendable"] = False ## RETAIN SAME EXTENDABILITY AS BEFORE
                    n.add(nm, df_idx, **df_attrs)
                else:
                    # For StorageUnits and Generators, make them non-extendable in brownfield
                    df_attrs = df.loc[df_idx].copy()
                    df_attrs["p_nom_extendable"] = False  # Brownfield assets are not extendable
                    n.add(nm, df_idx, **df_attrs)
        logger.info(f"Consistency check, issues: {n.consistency_check()}")

        # copy time-dependent
        selection = n.component_attrs[nm].type.str.contains(
            "series",
        )

        for tattr in n.component_attrs[nm].index[selection]:
            n.import_series_from_dataframe(time_df[tattr], nm, tattr)

    ##### DEBUGGING #####
    for name in ["generators", "storage_units", "links"]:
        df = getattr(n, name)
        print(f"{name} after brownfield NaNs:\n", df.isna().sum())
        # Optionally print p_nom_min/max stats
        if "p_nom_min" in df.columns:
            print(f"{name} p_nom_min: min={df['p_nom_min'].min()}, max={df['p_nom_min'].max()}")
        if "p_nom_max" in df.columns:
            print(f"{name} p_nom_max: min={df['p_nom_max'].min()}, max={df['p_nom_max'].max()}")

    # roll over the last snapshot of time varying storage state of charge to be the state_of_charge_initial for the next time period
    n.storage_units.loc[:, "state_of_charge_initial"] = n.storage_units_t.state_of_charge.loc[planning_horizon].iloc[-1]


def constant_cost(n, config, ref_year, model_kwargs):  # , **kwargs):
    """Based on conversation with Koen and based on approach for MGA.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network object.
    config : dict
        Configuration dictionary containing various settings.
    ref_year : int
        The reference year for existing cost calculations. This must be a year in the existing_n file.
    """
    # pull costs from existing network
    pathfile = config["electricity"]["cost_constraints_path"]

    if not os.path.exists(pathfile):
        logger.error(f"Reference network file not found: {pathfile}")
        raise FileNotFoundError(f"Could not find reference network at {pathfile}")

    existing_n = pypsa.Network(pathfile)
    # period_sns = sns[sns.get_level_values("period") == period_values[0]]

    optimal_cost = (existing_n.statistics.capex().sum() + existing_n.statistics.opex().sum())[ref_year]
    fixed_cost = existing_n.statistics.installed_capex().sum()[
        ref_year
    ]  ##TODO: fix this to be fixed cost of our current system
    current_yr = model_kwargs.get("snapshots").get_level_values("period").unique()
    # breakpoint()
    logger.info(f"Snapshots for our objective in constant cost {current_yr}")
    # get the objective from our model
    m = n.optimize.create_model(**model_kwargs)  ## TODO: fix the snapshots issue
    original_objective = m.objective
    logger.info(f"{original_objective}")
    # breakpoint()
    if not isinstance(original_objective, LinearExpression | QuadraticExpression):
        original_objective = original_objective.expression

    name = "system_cost"
    scale_factor = 1e-9

    name = "system_cost"
    if name in n.global_constraints.index:
        n.global_constraints = n.global_constraints.drop(name)

    n.add(
        "GlobalConstraint",
        name=name,
        type="budget",
        carrier_attribute="",
        sense="<=",
        constant=optimal_cost * scale_factor,
    )

    buffer = 1.02  # needs 2% buffer for feasibility in just one 2030 time period
    n.model.add_constraints(
        (original_objective + fixed_cost) * scale_factor <= optimal_cost * scale_factor * buffer,
        name=f"GlobalConstraint-{name}",
    )

    logger.info(f"Created constant cost constraint: Total system cost <= ${optimal_cost / 1e9:.2f}B")


def define_objective_co2(n, sns):
    """Defines and writes out the objective function for CO2 minimization."""
    weightings = n.snapshot_weightings.loc[n.snapshots]
    period = sns.unique("period") if "period" in sns.names else [sns[0]]
    total_emissions = 0

    period_sns = sns[sns.get_level_values("period") == period.values[0]] if "period" in sns.names else sns
    logger.info(f" Emissions constraint time period for {period.values[0]}")
    period_weighting = weightings.loc[period_sns, "generators"]
    emissions = n.carriers.co2_emissions.fillna(0)
    # emitting_carriers = emissions[emissions != 0].index

    active_gens = n.get_active_assets("Generator", period)
    gens_em = n.generators.loc[active_gens].query("carrier in @emitting_carriers")

    efficiency = gens_em["efficiency"].replace(0, 1)
    em_pu = gens_em["carrier"].map(emissions) / efficiency
    p_em = n.model["Generator-p"].loc[period_sns, gens_em.index]

    # Convert em_pu to xarray with proper coordinates to match p_em
    em_pu_xr = xr.DataArray(
        em_pu.values,
        coords=[em_pu.index],
        dims=["Generator"],
    )

    # Convert period_weighting to xarray with proper coordinates
    period_weighting_xr = xr.DataArray(
        period_weighting.values,
        coords=[period_weighting.index],
        dims=["snapshot"],
    )

    # Calculate emissions for this period
    period_emissions = p_em * em_pu_xr * period_weighting_xr
    total_emissions += period_emissions.sum()
    objective = total_emissions  # * penalty_factor
    logger.info(
        f"Total emissions expression: {total_emissions}",
    )  ## TODO: ensure this is for the right time period when multiple horizons
    return objective


def run_optimize(n, model_kwargs, **kwargs):
    """Initiate the correct type of pypsa.optimize function."""
    # load the constraint
    config = n.config
    logger.warning(f"Loading constraint: network id={id(n)}, n.model id={id(getattr(n, 'model', None))}")
    ref_year = config["scenario"]["ref_year"]

    print("\nAPPLYING COST CONSTRAINT")
    constant_cost(n, config, ref_year, model_kwargs)

    add_land_use_constraints(n)
    # load the objective
    print("\nSETTING CO2 MINIMIZATION OBJECTIVE")
    objective = define_objective_co2(n, model_kwargs.get("snapshots"))
    n.model.objective = objective

    # solve the model
    logger.info(f"CALLING solve_model on network id={id(n)}, n.model id={id(getattr(n, 'model', None))}")
    status, condition = n.optimize.solve_model(**kwargs)
    logger.info(f"RETURN from solve_model: n.model id is now {id(n.model)}")

    if status != "ok":
        logger.warning(
            f"Solving status '{status}' with termination condition '{condition}'",
        )
    if "infeasible" in condition or status != "ok":
        print("Model is infeasible, checking infeasibilities...")
        raise RuntimeError("Solving status 'infeasible'")


def prepare_solver_options(solving):
    # The following solver setup follows that of `solve_network` in `solve_network.py`:
    set_of_options = solving["solver"]["options"]
    cf_solving = solving["options"]

    kwargs = {}

    kwargs["solver_options"] = solving["solver_options"][set_of_options] if set_of_options else {}
    kwargs["solver_name"] = solving["solver"]["name"]
    # kwargs["extra_functionality"] = extra_functionality
    kwargs["assign_all_duals"] = cf_solving.get("assign_all_duals", False)
    kwargs["io_api"] = cf_solving.get("io_api", None)

    model_kwargs = {}
    model_kwargs["transmission_losses"] = cf_solving.get("transmission_losses", False)
    model_kwargs["linearized_unit_commitment"] = cf_solving.get(
        "linearized_unit_commitment",
        False,
    )

    if kwargs["solver_name"] == "gurobi":
        logging.getLogger("gurobipy").setLevel(logging.CRITICAL)

    if "model_options" in solving:
        model_kwargs = model_kwargs | solving["model_options"]

        if "solver_dir" in model_kwargs and "$" in model_kwargs["solver_dir"]:
            # Resolve env var as path
            model_kwargs["solver_dir"] = os.path.expandvars(model_kwargs["solver_dir"])
            logger.info(f"Set solver_dir to {model_kwargs['solver_dir']}")

    return kwargs, model_kwargs


def solve_network(n, config, solving, opts="", **kwargs):
    kwargs, model_kwargs = prepare_solver_options(solving)
    n.config = config
    n.opts = opts

    # Check for key options
    foresight = snakemake.params.foresight
    logger.info(f"Using {foresight} foresight")

    if not n.lines.s_nom_extendable.any():
        logger.warning("No expandable lines found.")

    match foresight:
        case "perfect":
            run_optimize(n, model_kwargs, **kwargs)
        case "myopic":
            # total_cumulative_cost = 0
            for i, planning_horizon in enumerate(n.investment_periods):
                logger.info(
                    f"Starting optimization for planning horizon {planning_horizon} ({i + 1}/{len(n.investment_periods)})",
                )
                sns_horizon = n.snapshots[n.snapshots.get_level_values(0) == planning_horizon]

                # Add sns_horizon to kwarg
                model_kwargs["snapshots"] = sns_horizon
                model_kwargs["multi_investment_periods"] = True
                if len(sns_horizon) == 0:
                    raise ValueError(f"Snapshots for planning horizon {planning_horizon} are empty!")

                logger.info(f"Set snapshots for current planning horizon, only solve for this period {sns_horizon}")
                run_optimize(n, model_kwargs, **kwargs)

                if i == len(n.investment_periods) - 1:
                    logger.info(f"Final time horizon {planning_horizon}")
                    continue

                logger.info(f"Preparing brownfield from {planning_horizon}")
                prep_brownfield(n, planning_horizon)
        case _:
            raise ValueError(f"Invalid foresight option: '{foresight}'. Must be 'perfect' or 'myopic'.")

    return n


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "solve_network",
            interconnect="western",
            simpl="12",
            clusters="4m",
            ll="v1.0",
            opts="4h",
            sector="E-G",
            planning_horizons="2030",
        )
    configure_logging(snakemake)
    update_config_from_wildcards(snakemake.config, snakemake.wildcards)
    if "sector_opts" in snakemake.wildcards.keys():
        update_config_with_sector_opts(
            snakemake.config,
            snakemake.wildcards.sector_opts,
        )

    opts = snakemake.wildcards.opts
    if "sector_opts" in snakemake.wildcards.keys():
        opts += "-" + snakemake.wildcards.sector_opts
    opts = [o for o in opts.split("-") if o != ""]
    solve_opts = snakemake.params.solving["options"]

    # sector specific co2 options
    if snakemake.wildcards.sector != "E":
        # sector co2 limits applied via config file, not through Co2L
        opts = [x for x in opts if not x.startswith("Co2L")]
        opts.append("sector")

    np.random.seed(solve_opts.get("seed", 123))

    n = pypsa.Network(snakemake.input.network)

    n = prepare_network(
        n,
        solve_opts,
    )

    n = solve_network(
        n,
        config=snakemake.config,
        solving=snakemake.params.solving,
        opts=opts,
        log_fn=snakemake.log.solver,
    )

    n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
    n.export_to_netcdf(snakemake.output[0])
    with open(snakemake.output.config, "w") as file:
        yaml.dump(
            n.meta,
            file,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )
