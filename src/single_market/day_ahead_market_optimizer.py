import numpy as np
import pandas as pd
import pyomo.environ as pyo
from loguru import logger
from pyomo.opt import SolverFactory


class DayAheadMarketOptimizationModel:
    def __init__(
        self,
        time_index,
        da_prices_forecast,
        da_prices,
        battery_capacity,
        charge_rate,
        discharge_rate,
        efficiency,
        max_cycles,
        start_end_soc,
    ):
        self.time_index = time_index
        self.da_prices_forecast = da_prices_forecast
        self.da_prices = da_prices
        self.battery_capacity = battery_capacity
        self.charge_rate = charge_rate
        self.discharge_rate = discharge_rate
        self.efficiency = efficiency
        self.max_cycles = max_cycles
        self.start_end_soc = start_end_soc
        self.time_periods = len(da_prices_forecast)

        # Create the model
        self.model = pyo.ConcreteModel()

        # Time periods
        self.model.T = pyo.RangeSet(0, self.time_periods - 1)

        # Variables
        self.model.spot_buy = pyo.Var(
            self.model.T,
            domain=pyo.NonNegativeReals,
            bounds=(
                0,
                self.charge_rate,
            ),  # load at metering point is what we can charge max
        )
        self.model.spot_sell = pyo.Var(
            self.model.T,
            domain=pyo.NonNegativeReals,
            bounds=(
                0,
                self.discharge_rate,
            ),  # load that reaches the grid is max dis * eff
        )
        self.model.soc = pyo.Var(
            self.model.T, bounds=(0, self.battery_capacity)
        )  # State of charge
        self.model.charge = pyo.Var(
            self.model.T, bounds=(0, self.charge_rate)
        )  # Charging power
        self.model.discharge = pyo.Var(
            self.model.T, bounds=(0, self.discharge_rate)
        )  # Discharging power

        # Binary variables for prohibiting simultaneous charge and discharge
        self.model.buy_indicator = pyo.Var(self.model.T, domain=pyo.Binary)

        # Objective function: Maximize revenue
        self.model.objective = pyo.Objective(
            rule=self.objective_function, sense=pyo.maximize
        )

        # Constraints
        self.model.charge_definition = pyo.Constraint(
            self.model.T, rule=self.charge_definition
        )
        self.model.discharge_definition = pyo.Constraint(
            self.model.T, rule=self.discharge_definition
        )
        self.model.soc_balance = pyo.Constraint(self.model.T, rule=self.soc_balance)
        self.model.final_soc_constraint = pyo.Constraint(rule=self.final_soc_constraint)
        self.model.prevent_overcharging = pyo.Constraint(
            self.model.T, rule=self.prevent_overcharging
        )
        self.model.prevent_overdrawing = pyo.Constraint(
            self.model.T, rule=self.prevent_overdrawing
        )
        self.model.no_simultaneous_buy_sell = pyo.Constraint(
            self.model.T, rule=self.no_simultaneous_buy_sell
        )
        self.model.no_simultaneous_sell_buy = pyo.Constraint(
            self.model.T, rule=self.no_simultaneous_sell_buy
        )
        self.model.max_discharge_power_constraint = pyo.Constraint(
            self.model.T, rule=self.max_discharge_power_constraint
        )
        self.model.max_charge_power_constraint = pyo.Constraint(
            self.model.T, rule=self.max_charge_power_constraint
        )
        self.model.max_cycles_constraint = pyo.Constraint(
            rule=self.max_cycles_constraint
        )

        self.model.prevent_overcommitment = pyo.Constraint(
            self.model.T, rule=self.prevent_overcommitment
        )
        self.model.prevent_overcommitment_discharge = pyo.Constraint(
            self.model.T, rule=self.prevent_overcommitment_discharge
        )

    def objective_function(self, model):
        revenue = 0
        for t in model.T:
            revenue += (
                self.da_prices_forecast[t]
                * (-1)
                * model.spot_buy[t]
                * 1
                / self.efficiency
                + self.da_prices_forecast[t] * model.spot_sell[t] * self.efficiency
            )
        return revenue

    def charge_definition(self, model, t):
        return model.charge[t] == model.spot_buy[t]

    def discharge_definition(self, model, t):
        return model.discharge[t] == model.spot_sell[t]

    def soc_balance(self, model, t):
        if t == 0:
            return model.soc[t] == self.start_end_soc
        else:
            return (
                model.soc[t]
                == model.soc[t - 1] + model.charge[t - 1] - model.discharge[t - 1]
            )

    def final_soc_constraint(self, model):
        return (
            self.start_end_soc
            + sum((model.charge[t] for t in model.T))
            - sum((model.discharge[t] for t in model.T))
            == self.start_end_soc
        )

    def prevent_overcharging(self, model, t):
        if t == 0:
            return pyo.Constraint.Skip
        return model.soc[t - 1] + model.charge[t] <= self.battery_capacity

    def prevent_overdrawing(self, model, t):
        if t == 0:
            return pyo.Constraint.Skip
        return model.soc[t - 1] - model.discharge[t] >= 0

    def no_simultaneous_buy_sell(self, model, t):
        return model.spot_buy[t] <= model.buy_indicator[t] * self.charge_rate

    def no_simultaneous_sell_buy(self, model, t):
        return model.spot_sell[t] <= (1 - model.buy_indicator[t]) * self.discharge_rate

    def max_discharge_power_constraint(self, model, t):
        return model.discharge[t] <= self.discharge_rate

    def max_charge_power_constraint(self, model, t):
        return model.charge[t] <= self.charge_rate

    def max_cycles_constraint(self, model):
        return sum(
            model.charge[t] + model.discharge[t] for t in model.T
        ) <= self.max_cycles * (2 * self.battery_capacity)

    def prevent_overcommitment(self, model, t):
        return model.charge[t] <= self.charge_rate

    def prevent_overcommitment_discharge(self, model, t):
        return model.discharge[t] <= self.discharge_rate

    def solve(self):
        # Solve the model
        # opt = SolverFactory("gurobi_direct")
        # opt = SolverFactory("glpk")
        opt = SolverFactory("gurobi")
        results = opt.solve(self.model, tee=True)

        # Print results
        print("Status:", results.solver.status)
        print("Termination Condition:", results.solver.termination_condition)

        # Extract and store the optimized values
        self.soc_values = [pyo.value(self.model.soc[t]) for t in self.model.T]
        self.charge_values = [pyo.value(self.model.charge[t]) for t in self.model.T]
        self.discharge_values = [
            pyo.value(self.model.discharge[t]) for t in self.model.T
        ]

        self.spot_buy_values = [pyo.value(self.model.spot_buy[t]) for t in self.model.T]
        self.spot_sell_values = [
            pyo.value(self.model.spot_sell[t]) for t in self.model.T
        ]

        self.revenue = pyo.value(self.model.objective)

        # Print the results
        self._log_results()
        results = self._create_result_df()
        return results

    def _log_results(self):
        logger.info("State of Charge: {}", self.soc_values)
        logger.info("Buy: {}", self.spot_buy_values)
        logger.info("Sell: {}", self.spot_sell_values)
        logger.info("Charge: {}", self.charge_values)
        logger.info("Discharge: {}", self.discharge_values)
        logger.info("Revenue: {}", self.revenue)

    def _create_result_df(self) -> pd.DataFrame:
        results = pd.DataFrame.from_dict(
            {
                "time": self.time_index,
                "volume_buy_bid": np.array(self.spot_buy_values),
                "volume_sell_bid": np.array(self.spot_sell_values),
                "capacity_trade": -np.array(self.spot_buy_values)
                + np.array(self.spot_sell_values),
                "volume_charge": np.array(self.charge_values),
                "volume_discharge": np.array(self.discharge_values),
                "epex_spot_60min_de_lu_eur_per_mwh_f": self.da_prices_forecast.round(
                    2
                ).flatten(),
                "epex_spot_60min_de_lu_eur_per_mwh": self.da_prices.flatten(),
                "soc": np.array(self.soc_values),
            }
        )

        results.set_index("time", inplace=True)
        results.index = results.index.tz_convert("utc")
        results["buy_indicator"] = results["volume_buy_bid"] != 0.0
        results["sell_indicator"] = results["volume_sell_bid"] != 0.0

        # charge_got_hit_indicator = results.epex_spot_60min_de_lu_eur_per_mwh_f >= results.epex_spot_60min_de_lu_eur_per_mwh
        # charge_got_hit_indicator = charge_got_hit_indicator.astype(int)

        results["charge_costs"] = (
            (-1)
            * results.volume_buy_bid
            * results.epex_spot_60min_de_lu_eur_per_mwh
            * 1
            / self.efficiency
        )

        # discharge_got_hit_indicator = results.epex_spot_60min_de_lu_eur_per_mwh_f <= results.epex_spot_60min_de_lu_eur_per_mwh
        # discharge_got_hit_indicator = discharge_got_hit_indicator.astype(int)

        results["discharge_revenues"] = (
            results.volume_sell_bid
            * results.epex_spot_60min_de_lu_eur_per_mwh
            * self.efficiency
        )

        return results
