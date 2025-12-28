# BessBidder: Multi-Market Battery Bidding for Continous Intraday and Day-ahead Market
![Python](https://img.shields.io/badge/python-3.12-blue.svg)

**Continuous Intraday Trading: An Open-Source Multi-Market Bidding Framework for Energy Storage Systems**
*by Kim K. Miskiw, Jan Ludwig, Leo Semmelmann, Christof Weinhardt*
Presented at the 15th ACM International Conference on Future Energy Systems (ACM e-Energy 2025), Rotterdam

---

This repository contains the code and data accompanying the paper:

> **Continuous Intraday Trading: An Open-Source Multi-Market Bidding Framework for Energy Storage Systems**
> Accepted at *ACM e-Energy 2025*
> \(https://dl.acm.org/doi/full/10.1145/3679240.3734631)


---

## Table of Contents

* [Setup](#setup)
* [Database Configuration](#database-configuration)
* [Data Acquisition](#data-acquisition)
* [Single-Market Bidding](#single-market-bidding)
* [Multi-Market Bidding](#multi-market-bidding)

  * [Myopic Bidding](#myopic-bidding)
  * [Coordinated Bidding (DRL)](#coordinated-bidding)
* [Results & Plots](#results--plots)
* [Code Formatting](#code-formatting)
* [License](#license)

## Setup

Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Database Configuration

### PostgreSQL Installation (Required)

PostgreSQL must be installed and running for example locall as a Windows service. 

**Windows Installation:**
1. Download the EDB PostgreSQL installer from [postgresql.org/download/windows](https://www.postgresql.org/download/windows/)
2. Run the installer and follow these settings:
   - Set a password for the `postgres` superuser (e.g., `postgres`)
   - **Enable "Install as a service"** (important!)
   - Use default port `5432`
   - Finish the installer
3. Verify the service is running:
   ```powershell
   Get-Service | Where-Object {$_.Name -like 'postgresql*'} | Format-Table Name, Status
   ```

### Configure Local Authentication

To enable password-free local development connections, edit `pg_hba.conf`:

1. Open the file (adjust the version number if needed):
   ```powershell
   notepad "C:\Program Files\PostgreSQL\16\data\pg_hba.conf"
   ```
2. Find the IPv4 and IPv6 local connection lines and change the authentication method from `scram-sha-256` (or `md5`) to `trust`:
   ```
   # IPv4 local connections:
   host    all             all             127.0.0.1/32            trust
   # IPv6 local connections:
   host    all             all             ::1/128                 trust
   ```
3. Save and restart the PostgreSQL service:
   ```powershell
   Restart-Service -Name postgresql-x64-16
   ```

### Create Database User and Database

To prepare a local PostgreSQL instance run the following commands with your chosen <username>:

```powershell
psql -U postgres -h 127.0.0.1
```

Then in the psql prompt:

```sql
CREATE USER <username>;
CREATE DATABASE epex_data;
GRANT ALL PRIVILEGES ON DATABASE epex_data TO <username>;
ALTER DATABASE epex_data OWNER TO <username>;
```

Exit psql with `\q`.

### Create `.env` file for credentials:

This file manages all access credentials for the database and APIs. Specify them before starting to work with the notebook. *IMPORTANT NOTE*: EPEX Spot data is not open-source and for you to use the notebook, you will have to have bought the data. 

```env
ENTSOE_API_KEY=...
POSTGRES_DB_NAME=...
POSTGRES_DB_HOST=127.0.0.1
POSTGRES_USER=<username>
EPEX_SFTP_HOST=...
EPEX_SFTP_PORT=...
EPEX_SFTP_USER=...
EPEX_SFTP_PW=...
```

### Define study scope in `config.py`

In this file, you can define technical parameters of the model, the time horizon, and output paths. Please configure these settings before starting data acquisition, as the download depends on your study setup. 


## Data Acquisition

Execute the script `01b_data_acquisition.py` to fill the database with ENTSO-E and EPEX Spot data for the reproduction of the study. Note that the framework allows downloading data for 2019–2023.

Steps:

1. Load ENTSO-E generation/load data
2. Load intraday transaction data (pre/post 2022 formats)

Alternatively, run the scripts in `src/data_acquisition/epex_sftp/` directly.

These scripts collect all input data required to run the simulation, except for EXAA prices which are used as price forecasts. Access to a dedicated API is available but incurs additional cost. As a workaround, we currently use historical EPEX spot prices as a proxy, which implies a perfect-foresight assumption in the simulation setup. The results in the orginial paper, however, are calculated on the EXAA prices as forecasts. 

## Single-Market Bidding

### Day-ahead Market (MILP Benchmark)

Run `02a_single_market_day_ahead_milp.py` to compute benchmark DA bids using MILP.
Results are saved to `output/single_market/day_ahead_milp/`.

### Intraday Market (Rolling Intrinsic)

Run `02b_single_market_rolling_intrinsic_h.py` with CLI arguments:

```bash
python 02b_single_market_rolling_intrinsic_h.py
```

To increase the speed of the rolling intrinsic algorithm we switched the implementation to gurobipy. To use this one needs a gurobipy license which can be obtained free of charge for academics.


Logs will be saved in `output/single_market/rolling_intrinsic/ri_basic/`

## Multi-Market Bidding

### Myopic Bidding

Run `03_myopic-multi_market.py` to combine MILP-based DA bidding with rolling intrinsic-based intraday bidding.

### Coordinated Bidding (Deep Reinforcement Learning)

#### Data Preparation

Run `04a_transform_data_for_coordinated_multi_market.py` to add features (e.g. IDFull) and split train/test data.

#### Training

Run `04b_train_coordinated_multi_market.py` to train the PPO agent.
Choose `intraday_product_type='H'` or `'QH'`. Model and tensorboard logs are versioned under `output/coordinated_multi_market/`.

#### Testing

Run `04c_test_coordinated_multi_market.py` to evaluate agent behavior.
This feeds DA decisions into rolling intrinsic logic and outputs test logs and reports.

## Results & Plots

Use the following scripts to create figures and tables:

* `plot_ACMeEnergy_Single_Market.py`
* `plot_ACMeEnergy_Myopic_Multi_Market.py`
* `plot_ACMeEnergy_Coordinated_Multi_Market.py`

Plots are saved in their respective `output` subfolder under `figures`.


## Code Formatting

This repository uses [Black](https://github.com/psf/black) for code formatting and [pre-commit](https://pre-commit.com/) hooks.

To install:

```bash
pip install pre-commit
pre-commit install
```

To format all files:

```bash
pre-commit run --all-files
```

---

## Contributors

This project was developed in collaboration between members of the Karlsruhe Institute of Technology (KIT).

| Name                    | Role & Contribution                                                                 |
|-------------------------|-------------------------------------------------------------------------------------|
| **Kim K. Miskiw**       | Project lead, DRL model implementation, results analysis, repo maintenance, documentation, debugging and validation, writing               |
| **Jan Ludwig**          | Data pipeline (ENTSO-E, EPEX), DRL model implementation, MILP modelling, rolling intrinsic strategies, debugging and validation              |
| **Leo Semmelmann**      | Data pipeline (ENTSO-E, EPEX), rolling intrinsic strategies, conceptual support, review               |
| **Christof Weinhardt** | Supervisory role, conceptual support     

## License

This repository is licensed under the [GNU Affero General Public License v3.0](./LICENSES/AGPL-3.0-or-later.txt).
This ensures that derivative works must be released under the same license.

## Contact

For questions or early access inquiries, contact:
[kim.miskiw@kit.edu](mailto:kim.miskiw@kit.edu)
