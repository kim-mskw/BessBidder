import warnings

from dotenv import load_dotenv

from src.shared.config import (
    BUCKET_SIZE,
    C_RATE,
    MAX_CYCLES_PER_YEAR,
    MIN_TRADES,
    RTE,
    START,
    END,
)

from src.single_market.rolling_intrinsic_gurobi_qh import simulate_period

load_dotenv()

warnings.simplefilter(action="ignore", category=FutureWarning)

if __name__ == "__main__":
    simulate_period(
        START,
        END,
        discount_rate=0,
        c_rate=C_RATE,
        roundtrip_eff=RTE,
        max_cycles=MAX_CYCLES_PER_YEAR,
        min_trades=MIN_TRADES,
    )
