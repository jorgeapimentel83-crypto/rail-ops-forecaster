"""
Synthetic data generator for Phase 1: terminal dwell forecasting.
 
Generates daily terminal-level observations for 8 classification yards
over a configurable date range. The data is synthetic but calibrated to
publicly reported freight rail operating ranges (STB weekly performance
reports, AAR carload statistics, Class I earnings disclosures).
 
Target variable:
    target_dwell_hours is NEXT-DAY dwell — for a row dated Day N,
    the target is the dwell that actually occurs on Day N+1. This
    means the model's task is forecasting, not description.
 
Generation logic:
    1. Features for Day N are generated from terminal profiles.
    2. Same-day dwell is computed from Day N's features (intermediate).
    3. Same-day dwell is shifted forward by one day within each terminal.
    4. Day N's target becomes Day N+1's actual dwell.
    5. The intermediate same-day column is removed from the final output.
 
Business formula for same-day dwell:
    dwell = base_dwell
          × volume_pressure(inbound_cars, cars_on_hand, yard_occupancy)
          × resource_factor(crew_starts, locomotive_availability)
          × weekend_adjustment
          + noise
          + congestion_spike (if active)
 
The multiplication structure means factors compound — high volume AND low
crew availability together produce a larger dwell increase than either alone.
"""
 
import numpy as np
import pandas as pd
from dataclasses import dataclass
 
 
# ── Terminal profiles ──────────────────────────────────────────────────
# Each terminal has distinct operating characteristics. Base dwell ranges
# from 16–23h, consistent with STB-reported Class I averages (15–28h).
# Capacity, typical volume, and staffing levels define the terminal's
# normal operating envelope. Dwell behavior emerges from how daily
# conditions compare to that envelope.
 
@dataclass
class TerminalProfile:
    terminal_id: str
    terminal_name: str
    region: str
    base_dwell_hours: float
    avg_daily_inbound_trains: float  # typical inbound trains per day
    avg_daily_inbound_cars: float    # typical inbound cars per day
    yard_capacity_cars: int          # total track capacity in cars
    avg_crew_starts: int             # typical daily crew starts available
    weekend_dwell_shift: float       # multiplier on weekends (>1 = higher dwell)
 
 
TERMINAL_PROFILES = [
    TerminalProfile(
        terminal_id="T01", terminal_name="Barstow", region="West",
        base_dwell_hours=22.0, avg_daily_inbound_trains=12,
        avg_daily_inbound_cars=280, yard_capacity_cars=900,
        avg_crew_starts=18, weekend_dwell_shift=0.97,
    ),
    TerminalProfile(
        terminal_id="T02", terminal_name="Alliance", region="Midwest",
        base_dwell_hours=20.0, avg_daily_inbound_trains=14,
        avg_daily_inbound_cars=320, yard_capacity_cars=1100,
        avg_crew_starts=22, weekend_dwell_shift=0.98,
    ),
    TerminalProfile(
        terminal_id="T03", terminal_name="Galesburg", region="Midwest",
        base_dwell_hours=23.0, avg_daily_inbound_trains=11,
        avg_daily_inbound_cars=260, yard_capacity_cars=750,
        avg_crew_starts=15, weekend_dwell_shift=1.03,
    ),
    TerminalProfile(
        terminal_id="T04", terminal_name="Memphis", region="South",
        base_dwell_hours=21.0, avg_daily_inbound_trains=13,
        avg_daily_inbound_cars=300, yard_capacity_cars=950,
        avg_crew_starts=20, weekend_dwell_shift=1.01,
    ),
    TerminalProfile(
        terminal_id="T05", terminal_name="Kansas City", region="Midwest",
        base_dwell_hours=19.0, avg_daily_inbound_trains=15,
        avg_daily_inbound_cars=340, yard_capacity_cars=1200,
        avg_crew_starts=24, weekend_dwell_shift=0.96,
    ),
    TerminalProfile(
        terminal_id="T06", terminal_name="Stockton", region="West",
        base_dwell_hours=17.0, avg_daily_inbound_trains=8,
        avg_daily_inbound_cars=180, yard_capacity_cars=550,
        avg_crew_starts=12, weekend_dwell_shift=1.02,
    ),
    TerminalProfile(
        terminal_id="T07", terminal_name="Amarillo", region="South",
        base_dwell_hours=18.0, avg_daily_inbound_trains=9,
        avg_daily_inbound_cars=200, yard_capacity_cars=600,
        avg_crew_starts=13, weekend_dwell_shift=1.00,
    ),
    TerminalProfile(
        terminal_id="T08", terminal_name="Havre", region="North",
        base_dwell_hours=16.0, avg_daily_inbound_trains=6,
        avg_daily_inbound_cars=130, yard_capacity_cars=400,
        avg_crew_starts=9, weekend_dwell_shift=1.04,
    ),
]
 
 
# ── Seasonal volume patterns ──────────────────────────────────────────
# Monthly multipliers on inbound volume. Grain harvest (Sep–Nov) and
# pre-holiday intermodal (Aug–Oct) drive peaks. Jan–Feb are lightest.
 
MONTHLY_VOLUME_FACTORS = {
    1: 0.88,   # January — post-holiday lull
    2: 0.90,   # February — winter low
    3: 0.95,   # March — gradual ramp
    4: 0.98,   # April — spring volumes building
    5: 1.00,   # May — baseline
    6: 1.02,   # June — steady
    7: 1.00,   # July — slight holiday dip
    8: 1.06,   # August — intermodal peak begins
    9: 1.10,   # September — grain harvest starts
    10: 1.12,  # October — harvest + intermodal peak
    11: 1.05,  # November — harvest tailing off
    12: 0.93,  # December — holiday slowdown
}
 
 
def generate_terminal_features(
    profile: TerminalProfile,
    dates: pd.DatetimeIndex,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Generate daily feature observations for a single terminal.
 
    Each feature is generated from the terminal's profile with realistic
    day-to-day variability. Features are generated independently of the
    target — the target is computed from features in a separate step.
    This keeps the data generation process transparent and auditable.
 
    Parameters
    ----------
    profile : TerminalProfile
        Operating characteristics for this terminal.
    dates : pd.DatetimeIndex
        Date range to generate.
    rng : np.random.Generator
        Seeded random number generator for reproducibility.
 
    Returns
    -------
    pd.DataFrame
        One row per date with all Phase 1 feature columns.
    """
    n_days = len(dates)
    months = dates.month
    is_weekend = dates.dayofweek.isin([5, 6]).astype(int)
 
    # ── Seasonal volume multipliers ──
    seasonal_factors = np.array([MONTHLY_VOLUME_FACTORS[m] for m in months])
 
    # ── Inbound train count ──
    # Normal day-to-day variability around the terminal's average,
    # scaled by seasonal pattern. Weekends typically see ~15% fewer trains.
    weekend_volume_reduction = np.where(is_weekend, 0.85, 1.0)
    train_mean = (
        profile.avg_daily_inbound_trains * seasonal_factors * weekend_volume_reduction
    )
    inbound_trains = rng.poisson(lam=train_mean)
    inbound_trains = np.clip(inbound_trains, 1, None)  # at least 1 train
 
    # ── Inbound car count ──
    # Cars per train varies. Average is ~23 cars/train for manifest,
    # higher for unit trains. We use the terminal's avg ratio with noise.
    avg_cars_per_train = profile.avg_daily_inbound_cars / profile.avg_daily_inbound_trains
    cars_per_train = rng.normal(loc=avg_cars_per_train, scale=3.0, size=n_days)
    cars_per_train = np.clip(cars_per_train, 10, 50)
    inbound_cars = np.round(inbound_trains * cars_per_train).astype(int)
 
    # ── Cars on hand ──
    # This is a stock variable — it depends on yesterday's level plus
    # today's arrivals minus today's departures. We simulate a simple
    # inventory process where departures roughly track arrivals with lag.
    #
    # Start at a reasonable steady-state occupancy (~55-65% of capacity).
    cars_on_hand = np.zeros(n_days, dtype=float)
    initial_coh = int(profile.yard_capacity_cars * rng.uniform(0.55, 0.65))
    cars_on_hand[0] = initial_coh
 
    for i in range(1, n_days):
        # Departures are noisy — roughly equal to arrivals but with lag
        # and some mean-reversion toward steady state.
        steady_state = profile.yard_capacity_cars * 0.60
        mean_reversion = 0.15 * (steady_state - cars_on_hand[i - 1])
        departures = inbound_cars[i - 1] + mean_reversion + rng.normal(0, 10)
        departures = max(departures, inbound_cars[i - 1] * 0.5)
 
        cars_on_hand[i] = cars_on_hand[i - 1] + inbound_cars[i] - departures
 
        # Floor at 10% capacity, cap at 98% (embargo would kick in)
        cars_on_hand[i] = np.clip(
            cars_on_hand[i],
            profile.yard_capacity_cars * 0.10,
            profile.yard_capacity_cars * 0.98,
        )
 
    cars_on_hand = np.round(cars_on_hand).astype(int)
 
    # ── Yard occupancy ──
    # Derived directly from cars_on_hand / capacity.
    yard_occupancy_pct = np.round(cars_on_hand / profile.yard_capacity_cars * 100, 1)
 
    # ── Crew starts available ──
    # Daily crew availability fluctuates around the terminal's average.
    # Weekends see ~20% fewer crew starts (rest days, HOS patterns).
    # Month-to-month variation is mild.
    crew_base = profile.avg_crew_starts * np.where(is_weekend, 0.80, 1.0)
    crew_starts = rng.poisson(lam=crew_base)
    crew_starts = np.clip(crew_starts, 2, None)  # minimum staffing
 
    # ── Locomotive availability ──
    # Expressed as percentage. Normally 80–95%. Drops correlate mildly
    # with high volume (more power in use) and weekends (shop schedules).
    loco_base = rng.normal(loc=88.0, scale=4.0, size=n_days)
    # Slight depression when volume is high (more locos already assigned)
    volume_ratio = inbound_cars / profile.avg_daily_inbound_cars
    loco_adjustment = -3.0 * (volume_ratio - 1.0)  # high volume → lower avail
    loco_availability = np.clip(loco_base + loco_adjustment, 60.0, 99.0)
    loco_availability = np.round(loco_availability, 1)
 
    return pd.DataFrame({
        "date": dates,
        "terminal_id": profile.terminal_id,
        "terminal_name": profile.terminal_name,
        "region": profile.region,
        "inbound_train_count": inbound_trains,
        "inbound_car_count": inbound_cars,
        "cars_on_hand": cars_on_hand,
        "yard_occupancy_pct": yard_occupancy_pct,
        "crew_starts_available": crew_starts,
        "locomotive_availability_pct": loco_availability,
        "is_weekend": is_weekend,
        "month": months,
    })
 
 
def compute_same_day_dwell(
    df: pd.DataFrame,
    profiles: dict[str, TerminalProfile],
    rng: np.random.Generator,
    congestion_spike_probability: float = 0.04,
    congestion_spike_hours: tuple[float, float] = (5.0, 12.0),
) -> pd.Series:
    """
    Compute same-day dwell from features using business rules.
 
    IMPORTANT: This produces an intermediate value — the dwell that
    *actually occurred* on each day given that day's conditions. It is
    NOT the final prediction target. The generate_dataset function
    shifts this column forward by one day to create the next-day target.
 
    The formula:
        dwell = base_dwell
              × volume_pressure
              × resource_factor
              × weekend_adjustment
              + noise
              + congestion_spike
 
    Parameters
    ----------
    df : pd.DataFrame
        Feature dataframe from generate_terminal_features.
    profiles : dict
        Mapping of terminal_id to TerminalProfile.
    rng : np.random.Generator
        Seeded random number generator.
    congestion_spike_probability : float
        Probability that any given terminal-day has a congestion event.
    congestion_spike_hours : tuple
        Min and max additional dwell hours during a spike.
 
    Returns
    -------
    pd.Series
        same_day_dwell_hours for each row. This is an intermediate
        column used only to derive the shifted next-day target.
    """
    dwell = np.zeros(len(df), dtype=float)
 
    for terminal_id, group in df.groupby("terminal_id"):
        idx = group.index
        p = profiles[terminal_id]
 
        # ── Volume pressure factor ──
        # Ratio of actual to average volume. Values > 1 mean heavier
        # than normal. We use inbound cars as the primary volume signal
        # and add a yard occupancy penalty that accelerates above 80%.
        car_ratio = group["inbound_car_count"].values / p.avg_daily_inbound_cars
        occupancy = group["yard_occupancy_pct"].values / 100.0
 
        # Linear contribution from car volume
        volume_linear = 1.0 + 0.3 * (car_ratio - 1.0)
 
        # Nonlinear occupancy penalty: mild below 80%, steep above
        excess_occupancy = np.maximum(occupancy - 0.80, 0.0)
        occupancy_penalty = np.where(
            occupancy > 0.80,
            1.0 + 1.5 * excess_occupancy ** 1.5,
            1.0 + 0.2 * (occupancy - 0.60),
        )
 
        volume_pressure = volume_linear * occupancy_penalty
 
        # ── Resource factor ──
        # More crew and locomotive availability → lower dwell.
        # We compute a ratio to average and invert it: when resources
        # are above average, the factor is < 1.0 (reducing dwell).
        crew_ratio = group["crew_starts_available"].values / p.avg_crew_starts
        # Avoid division issues — clip to reasonable range
        crew_ratio = np.clip(crew_ratio, 0.3, 2.0)
        # Invert: high crew → low factor
        crew_factor = 1.0 + 0.4 * (1.0 - crew_ratio)
 
        loco_ratio = group["locomotive_availability_pct"].values / 88.0
        loco_ratio = np.clip(loco_ratio, 0.5, 1.2)
        loco_factor = 1.0 + 0.2 * (1.0 - loco_ratio)
 
        resource_factor = crew_factor * loco_factor
 
        # ── Weekend adjustment ──
        weekend_adj = np.where(
            group["is_weekend"].values == 1,
            p.weekend_dwell_shift,
            1.0,
        )
 
        # ── Combine multiplicative factors ──
        dwell_raw = p.base_dwell_hours * volume_pressure * resource_factor * weekend_adj
 
        # ── Additive noise ──
        # Normal noise scaled to ~8% of base dwell. This represents
        # unobserved day-to-day variability (mechanical issues, unusual
        # car types, interchange delays, etc.)
        noise_std = p.base_dwell_hours * 0.08
        noise = rng.normal(0, noise_std, size=len(idx))
 
        # ── Congestion spikes ──
        # Rare events (3-5% of days) where dwell jumps sharply.
        # These represent compounding operational failures.
        spike_mask = rng.random(size=len(idx)) < congestion_spike_probability
        spike_hours = rng.uniform(
            congestion_spike_hours[0],
            congestion_spike_hours[1],
            size=len(idx),
        )
        spikes = np.where(spike_mask, spike_hours, 0.0)
 
        # ── Final dwell ──
        dwell_final = dwell_raw + noise + spikes
 
        # Floor at 6 hours (even an empty terminal has minimum processing)
        # Cap at 60 hours (beyond that, real embargoes would intervene)
        dwell_final = np.clip(dwell_final, 6.0, 60.0)
 
        dwell[idx] = np.round(dwell_final, 1)
 
    return pd.Series(dwell, index=df.index, name="same_day_dwell_hours")
 
 
def generate_dataset(
    start_date: str = "2022-01-01",
    end_date: str = "2024-12-31",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate the full Phase 1 synthetic dataset.
 
    The target variable (target_dwell_hours) represents NEXT-DAY dwell:
    for each row dated Day N, the target is the dwell that actually
    occurs on Day N+1. This means the model's task is:
 
        "Given today's operational conditions, predict tomorrow's dwell."
 
    Implementation:
        1. Generate features for all terminals and all dates.
        2. Compute same-day dwell from each day's features (intermediate).
        3. Sort by terminal and date.
        4. Shift same-day dwell forward by 1 day within each terminal,
           so Day N's target becomes Day N+1's actual dwell.
        5. Drop the last day for each terminal (no next-day target exists).
        6. Remove the intermediate same-day column from the final output.
 
    Parameters
    ----------
    start_date : str
        First date in the dataset (inclusive).
    end_date : str
        Last date in the dataset (inclusive).
    seed : int
        Random seed for full reproducibility.
 
    Returns
    -------
    pd.DataFrame
        Complete dataset with one row per terminal per day.
        Shape: (n_terminals × (n_days - 1), 13 columns).
        The last date in the range is excluded because it has no
        next-day target.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
 
    # Build profile lookup
    profiles = {p.terminal_id: p for p in TERMINAL_PROFILES}
 
    # Generate features for each terminal
    frames = []
    for profile in TERMINAL_PROFILES:
        df_terminal = generate_terminal_features(profile, dates, rng)
        frames.append(df_terminal)
 
    df = pd.concat(frames, ignore_index=True)
 
    # Step 1: Compute same-day dwell (intermediate — what happened today)
    df["_same_day_dwell"] = compute_same_day_dwell(df, profiles, rng)
 
    # Step 2: Sort by terminal then date so the shift operates correctly
    df = df.sort_values(["terminal_id", "date"]).reset_index(drop=True)
 
    # Step 3: Shift same-day dwell forward by 1 day within each terminal.
    # After the shift, row N's target is the dwell that occurred on Day N+1.
    #
    #   Day N features  →  target = Day N+1 actual dwell
    #
    # The last day per terminal gets NaN (no tomorrow to shift from).
    df["target_dwell_hours"] = df.groupby("terminal_id")["_same_day_dwell"].shift(-1)
 
    # Step 4: Drop rows where the target is NaN (last day per terminal)
    df = df.dropna(subset=["target_dwell_hours"]).reset_index(drop=True)
 
    # Step 5: Remove the intermediate column — it should NOT be a feature.
    # If we left it in, the model could trivially learn "tomorrow's dwell ≈
    # today's dwell" and the naive benchmark would be nearly unbeatable.
    df = df.drop(columns=["_same_day_dwell"])
 
    # Final sort: date then terminal for clean output
    df = df.sort_values(["date", "terminal_id"]).reset_index(drop=True)
 
    return df
 
 
# ── CLI entry point ───────────────────────────────────────────────────
 
if __name__ == "__main__":
    import os
 
    print("Generating Phase 1 synthetic dataset...")
    df = generate_dataset()
 
    # Ensure output directory exists
    output_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "synthetic"
    )
    os.makedirs(output_dir, exist_ok=True)
 
    output_path = os.path.join(output_dir, "phase1_terminal_dwell.csv")
    df.to_csv(output_path, index=False)
 
    print(f"Dataset shape: {df.shape}")
    print(f"Date range:    {df['date'].min()} to {df['date'].max()}")
    print(f"Terminals:     {df['terminal_id'].nunique()}")
    print(f"Saved to:      {output_path}")
    print()
    print("Dwell summary by terminal:")
    print(
        df.groupby("terminal_name")["target_dwell_hours"]
        .agg(["mean", "std", "min", "max"])
        .round(1)
        .to_string()
    )
