import math
import pandas as pd
import re

# ==============================================================================
# CONFIGURATION
# ==============================================================================
FLIGHT_TIME_LIMIT_S = 300
COURSE_ALTITUDE_FT = 75
LOAD_FACTOR_N_MAX = 3.0
RUNWAY_LENGTH_FT = 1000
BASE_AIRFRAME_WEIGHT_LB = 28.0
CSV_FILENAME = 'tmotor_data.csv'

# Altitudes to sweep (feet)
ALTITUDES_FT = [0, 1300, 2600, 4900]

# Output CSV path
OUTPUT_CSV = 'Motor_Prop_Combinations.csv'

# Aircraft geometry/aero
WING_AREA_FT2 = 9.6681
ASPECT_RATIO = 3.6
OSWALD_E = 0.80
CD0 = 0.039
CL_TAKEOFF = 1.4          
CL_MAX = 1.68
MU_ROLL = 0.02
CL_GROUND = 0.5

# Propulsion/energy modeling
ETA_PROPULSIVE = 0.65      # electrical -> shaft -> propulsive efficiency (flight)
BATTERY_DOD = 0.80         # usable fraction
BATTERY_ENERGY_DENSITY_Wh_per_kg = 160.0
CLAMP_PITCH_FACTOR = 0.95  # allow a bit more headroom to avoid false "no solution" at high n
STRAIGHT_FRAC_VMAX = 0.90  # cruise fraction of Vmax
TURN_SPEED_MARGIN = 1.20   # require V_turn >= 1.2 * Vstall(n_turn)
LIFTOFF_SAFETY = 1.20      # V_lof = 1.2 * Vstall
MTOW_LB = 31.23


# Debug
DEBUG_SPEED = False  # set True for speed error diagnostics

# ==============================================================================
# CORE MODELS
# ==============================================================================

class Aircraft:
    def __init__(self, wing_area_ft2, cl_takeoff, cl_max, cd_zero,
                 aspect_ratio, friction_coeff, oswald_e=0.8, cl_ground=0.5):
        self.wing_area = wing_area_ft2
        self.cl_takeoff = cl_takeoff
        self.cl_max = cl_max
        self.cd_zero = cd_zero
        self.aspect_ratio = aspect_ratio
        self.mu = friction_coeff
        self.e = oswald_e
        self.cl_ground = cl_ground

def isa_density_slug_ft3_from_alt_ft(alt_ft: float, deltaT_C: float = 0.0) -> float:
    """ISA density in slug/ft^3 from altitude (ft)."""
    T0 = 288.15 + deltaT_C  # K
    p0 = 101325.0
    L  = 0.0065
    R  = 287.058
    g  = 9.80665
    alt_m = alt_ft * 0.3048
    if alt_m < 11000.0:
        T = T0 - L * alt_m
        p = p0 * (T / T0) ** (g / (R * L))
        rho_kg_m3 = p / (R * T)
    else:
        T = T0 - L * 11000.0
        p = p0 * (T / T0) ** (g / (R * L))
        rho_kg_m3 = p / (R * T)
    return rho_kg_m3 * 0.0019403203  # kg/m^3 -> slug/ft^3

def induced_factor(e: float, AR: float) -> float:
    return 1.0 / (math.pi * e * AR)

def get_prop_pitch_in(prop_str) -> float:
    """Parse '13x8', 'G30*10.5', 'VZ29×11' etc. Return pitch (inches) or 0."""
    s = str(prop_str).replace("×", "x")
    m = re.search(r"(?i)(\d+(?:\.\d+)?)\s*[x\*]\s*(\d+(?:\.\d+)?)", s)
    return float(m.group(2)) if m else 0.0

def stall_speed_fps(W_lbf: float, rho: float, S: float, CLmax: float, n: float = 1.0) -> float:
    return math.sqrt( (2.0 * n * W_lbf) / (rho * S * CLmax) )

def D_lbf(V_fps: float, rho: float, W_lbf: float, S: float, CD0: float, k: float, n: float = 1.0) -> float:
    q = 0.5 * rho * V_fps * V_fps
    CL = (n * W_lbf) / max(q * S, 1e-9)
    CD = CD0 + k * CL * CL
    return q * S * CD

def T_available_lbf(V_fps: float, T0_lbf: float, V_pitch_fps: float) -> float:
    """Linear decay to zero at pitch speed (simple prop model)."""
    if V_fps >= V_pitch_fps:
        return 0.0
    return T0_lbf * (1.0 - V_fps / max(V_pitch_fps, 1e-6))

def solve_max_speed_fps(W_lbf: float, rho: float, S: float, CD0: float, k: float,
                        T0_lbf: float, V_pitch_fps: float, n: float,
                        clamp_pitch_factor: float = CLAMP_PITCH_FACTOR,
                        debug: bool = DEBUG_SPEED) -> float:
    """Find max V where T(V) >= D(V), from Vstall(n) up to clamp*V_pitch."""
    Vstall = stall_speed_fps(W_lbf, rho, S, CL_MAX, n)
    Vcap = clamp_pitch_factor * V_pitch_fps
    if Vcap <= Vstall:
        if debug:
            print(f"[speed] Vcap({Vcap*0.6818:.1f} mph) <= Vstall(n={n:.1f})({Vstall*0.6818:.1f} mph)")
        return 0.0
    # quick infeasibility at stall
    if T_available_lbf(Vstall, T0_lbf, V_pitch_fps) < D_lbf(Vstall, rho, W_lbf, S, CD0, k, n):
        if debug:
            Tst = T_available_lbf(Vstall, T0_lbf, V_pitch_fps)
            Dst = D_lbf(Vstall, rho, W_lbf, S, CD0, k, n)
            print(f"[speed] T({Vstall*0.6818:.1f} mph) < D at stall (T={Tst:.2f}, D={Dst:.2f}) for n={n:.1f}")
        return 0.0
    last_ok = 0.0
    start = int(math.ceil(Vstall))
    end = int(Vcap) + 1
    for V in range(start, end):
        T = T_available_lbf(V, T0_lbf, V_pitch_fps)
        D = D_lbf(V, rho, W_lbf, S, CD0, k, n)
        if T >= D:
            last_ok = float(V)
        else:
            break
    if last_ok == 0.0 and debug:
        Vmid = 0.5 * (Vstall + Vcap)
        print(f"[speed] no T>=D up to cap ({Vcap*0.6818:.1f} mph). "
              f"Mid {Vmid*0.6818:.1f} mph: T={T_available_lbf(Vmid,T0_lbf,V_pitch_fps):.2f}, "
              f"D={D_lbf(Vmid,rho,W_lbf,S,CD0,k,n):.2f} (n={n:.1f})")
    return last_ok

def calculate_takeoff_performance(aircraft: Aircraft, total_weight_lb: float,
                                  static_thrust_lbf: float, air_density_slug_ft3: float):
    """Ground-roll using average forces; liftoff speed = 1.2 * Vstall(CLmax)."""
    mass_slug = total_weight_lb / 32.174
    k = induced_factor(aircraft.e, aircraft.aspect_ratio)
    v_stall = stall_speed_fps(total_weight_lb, air_density_slug_ft3, aircraft.wing_area, aircraft.cl_max, n=1.0)
    v_takeoff = LIFTOFF_SAFETY * v_stall
    cd_ground = aircraft.cd_zero + k * (aircraft.cl_ground ** 2)
    v_avg = 0.7 * v_takeoff
    q_avg = 0.5 * air_density_slug_ft3 * v_avg**2
    lift_avg = q_avg * aircraft.wing_area * aircraft.cl_ground
    drag_avg = q_avg * aircraft.wing_area * cd_ground
    friction_force_avg = aircraft.mu * max(0, total_weight_lb - lift_avg)
    thrust_corrected = static_thrust_lbf * (air_density_slug_ft3 / 0.002377)
    net_force_avg = thrust_corrected - drag_avg - friction_force_avg
    if net_force_avg <= 0:
        return {"error": "Insufficient thrust."}
    avg_acceleration = net_force_avg / mass_slug
    takeoff_distance = v_takeoff**2 / (2 * avg_acceleration)
    takeoff_time = v_takeoff / avg_acceleration
    return {"takeoff_distance_ft": takeoff_distance, "takeoff_time_s": takeoff_time}

def electrical_power_required_W(D_lbf: float, V_fps: float, eta_propulsive: float = ETA_PROPULSIVE) -> float:
    """Convert aero power D*V (lbf*ft/s) to electrical Watts via η."""
    return (D_lbf * V_fps) / max(eta_propulsive, 1e-6) * 1.35581795

# ==============================================================================
# ADAPTIVE TURN LOAD FACTOR (n_turn ≤ 3)
# ==============================================================================

def find_feasible_turn(W_lbf: float, rho: float, S: float, CD0: float, k: float,
                       T0_lbf: float, Vpitch_fps: float, n_max: float = LOAD_FACTOR_N_MAX):
    """
    Find the highest load factor n_turn <= n_max for which:
      - a max speed Vmax(n_turn) exists (T>=D),
      - AND there exists a sustainable turn speed >= 1.2*Vstall(n_turn).
    Returns (n_turn, V_turn), or (None, 0) if even n=1 fails.
    """
    # Try descending n: 3.0, 2.8, ..., 1.0
    for n in [round(x, 2) for x in list(frange(n_max, 1.0 - 1e-6, -0.1))]:
        Vmax_n = solve_max_speed_fps(W_lbf, rho, S, CD0, k, T0_lbf, Vpitch_fps, n=n)
        if Vmax_n <= 0:
            continue
        Vstall_n = stall_speed_fps(W_lbf, rho, S, CL_MAX, n=n)
        V_req = TURN_SPEED_MARGIN * Vstall_n
        if Vmax_n >= V_req:
            # pick a realistic turn speed inside the feasible band
            V_turn = max(0.9 * Vmax_n, V_req)
            return n, V_turn
    # Try n=1 as a last resort (should rarely fail if straight flight exists)
    Vmax_1 = solve_max_speed_fps(W_lbf, rho, S, CD0, k, T0_lbf, Vpitch_fps, n=1.0)
    if Vmax_1 > 0:
        Vstall_1 = stall_speed_fps(W_lbf, rho, S, CL_MAX, n=1.0)
        V_turn = max(0.9 * Vmax_1, TURN_SPEED_MARGIN * Vstall_1)
        return 1.0, min(V_turn, Vmax_1)
    return None, 0.0

def frange(start, stop, step):
    x = start
    if step > 0:
        while x < stop:
            yield x
            x += step
    else:
        while x > stop:
            yield x
            x += step

# ==============================================================================
# BATTERY SIZING WITH MTOW (ITERATIVE)
# ==============================================================================

def motor_prop_weight_lb(row) -> float:
    motor_g = float(row.get("Weight (g)", 0.0))
    return motor_g * 0.00220462


def size_battery_and_laps(row, aircraft: Aircraft, rho: float,
                          base_airframe_lb: float, mtow_lb: float) -> dict:
    """
    Iterate battery sizing using drag-based power and MTOW constraint.
    Uses adaptive n_turn <= 3 for turns.
    """
    thrust_g = float(row["Thrust (g)"])
    rpm = float(row["RPM"])
    Pin_W = float(row["Input Power (W)"])
    pitch_in = get_prop_pitch_in(row["Prop"])
    if pitch_in <= 0 or rpm <= 0:
        return {"feasible": False, "reason": "No pitch/RPM parsed"}

    # Static thrust scaled to site density for flight calculations
    T0_lbf = (thrust_g / 453.592) * (rho / 0.002377)
    Vpitch_fps = (pitch_in * rpm) / 720.0
    k = induced_factor(aircraft.e, aircraft.aspect_ratio)
    S = aircraft.wing_area

    mprop_lb = motor_prop_weight_lb(row)
    battery_lb = 0.5
    total_lb = base_airframe_lb + mprop_lb + battery_lb

    best = None
    for _ in range(8):
        W = total_lb

        # Straight (n=1): Vmax and cruise
        Vmax_fps = solve_max_speed_fps(W, rho, S, CD0, k, T0_lbf, Vpitch_fps, n=1.0)
        if Vmax_fps <= 0:
            return {"feasible": False, "reason": "No T>=D (n=1)"}
        V_cruise = STRAIGHT_FRAC_VMAX * Vmax_fps

        # Turn: find highest feasible n_turn ≤ 3
        n_turn, V_turn = find_feasible_turn(W, rho, S, CD0, k, T0_lbf, Vpitch_fps, n_max=LOAD_FACTOR_N_MAX)
        if n_turn is None or V_turn <= 0:
            return {"feasible": False, "reason": "No feasible turn n<=3"}

        # Turn geometry & times
        g = 32.174
        R_turn = V_turn**2 / (g * math.sqrt(max(n_turn**2 - 1.0, 1e-6)))
        t_180 = (math.pi * R_turn) / V_turn
        t_360 = 2 * t_180

        t_straights = (RUNWAY_LENGTH_FT/2 + RUNWAY_LENGTH_FT + RUNWAY_LENGTH_FT/2) / V_cruise
        t_arcs = 2 * t_180 + t_360
        t_lap = t_straights + t_arcs

        # Takeoff at this weight
        takeoff = calculate_takeoff_performance(aircraft, W, thrust_g/453.592, rho)
        t_to = takeoff.get("takeoff_time_s", 30.0)
        s_to = takeoff.get("takeoff_distance_ft", float("inf"))
        meets_30ft = isinstance(s_to, (int, float)) and (s_to <= 30.0)

        # First lap time (add a small climb)
        t_climb = COURSE_ALTITUDE_FT / 15.0
        t_first = t_to + t_climb + t_lap

        # Time-limited laps
        if t_first >= FLIGHT_TIME_LIMIT_S:
            N_time = 0
        else:
            N_time = 1 + math.floor((FLIGHT_TIME_LIMIT_S - t_first) / t_lap)

        # Drag-based electrical power
        D_c = D_lbf(V_cruise, rho, W, S, CD0, k, n=1.0)
        D_t = D_lbf(V_turn, rho, W, S, CD0, k, n=n_turn)
        P_cruise_W = electrical_power_required_W(D_c, V_cruise, ETA_PROPULSIVE)
        P_turn_W   = electrical_power_required_W(D_t, V_turn,   ETA_PROPULSIVE)

        # Energy (Wh)
        E_takeoff_Wh = (Pin_W * t_to) / 3600.0
        E_lap_Wh = (P_cruise_W * t_straights + P_turn_W * t_arcs) / 3600.0

        # Pick largest N <= N_time that fits MTOW with sized battery
        picked = None
        for N in range(N_time, -1, -1):
            E_total_Wh = E_takeoff_Wh + N * E_lap_Wh
            Wh_needed = E_total_Wh / max(BATTERY_DOD, 1e-6)
            battery_kg = Wh_needed / max(BATTERY_ENERGY_DENSITY_Wh_per_kg, 1e-6)
            battery_try_lb = battery_kg * 2.20462
            total_try_lb = base_airframe_lb + mprop_lb + battery_try_lb
            if total_try_lb <= MTOW_LB:
                picked = (battery_try_lb, total_try_lb, N, E_total_Wh)
                break

        if picked is None:
            return {
                "feasible": False, "reason": "MTOW exceeded for any laps",
                "Motor_Prop_lb": mprop_lb, "Battery_lb": 0.0, "Total_lb": base_airframe_lb + mprop_lb
            }

        new_batt_lb, new_total_lb, N_ok, E_tot = picked
        best = dict(
            feasible=True, reason="OK",
            Motor_Prop_lb=mprop_lb, Battery_lb=new_batt_lb, Total_lb=new_total_lb,
            Meets_30ft=meets_30ft, S_TO_ft=s_to, T_TO_s=t_to,
            V_stall_mph=stall_speed_fps(new_total_lb, rho, S, CL_MAX, n=1.0) * 0.681818,
            V_max_mph=Vmax_fps * 0.681818, V_cruise_mph=V_cruise * 0.681818,
            n_turn=n_turn, V_turn_mph=V_turn * 0.681818, R_turn_ft=R_turn,
            t_lap_s=t_lap, N_laps_5min=N_ok, Mission_energy_Wh=E_tot,
            V_pitch_mph=Vpitch_fps * 0.681818, t_first=t_first
        )
        if abs(new_batt_lb - battery_lb) < 0.05:
            break
        battery_lb = new_batt_lb
        total_lb = new_total_lb

    return best if best is not None else {"feasible": False, "reason": "No convergence"}

# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == '__main__':
    # Aircraft object
    aircraft = Aircraft(
        wing_area_ft2=WING_AREA_FT2, cl_takeoff=CL_TAKEOFF, cl_max=CL_MAX,
        cd_zero=CD0, aspect_ratio=ASPECT_RATIO, friction_coeff=MU_ROLL,
        oswald_e=OSWALD_E, cl_ground=CL_GROUND
    )

    # Load data
    try:
        motor_df = pd.read_csv(CSV_FILENAME)
        for col in ['Thrust (g)', 'Weight (g)', 'Input Power (W)', 'RPM', 'Voltage (V)']:
            if col in motor_df.columns:
                motor_df[col] = pd.to_numeric(motor_df[col], errors='coerce')
        motor_df.dropna(subset=['Thrust (g)', 'Weight (g)', 'Input Power (W)', 'RPM', 'Prop'], inplace=True)
    except FileNotFoundError:
        print(f"Error: '{CSV_FILENAME}' not found.")
        raise SystemExit(1)

    rows_out = []

    for alt_ft in ALTITUDES_FT:
        rho = isa_density_slug_ft3_from_alt_ft(alt_ft)

        for _, combo in motor_df.iterrows():
            res = size_battery_and_laps(
                combo, aircraft, rho,
                base_airframe_lb=BASE_AIRFRAME_WEIGHT_LB,
                mtow_lb=MTOW_LB
            )

            out_row = {
                "Altitude_ft": alt_ft,
                "Density_slug_ft3": rho,
                "Motor": combo.get("Motor", ""),
                "Prop": combo.get("Prop", ""),
                "Voltage_V": combo.get("Voltage (V)", ""),
                "Thrust_g": combo.get("Thrust (g)", ""),
                "RPM": combo.get("RPM", ""),
                "InputPower_W": combo.get("Input Power (W)", ""),
                "Reason": res.get("reason", ""),
                "Feasible": res.get("feasible", False),
                "Meets_30ft": res.get("Meets_30ft", False),
                "S_TO_ft": res.get("S_TO_ft", None),
                "T_TO_s": res.get("T_TO_s", None),
                "V_pitch_mph": res.get("V_pitch_mph", None),
                "V_stall_mph": res.get("V_stall_mph", None),
                "V_max_mph": res.get("V_max_mph", None),
                "V_cruise_mph": res.get("V_cruise_mph", None),
                "n_turn": res.get("n_turn", None),
                "V_turn_mph": res.get("V_turn_mph", None),
                "R_turn_ft": res.get("R_turn_ft", None),
                "Lap_time_s": res.get("t_lap_s", None),
                "First_lap_s": res.get("t_first", None),
                "N_laps_5min": res.get("N_laps_5min", 0),
                "Mission_energy_Wh": res.get("Mission_energy_Wh", None),
                "MotorProp_lb": res.get("Motor+Prop_lb", None),
                "Battery_lb": res.get("Battery_lb", None),
                "Total_lb": res.get("Total_lb", None),
            }
            rows_out.append(out_row)

    pd.DataFrame(rows_out).to_csv(OUTPUT_CSV, index=False)
    print(f"Saved results to: {OUTPUT_CSV}")
