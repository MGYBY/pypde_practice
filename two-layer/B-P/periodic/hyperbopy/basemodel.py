class BaseModel:

    # Default physical parameters
    GRAVITATIONAL_CONSTANT = 9.81  # [m2/s]
    DENSITY_RATIO = 0.95  # [-]
    EXTERNAL_PRESSURE = 0  # [Pa]
    NONHYDRO_COEFF_M = 0  # [-]
    NONHYDRO_COEFF_N = 0  # [-]
    DEN_RATIO = 0.9999999
    VIS_RATIO = 1.0
    GRAV_CONSTANT = 0.80
    H_RATIO = 1.00

    # Default numerical parameters
    THETA = 1  # Numerical dissipation parameter
    DT_FACT = 0.5  # factor to use in front of the time step
    MIN_VAR = 1e-10  # Minimal value to use to avoid using 0 for a variable
    EPSILON = MIN_VAR**4  # regularization used to avoid division by zero
