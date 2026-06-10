from multiprocessing.shared_memory import SharedMemory

for name in [
    "go1_pact_q", "go1_pact_qd", "go1_pact_qd_prev", "go1_pact_mass_mat",
    "go1_pact_wb_dynamics", "go1_pact_wb_contacts", "go1_pact_bias",
    "go1_pact_grf", "go1_pact_acc6d", "go1_pact_dt"
]:
    try:
        SharedMemory(name=name).unlink()
    except FileNotFoundError:
        pass
