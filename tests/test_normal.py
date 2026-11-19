from aerorsi.repair.normal import select_states


def row(step, distance, stop=False, valid=True, terminal=""):
    return {"step": step, "distance_before_m": distance, "stop": stop, "valid": valid, "terminal": terminal}


def test_three_normal_states_of_a_successful_flight():
    trace = [row(1, 90, stop=True), row(2, 80), row(3, 45), row(4, 33), row(5, 28), row(6, 15, stop=True, terminal="success")]
    assert select_states(trace) == [(2, ["early"]), (5, ["approach"]), (6, ["successful_stop"])]


def test_failed_flight_has_none():
    assert select_states([row(1, 90), row(2, 80, terminal="collision")]) == []


def test_roles_share_a_step():
    trace = [row(1, 31), row(2, 18, stop=True, terminal="success")]
    assert select_states(trace) == [(1, ["early", "approach"]), (2, ["successful_stop"])]
