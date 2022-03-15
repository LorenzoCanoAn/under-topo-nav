class StateMachine:
    states = {}

    def __init__(self):
        self.current_state_name = "start_state"
        self.current_state = self.states[self.current_state_name]
        self.state_history = []
        self.transition_pending = False

    def run_as_independent_loop(self):
        while not self.start_signal():
            pass
        while not self.current_state.is_final :
            self.do_one_step()
        self.current_state.state_action()

    def do_one_step(self):
        self.update_transition_variables()
        self.current_state.run()

    def change_state(self, new_state_name):
        if self.transition_pending:
            self.current_state_name = new_state_name
            self.current_state = self.states[self.current_state_name]
            self.current_state.when_transitioned_do()
            self.transition_pending = False


class State:
    name = "state"
    is_final = False
    decission_variables = []

    def __init__(self, sm: StateMachine):
        self.sm = sm
        self.next_state_name = self.name

    def run(self):
        self.state_action()
        self.do_transition()

    def do_transition(self):
        self.sm.change_state(self.next_state_name)

    def set_transition(self, next_state_name):
        self.next_state_name = next_state_name
        self.sm.transition_pending = True

    def when_transitioned_do(self):
        pass


class DecissionVariable:
    def __init__(self, update_method):
        self.update_method = update_method

    def update(self):
        self.value = self.update_method()

