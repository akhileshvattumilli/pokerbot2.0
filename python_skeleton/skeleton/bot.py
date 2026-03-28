'''
Simple base class that Bot implementations should inherit from.
'''

class Bot():
    '''
    Base bot class. Override these methods in your Player class.
    '''

    def handle_new_round(self, game_state, round_state, active):
        '''
        Called when a new round starts. Override in Player.
        '''
        pass

    def handle_round_over(self, game_state, terminal_state, active):
        '''
        Called when a round ends. Override in Player.
        '''
        pass

    def get_action(self, game_state, round_state, active):
        '''
        Called to get an action for the current state. Override in Player.
        '''
        raise NotImplementedError('Implement get_action')
