'''
Simple example pokerbot, written in Python, using the elite Bot base class.
'''
from skeleton.runner import parse_args, run_bot
from skeleton.bot import Bot

class Player(Bot):
    '''
    A pokerbot that inherits from the elite Bot implementation.
    '''
    def __init__(self):
        super().__init__()

if __name__ == '__main__':
    run_bot(Player(), parse_args())
