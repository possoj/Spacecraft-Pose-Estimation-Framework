"""
Copyright (c) 2025 Julien Posso
"""


class Jetson:
    """Configuration class for the Nvidia Jetson board."""
    name = 'Jetson'
    ip = '192.168.2.46'  # Jetson's IP address
    port = 22
    username = 'jetson'
    password = 'nvidia'


class Ultra96:
    """Config class that contains specific parameters of Ultra96 board"""
    name = 'Ultra96'
    ip = '192.168.3.1'  # IP of the board""
    port = 22
    username = 'xilinx'
    password = 'xilinx'


class ZCU104:
    """Config class that contains specific parameters of ZCU104 board"""
    name = 'ZCU104'
    ip = '192.168.2.99'   # IP of the board
    port = 22
    username = 'xilinx'
    password = 'xilinx'


def import_board(board_name):
    """Import the config class that contains the board parameters"""
    board_list = ('Ultra96', 'ZCU104', 'Jetson')
    assert board_name in board_list, f'Only support these boards: {board_list}'
    class_obj = globals()[board_name]
    board = class_obj()
    return board
