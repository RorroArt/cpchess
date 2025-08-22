"""
Backend abstract class
"""
from chess_types import Piece, Row, Column, Board, PiecePosition
from typing import Tuple, List

class Backend:
    def __init__(self):
        self._semaphore = 1

    def get_semaphore(self):
        "Semaphore two model who can perform actions"
        return self._semaphore

    def get_piece(self, piece: Piece) -> List[PiecePosition] | PiecePosition:
        "Returns the position(s) of a the requested piece"
        pass

    def get_board(self) -> Board:
        "Returns the position of all Pieces in the board"
        pass

    def move_piece(
        self, 
        piece: Piece, 
        src: Tuple[Row, Column],
        dest: Tuple[Row, Column]
    ) -> bool:
        "Moves piece if valid movement and returns True if success"
        pass