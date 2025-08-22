# from dataclasses import dataclass
from typing import Literal, List, Tuple

# NAMING

PieceName = Literal['K', 'Q', 'R', 'B', 'N', 'P']
Colors = Literal['White', 'Black']
Piece = Tuple[PieceName, Colors]
Row = Literal['1', '2', '3', '4', '5', '6', '7', '8']
Column = Literal['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']


# Board
PiecePosition = Tuple[Piece, Row, Column]
Board = List[PiecePosition]

