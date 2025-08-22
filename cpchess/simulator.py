from backend import Backend
from chess_types import Piece, Row, Column, Board, PiecePosition
from typing import Tuple, List

class Simulator(Backend):
    "Simulator Backend for virtual testing"
    
    SEMAPHORE_WHITE_TURN = 1
    SEMAPHORE_BLACK_TURN = 2
    SEMAPHORE_BOARD_UPDATE = 0
    
    def __init__(self):
        super().__init__()
        self._semaphore = self.SEMAPHORE_WHITE_TURN  # White starts
        self._board = self._initialize_board()
    
    def _initialize_board(self) -> Board:
        """Initialize chess board with starting positions"""
        board = []
        
        board.extend([
            (('R', 'White'), '1', 'A'), (('N', 'White'), '1', 'B'),
            (('B', 'White'), '1', 'C'), (('Q', 'White'), '1', 'D'),
            (('K', 'White'), '1', 'E'), (('B', 'White'), '1', 'F'),
            (('N', 'White'), '1', 'G'), (('R', 'White'), '1', 'H')
        ])
        
        for col in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
            board.append((('P', 'White'), '2', col))
        
        board.extend([
            (('R', 'Black'), '8', 'A'), (('N', 'Black'), '8', 'B'),
            (('B', 'Black'), '8', 'C'), (('Q', 'Black'), '8', 'D'),
            (('K', 'Black'), '8', 'E'), (('B', 'Black'), '8', 'F'),
            (('N', 'Black'), '8', 'G'), (('R', 'Black'), '8', 'H')
        ])
        
        for col in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
            board.append((('P', 'Black'), '7', col))
        
        return board
    
    def get_piece(self, piece: Piece) -> List[PiecePosition] | PiecePosition | None:
        """Returns the position(s) of a the requested piece"""
        if self._semaphore == self.SEMAPHORE_BOARD_UPDATE:
            return None
        
        positions = []
        for piece_pos in self._board:
            if piece_pos[0] == piece:
                positions.append(piece_pos)
        
        if len(positions) == 1:
            return positions[0]
        return positions
    
    def get_board(self) -> Board:
        """Returns the position of all Pieces in the board"""
        if self._semaphore == self.SEMAPHORE_BOARD_UPDATE:
            return []
        return self._board.copy()
    
    def move_piece(
        self, 
        piece: Piece, 
        src: Tuple[Row, Column],
        dest: Tuple[Row, Column]
    ) -> bool:
        """Moves piece if valid movement and returns True if success"""
        if self._semaphore == self.SEMAPHORE_BOARD_UPDATE:
            return False
        
        piece_color = piece[1]
        if (piece_color == 'White' and self._semaphore != self.SEMAPHORE_WHITE_TURN) or \
           (piece_color == 'Black' and self._semaphore != self.SEMAPHORE_BLACK_TURN):
            return False
        
        src_row, src_col = src
        dest_row, dest_col = dest
        
        piece_at_src = None
        src_index = None
        for i, piece_pos in enumerate(self._board):
            if piece_pos[1] == src_row and piece_pos[2] == src_col:
                piece_at_src = piece_pos
                src_index = i
                break
        
        if not piece_at_src or piece_at_src[0] != piece:
            return False
        
        dest_occupied = None
        dest_index = None
        for i, piece_pos in enumerate(self._board):
            if piece_pos[1] == dest_row and piece_pos[2] == dest_col:
                dest_occupied = piece_pos
                dest_index = i
                break
        
        if dest_occupied and dest_occupied[0][1] == piece[1]:
            return False
        
        self._semaphore = self.SEMAPHORE_BOARD_UPDATE
        
        self._board[src_index] = (piece, dest_row, dest_col)
        
        if dest_occupied:
            self._board.pop(dest_index)
        
        if piece_color == 'White':
            self._semaphore = self.SEMAPHORE_BLACK_TURN
        else:
            self._semaphore = self.SEMAPHORE_WHITE_TURN
        
        return True
    
    def get_current_turn(self) -> str:
        """Get whose turn it is currently"""
        if self._semaphore == self.SEMAPHORE_WHITE_TURN:
            return 'White'
        elif self._semaphore == self.SEMAPHORE_BLACK_TURN:
            return 'Black'
        else:
            return 'BoardUpdate'
    
    def set_board_update(self) -> bool:
        """Set board to update state, preventing moves"""
        if self._semaphore != self.SEMAPHORE_BOARD_UPDATE:
            self._semaphore = self.SEMAPHORE_BOARD_UPDATE
            return True
        return False
    
    def set_white_turn(self) -> bool:
        """Set turn to white player"""
        if self._semaphore == self.SEMAPHORE_BOARD_UPDATE:
            self._semaphore = self.SEMAPHORE_WHITE_TURN
            return True
        return False
    
    def set_black_turn(self) -> bool:
        """Set turn to black player"""
        if self._semaphore == self.SEMAPHORE_BOARD_UPDATE:
            self._semaphore = self.SEMAPHORE_BLACK_TURN
            return True
        return False