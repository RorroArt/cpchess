#!/usr/bin/env python3

from simulator import Simulator
from chess_types import Piece, Row, Column
from typing import Tuple, Optional

class ChessGame:
    def __init__(self):
        self.simulator = Simulator()
        self.piece_symbols = {
            ('K', 'White'): '♔', ('Q', 'White'): '♕', ('R', 'White'): '♖',
            ('B', 'White'): '♗', ('N', 'White'): '♘', ('P', 'White'): '♙',
            ('K', 'Black'): '♚', ('Q', 'Black'): '♛', ('R', 'Black'): '♜',
            ('B', 'Black'): '♝', ('N', 'Black'): '♞', ('P', 'Black'): '♟'
        }
    
    def display_board(self):
        """Display the chess board in a readable format"""
        board = self.simulator.get_board()
        if not board:
            print("Board is being updated...")
            return
        
        # Create 8x8 grid
        grid = [['.' for _ in range(8)] for _ in range(8)]
        
        # Fill grid with pieces
        for piece_pos in board:
            piece, row, col = piece_pos
            row_idx = 8 - int(row)  # Convert '1'-'8' to 7-0
            col_idx = ord(col) - ord('A')  # Convert 'A'-'H' to 0-7
            grid[row_idx][col_idx] = self.piece_symbols.get(piece, '?')
        
        print("\n  A B C D E F G H")
        print("  ________________")
        for i, row in enumerate(grid):
            row_num = 8 - i
            print(f"{row_num}|{' '.join(row)}|{row_num}")
        print("  ________________")
        print("  A B C D E F G H\n")
    
    def parse_move(self, move_str: str) -> Optional[Tuple[Tuple[Row, Column], Tuple[Row, Column]]]:
        """Parse move string like 'e2 e4' into source and destination coordinates"""
        move_str = move_str.strip().lower()
        parts = move_str.split()
        
        if len(parts) != 2:
            return None
        
        src_str, dest_str = parts
        
        if len(src_str) != 2 or len(dest_str) != 2:
            return None
        
        try:
            src_col = src_str[0].upper()
            src_row = src_str[1]
            dest_col = dest_str[0].upper()
            dest_row = dest_str[1]
            
            # Validate coordinates
            if src_col not in 'ABCDEFGH' or dest_col not in 'ABCDEFGH':
                return None
            if src_row not in '12345678' or dest_row not in '12345678':
                return None
            
            return ((src_row, src_col), (dest_row, dest_col))
        except:
            return None
    
    def get_piece_at_position(self, row: Row, col: Column) -> Optional[Piece]:
        """Get the piece at a specific position"""
        board = self.simulator.get_board()
        if not board:
            return None
        
        for piece_pos in board:
            if piece_pos[1] == row and piece_pos[2] == col:
                return piece_pos[0]
        return None
    
    def play(self):
        """Main game loop"""
        print("Welcome to Chess 1v1!")
        print("Enter moves in format: 'e2 e4' (source destination)")
        print("Type 'quit' to exit\n")
        
        while True:
            self.display_board()
            
            current_turn = self.simulator.get_current_turn()
            if current_turn == 'BoardUpdate':
                print("Board is being updated, please wait...")
                continue
            
            print(f"{current_turn}'s turn")
            
            # Get user input
            try:
                move_input = input("Enter your move: ").strip()
            except KeyboardInterrupt:
                print("\nGame interrupted. Goodbye!")
                break
            
            if move_input.lower() == 'quit':
                print("Thanks for playing!")
                break
            
            # Parse move
            parsed_move = self.parse_move(move_input)
            if not parsed_move:
                print("Invalid move format. Use: 'e2 e4'")
                continue
            
            src, dest = parsed_move
            
            # Get piece at source position
            piece = self.get_piece_at_position(src[0], src[1])
            if not piece:
                print(f"No piece at {src[1].lower()}{src[0]}")
                continue
            
            # Check if it's the right color's turn
            if piece[1] != current_turn:
                print(f"It's {current_turn}'s turn, but you selected a {piece[1]} piece")
                continue
            
            # Attempt the move
            success = self.simulator.move_piece(piece, src, dest)
            if not success:
                print("Invalid move! Try again.")
                continue
            
            print(f"Moved {piece[0]} from {src[1].lower()}{src[0]} to {dest[1].lower()}{dest[0]}")

if __name__ == "__main__":
    game = ChessGame()
    game.play()