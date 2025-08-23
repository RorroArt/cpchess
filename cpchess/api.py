""" Queen vs Knight API """
from simulator import Simulator


sim = Simulator()

def init_board():
    """
    Reinitializes the board only with white queen and black knight
    """
    sim.set_board_update()
    
    sim._board = []
    
    sim._board.append((('Q', 'White'), '1', 'D'))
    sim._board.append((('N', 'Black'), '5', 'B'))
    
    sim.set_white_turn()


def check_win():
    """ 
    If there is one piece and is queen return true
    else return false
    """
    board = sim.get_board()
    if not board:
        return False
    
    if len(board) == 1:
        remaining_piece = board[0]
        if remaining_piece[0] == ('Q', 'White'):
            return True
    
    return False

def check_loss():
    """
    If there is one piece and is knight return True
    """
    board = sim.get_board()
    if not board:
        return False
    
    if len(board) == 1:
        remaining_piece = board[0]
        if remaining_piece[0] == ('N', 'Black'):
            return True
    
    return False


def my_turn():
    """
    Return true if white's turn
    else ask the other player for its turn
    """
    # print('Trapped ')
    current_turn = sim.get_current_turn()
    # print(current_turn)
    if current_turn == 'White':
        return True
    else:
        
        queen_pos = get_queen()
        knight_pos = get_knight()
        if not knight_pos: return True 
        print(f"Queen is at {queen_pos}, Knight is at {knight_pos}")
        
        knight_dest = input('Move knight to: ')
        
        if len(knight_dest) == 2:
            dest_col = knight_dest[0].upper()
            dest_row = knight_dest[1]
            
            if dest_col in 'ABCDEFGH' and dest_row in '12345678':
                knight_result = sim.get_piece(('N', 'Black'))
                if knight_result:
                    sim.move_piece(('N', 'Black'), (knight_result[1], knight_result[2]), (dest_row, dest_col))
        
        assert check_loss, "Model Lost :("
    return True


def move_queen(dest: str) -> bool:
    """
    Moves the queen to destination (e.g., 'e4')
    """
    # print('Moving queen')
    queen_pos = get_queen()
    if not queen_pos:
        return False
    
    if len(dest) != 2:
        return False
    
    dest_col = dest[0].upper()
    dest_row = dest[1]
    
    if dest_col not in 'ABCDEFGH' or dest_row not in '12345678':
        # print('here')
        return False
    # Parse the chess notation string (e.g., "d1")
    src_row = int(queen_pos[1])
    src_col = ord(queen_pos[0].upper()) - ord('A')
    dest_row_int = int(dest_row)
    dest_col_int = ord(dest_col) - ord('A')
    
    row_diff = abs(dest_row_int - src_row)
    col_diff = abs(dest_col_int - src_col)
    
    if not (row_diff == 0 or col_diff == 0 or row_diff == col_diff):
        # print('there')
        return False
    
    # Get the queen tuple for move_piece call
    queen_tuple = sim.get_piece(('Q', 'White'))
    r = sim.move_piece(('Q', 'White'), (queen_tuple[1], queen_tuple[2]), (dest_row, dest_col))
    # print(f'The hell {r}')
    return r


def get_queen():
    """
    Returns queen position in chess notation (e.g., 'e2') or None
    """
    result = sim.get_piece(('Q', 'White'))
    if isinstance(result, tuple) and len(result) == 3:
        col = result[2].lower() 
        row = result[1] 
        return f"{col}{row}"
    return None


def get_knight():
    """
    Returns knight position in chess notation (e.g., 'b8') or None
    """
    result = sim.get_piece(('N', 'Black'))
    if isinstance(result, tuple) and len(result) == 3:
        col = result[2].lower()  
        row = result[1] 
        return f"{col}{row}"
    return None


def policy():
    """Move the queen until it captures the knight."""
    # helper: convert 'e4' → (file, rank)  (0‑indexed, a1 = (0,0))
    def square_to_coord(sq):
        file = ord(sq[0]) - ord('a')
        rank = int(sq[1]) - 1
        return file, rank

    def coord_to_square(x, y):
        return chr(ord('a') + x) + str(y + 1)

    # king‑like distance (queen moves along rank/file/diag)
    def king_distance(a, b):
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

    # all 8 directions a queen can move
    DIRECTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1),
                  (1, 1), (1, -1), (-1, 1), (-1, -1)]

    while my_turn():
        if check_win():
            break

        # if not my_turn():
        #     continue

        q_sq = get_queen()
        k_sq = get_knight()

        qx, qy = square_to_coord(q_sq)
        kx, ky = square_to_coord(k_sq)

        # --------------------------------------------------
        # 1) Can capture straight away?
        # --------------------------------------------------
        if qx == kx or qy == ky or abs(qx - kx) == abs(qy - ky):
            move_queen(k_sq)          # capture
            continue

        # --------------------------------------------------
        # 2) Otherwise move to a square that gets strictly
        #    closer (by king‑like distance) to the knight.
        #    Prefer moves that maximise the reduction.
        # --------------------------------------------------
        best_dest = None
        best_dist = king_distance((qx, qy), (kx, ky))

        for dx, dy in DIRECTIONS:
            nx, ny = qx, qy
            while True:
                nx += dx
                ny += dy
                if not (0 <= nx < 8 and 0 <= ny < 8):
                    break
                # a legal queen destination
                new_dist = king_distance((nx, ny), (kx, ky))
                if new_dist < best_dist:
                    best_dist = new_dist
                    best_dest = (nx, ny)

        # Usually best_dest will not be None (queen can always move closer)
        if best_dest is not None:
            dest_sq = coord_to_square(*best_dest)
            move_queen(dest_sq)
        else:
            # Fallback: move one square towards the knight (should never hit here)
            dx = 1 if kx > qx else (-1 if kx < qx else 0)
            dy = 1 if ky > qy else (-1 if ky < qy else 0)
            if dx == 0 and dy == 0:
                # already on the knight – capture
                move_queen(k_sq)
            else:
                # pick the first legal square in the direction of the knight
                nx, ny = qx + dx, qy + dy
                while not (0 <= nx < 8 and 0 <= ny < 8):
                    nx -= dx
                    ny -= dy
                move_queen(coord_to_square(nx, ny))


if __name__ == '__main__':
    init_board()
    policy()

    print(f'Model WON! final queen position {get_queen()}')