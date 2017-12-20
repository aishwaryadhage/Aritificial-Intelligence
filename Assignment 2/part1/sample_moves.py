sample_string = list("RNBQKBNR"
                     "PPPPPPPP"
                     "........"
                     "........"
                     "....K..."
                     "........"
                     "pppppppp"
                     "rnbqkbnr")


def two_d_nighthawk(piece, r_pos, c_pos):
    if len(piece) != 1:
        return -1
    next_positions = []
    if piece == "K" or piece == "k":
        for row in range(max(0, r_pos - 2), min(r_pos + 3, 8)):
            for col in range(max(0, c_pos - 2), min(c_pos + 3, 8)):
                if (row != r_pos and 0 <= row <= 7) \
                        or (col != c_pos and 0 <= col <= 7):
                    if abs(row - r_pos) + abs(col - c_pos) == 3:
                        next_positions.append([row, col])
    return next_positions


def one_d_parakeet(board, piece, position):
    if len(piece) != 1:
        return -1
    next_positions = []
    if piece == "P" or piece == "p":
        if piece.isupper() and 8 <= position <= 63:
            # attack conditions
            if abs((position + 7) % 8 - position % 8) < 2 and board[
                        position + 7].isalpha() and piece.isupper() == board[
                        position + 7].islower():
                next_positions.append(position + 7)
            if abs((position + 9) % 8 - position % 8) < 2 and board[
                        position + 9].isalpha() and piece.isupper() == board[
                        position + 9].islower():
                next_positions.append(position + 9)
            # progress conditions
            if position / 8 == 1:
                if board[position + 8] == "." and board[position + 16] == ".":
                    next_positions.append(position + 8)
                    next_positions.append(position + 16)
            else:
                if board[position + 8] == ".":
                    next_positions.append(position + 8)
        elif piece.islower() and 0 <= position <= 55:
            # attack conditions
            if abs((position - 7) % 8 - position % 8) < 2 and board[
                        position - 7].isalpha() and piece.isupper() == board[
                        position - 7].islower():
                next_positions.append(position - 7)
            if abs((position - 9) % 8 - position % 8) < 2 and board[
                        position - 9].isalpha() and piece.isupper() == board[
                        position - 9].islower():
                next_positions.append(position - 9)
            # progress conditions
            if position / 8 == 6:
                if board[position - 8] == "." and board[position - 16] == ".":
                    next_positions.append(position - 8)
                    next_positions.append(position - 16)
            else:
                if board[position - 8] == ".":
                    next_positions.append(position - 8)
    return next_positions


def one_d_nighthawk(board, piece, position):
    if len(piece) != 1:
        return -1
    next_positions = []
    if piece == "N" or piece == "n":
        for index in [17, 15, 10, 6]:
            if 0 <= position + index <= 63 \
                    and (board[position + index].islower() == piece.isupper()
                         or board[position + index] == ".") \
                    and abs(position % 8 - abs(position + index) % 8) < 3:
                next_positions.append(position + index)

            if 0 <= position - index <= 63 \
                    and (board[position - index].islower() == piece.isupper()
                         or board[position - index] == ".") \
                    and abs(position % 8 - abs(position - index) % 8) < 3:
                next_positions.append(position - index)
        return next_positions


def one_d_blue_jay(board, piece, position):
    if len(piece) != 1:
        return -1
    next_positions = []
    if piece == "B" or piece == "b":
        for index in range(9, 64, 9):
            if (position + index) % 8 == position % 8:
                break
            if 0 <= position + index <= 63:
                if (board[position + index].islower() == piece.isupper()
                    or board[position + index] == ".") \
                        and ((position + index) % 8 > position % 8):
                    next_positions.append(position + index)
                if board[position + index].isalpha():
                    break
        for index in range(9, 64, 9):
            if (position - index) % 8 == position % 8:
                break
            if 0 <= position - index <= 63:
                if (board[position - index].islower() == piece.isupper()
                    or board[position - index] == ".") \
                        and ((position - index) % 8 < position % 8):
                    next_positions.append(position - index)
                if board[position - index].isalpha():
                    break
        for index in range(7, 64, 7):
            if (position + index) % 8 == position % 8:
                break
            if 0 <= position + index <= 63:
                if (board[position + index].islower() == piece.isupper()
                    or board[position + index] == ".") \
                        and ((position + index) % 8 < position % 8):
                    next_positions.append(position + index)
                if board[position + index].isalpha():
                    break
        for index in range(7, 64, 7):
            if (position - index) % 8 == position % 8:
                break
            if 0 <= position - index <= 63:
                if (board[position - index].islower() == piece.isupper()
                    or board[position - index] == ".") \
                        and ((position - index) % 8 > position % 8):
                    next_positions.append(position - index)
                if board[position - index].isalpha():
                    break
        return next_positions


def one_d_robin(board, piece, position):
    if len(piece) != 1:
        return -1
    next_positions = []
    if piece == "R" or piece == "r":
        for index in range(8, 64, 8):
            if 0 <= position - index <= 63:
                if (board[position - index].islower() == piece.isupper()
                    or board[position - index] == ".") \
                        and ((position - index) % 8 == position % 8):
                    next_positions.append(position - index)
                if board[position - index].isalpha():
                    break
        for index in range(8, 64, 8):
            if 0 <= position + index <= 63:
                if (board[position + index].islower() == piece.isupper()
                    or board[position + index] == ".") \
                        and ((position + index) % 8 == position % 8):
                    next_positions.append(position + index)
                if board[position + index].isalpha():
                    break
        for index in range(1, 8):
            if 0 <= position + index <= 63:
                if (board[position + index].islower() == piece.isupper()
                    or board[position + index] == ".") \
                        and ((position + index) / 8 == position / 8):
                    next_positions.append(position + index)
                if board[position + index].isalpha():
                    break
        for index in range(1, 8):
            if 0 <= position - index <= 63:
                if (board[position - index].islower() == piece.isupper()
                    or board[position - index] == ".") \
                        and ((position - index) / 8 == position / 8):
                    next_positions.append(position - index)
                if board[position - index].isalpha():
                    break
    return next_positions


def one_d_quetzal(board, piece, position):
    if len(piece) != 1:
        return -1
    next_positions = []
    if piece == "Q" or piece == "q":
        for index in range(8, 64, 8):
            if 0 <= position - index <= 63:
                if (board[position - index].islower() == piece.isupper()
                    or board[position - index] == ".") \
                        and ((position - index) % 8 == position % 8):
                    next_positions.append(position - index)
                if board[position - index].isalpha():
                    break
        for index in range(8, 64, 8):
            if 0 <= position + index <= 63:
                if (board[position + index].islower() == piece.isupper()
                    or board[position + index] == ".") \
                        and ((position + index) % 8 == position % 8):
                    next_positions.append(position + index)
                if board[position + index].isalpha():
                    break
        for index in range(1, 8):
            if 0 <= position + index <= 63:
                if (board[position + index].islower() == piece.isupper()
                    or board[position + index] == ".") \
                        and ((position + index) / 8 == position / 8):
                    next_positions.append(position + index)
                if board[position + index].isalpha():
                    break
        for index in range(1, 8):
            if 0 <= position - index <= 63:
                if (board[position - index].islower() == piece.isupper()
                    or board[position - index] == ".") \
                        and ((position - index) / 8 == position / 8):
                    next_positions.append(position - index)
                if board[position - index].isalpha():
                    break
        for index in range(9, 64, 9):
            if (position + index) % 8 == position % 8:
                break
            if 0 <= position + index <= 63:
                if (board[position + index].islower() == piece.isupper()
                    or board[position + index] == ".") \
                        and ((position + index) % 8 > position % 8):
                    next_positions.append(position + index)
                if board[position + index].isalpha():
                    break
        for index in range(9, 64, 9):
            if (position - index) % 8 == position % 8:
                break
            if 0 <= position - index <= 63:
                if (board[position - index].islower() == piece.isupper()
                    or board[position - index] == ".") \
                        and ((position - index) % 8 < position % 8):
                    next_positions.append(position - index)
                if board[position - index].isalpha():
                    break
        for index in range(7, 64, 7):
            if (position + index) % 8 == position % 8:
                break
            if 0 <= position + index <= 63:
                if (board[position + index].islower() == piece.isupper()
                    or board[position + index] == ".") \
                        and ((position + index) % 8 < position % 8):
                    next_positions.append(position + index)
                if board[position + index].isalpha():
                    break
        for index in range(7, 64, 7):
            if (position - index) % 8 == position % 8:
                break
            if 0 <= position - index <= 63:
                if (board[position - index].islower() == piece.isupper()
                    or board[position - index] == ".") \
                        and ((position - index) % 8 > position % 8):
                    next_positions.append(position - index)
                if board[position - index].isalpha():
                    break
    return next_positions


def one_d_kingfisher(board, piece, position):
    if len(piece) != 1:
        return -1
    next_positions = []
    if piece == "K" or piece == "k":
        for index in [1, 7, 8, 9]:
            if 0 <= position - index <= 63:
                if (board[position - index].islower() == piece.isupper()
                    or board[position - index] == ".") \
                        and abs((position - index) % 8 - position % 8) < 2:
                    next_positions.append(position - index)
            if 0 <= position + index <= 63:
                if (board[position + index].islower() == piece.isupper()
                    or board[position + index] == ".") \
                        and abs((position + index) % 8 - position % 8) < 2:
                    next_positions.append(position + index)
    return next_positions


# one_d_kingfisher(str(sys.argv[1]), str(sys.argv[2]), int(sys.argv[3]))
print one_d_kingfisher(sample_string, "K", 57)
